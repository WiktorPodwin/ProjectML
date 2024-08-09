import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from steps import ingest_df, clean_df, train_model, evaluate_model, data_transform
from src import deployment_trigger_prepare, predictor_prepare
from docker_services import MongoOperations
from sklearn.base import ClassifierMixin
import logging 

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """
    Deployment trigger config
    """
    min_accuracy: float = 0.6


@step
def deployment_trigger(config: DeploymentTriggerConfig) -> bool:
    """
    Implements a single model deployment trigger that looks at the input model accuracy and decides, if it is good enough to deploy or not
    """
    accuracy = deployment_trigger_prepare()
    return accuracy >= config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """
    MLFlow deployment getter parameters
    
    Attributes:
        pipeline_name: name of the pipeline that deployed the MLFlow prediction server
        step_name: the name of the step that deployed the MLFlow prediction server
        running: when this flag is set, the step returns only a running service
        model_name: the name of the model that is deployed
        """
    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model"
) -> MLFlowDeploymentService:
    """
    Get the prediction service started by the deployment pipeline.
    
    Args:
        pipeline_name: name of the pipeline that deployed the MLFlow prediction server
        step_name: the name of the step that deployed the MLFlow prediction server
        running: when this flag is set, the step returns only a running service
        model_name: the name of the model that is deployed
    """

    # get the MLFlow deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services with same pipeline name, step name, model name and running flag
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLFlow deployment service has been found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and {model_name} model is currently "
            f"running."
        )
    return existing_services[0]


@step(enable_cache=False)
def predictor(service: MLFlowDeploymentService) -> np.ndarray:
    """
    Makes predictions on thr input data using a deployed ML model

    Args:
        service: An instance of the MLFlowDeploymentService that represents the deployed model service
    Returns:
        np.ndarray: A numy array containing the predictions made by the model
    """
    service.start(timeout=10)
    data = predictor_prepare()
    prediction = service.predict(data)
    return prediction


@step(enable_cache=False)
def get_model() -> ClassifierMixin:
    """
    Import a trained model from the MongoDB
    
    Returns:
        ClassifierMixin: Trained model
    """
    MongoOper = MongoOperations()
    model = MongoOper.read_algorithm_from_mongo("Trained_model")
    return model


@pipeline(enable_cache=False, settings={'docker': docker_settings})
def continuous_deployment_pipeline(
    data_path: str = "data/SAHeart.csv",
    min_accuracy: float = 0.6,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    ingest_df(data_path)
    clean_df(after="ingest_df")
    data_transform(after="clean_df")
    train_model(after="data_transform")
    evaluate_model(after="train_model")
    classifier = get_model(after="train_model")
    deployment_decision = deployment_trigger(after="evaluate_model")
    logging.info("Classifier: ", classifier)
    
    mlflow_model_deployer_step(after="deployment_trigger",
        model = classifier,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False
    )
    prediction = predictor(service=service)
    return prediction
    