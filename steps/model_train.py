import logging
import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client
from src.model_development import KNN, LogisticRegressionModel, RandomForestModel, HyperparameterChoice
from .config import ModelNameConfig
from sklearn.base import ClassifierMixin

client = Client()
client.activate_stack("mlflow_stack_customer")

experiment_tracker = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig
) -> ClassifierMixin:
    """
    Trains the model on the ingested data
    
    Args:
        X_train: Traning data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing_labels
        config: Model name
    """
    try:
        model = None
        if config.name_of_model == "KNeighborsClassifier":
            mlflow.sklearn.autolog()   
            model = KNN()
        elif config.name_of_model == "LogisticRegression":
            mlflow.sklearn.autolog()
            model = LogisticRegressionModel()
        elif config.name_of_model == "RandomForestClassifier":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        else:
            raise ValueError(f'Model not supported: {config.name_of_model}')
        
        tuner = HyperparameterChoice(model, X_train, y_train, X_test, y_test)

        if config.fine_tuning == True:
            best_parameters = tuner.optimize()
            trained_model = model.train(X_train, y_train, **best_parameters)
        else: 
            trained_model = model.train(X_train, y_train)
        return trained_model
    
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
