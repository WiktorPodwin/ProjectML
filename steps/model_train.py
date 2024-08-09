import logging
import mlflow
from zenml import step
from zenml.client import Client
from src import KNN, LogisticRegressionModel, RandomForestModel, SVMModel, GaussianNBModel, BaggingModel, HyperparameterChoice
from .config import ModelNameConfig
from docker_services import MongoOperations

client = Client()
client.activate_stack("mlflow_stack_customer")
experiment_tracker = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def train_model(config: ModelNameConfig) -> None:
    """
    Trains the model on the ingested data
    
    Args:
        config: Model name
    """
    try:
        Mongo_Operations = MongoOperations()
        X_train = Mongo_Operations.read_data_from_mongo("X_train")
        y_train = Mongo_Operations.read_data_from_mongo("y_train")
        X_test = Mongo_Operations.read_data_from_mongo("X_test")
        y_test = Mongo_Operations.read_data_from_mongo("y_test")

        model = None
        if config.name_of_model == "KNeighborsClassifier":
            model = KNN()
            mlflow.sklearn.log_model(model, "model")
        elif config.name_of_model == "LogisticRegression":
            model = LogisticRegressionModel()
            mlflow.sklearn.log_model(model, "model")
        elif config.name_of_model == "RandomForestClassifier":
            model = RandomForestModel()
            mlflow.sklearn.log_model(model, "model")
        elif config.name_of_model == "SVM":
            model = SVMModel()
            mlflow.sklearn.log_model(model, "model")
        elif config.name_of_model == "GaussianNB":
            model = GaussianNBModel()
            mlflow.sklearn.log_model(model, "model")
        elif config.name_of_model == "BaggingClassifier":
            mlflow.autolog()
            model = BaggingModel()
            mlflow.sklearn.log_model(model, "model")
        else:
            raise ValueError(f'Model not supported: {config.name_of_model}')
                
        tuner = HyperparameterChoice(model, X_train, y_train, X_test, y_test)

        if config.fine_tuning == True:
            best_parameters = tuner.optimize()
            mlflow.log_params(best_parameters)
            trained_model = model.train(X_train, y_train, **best_parameters)
        else: 
            trained_model = model.train(X_train, y_train)

        Mongo_Operations.save_algorithm_to_mongo(algorithm=trained_model, collection_name="Trained_model", algorithm_name=config.name_of_model)
    
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise e
