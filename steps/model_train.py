import logging
import mlflow
from zenml import step
from zenml.client import Client
from src import KNN, LogisticRegressionModel, RandomForestModel, SVMModel, GaussianNBModel, BaggingModel, HyperparameterChoice, create_plot
from .config import ModelNameConfig
from docker_services import MongoOperations

client = Client()
experiment_tracker = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def train_model(config: ModelNameConfig) -> None:
    """
    Trains the model on the ingested data
    
    Args:
        config: Model name
    """
    try:
        mongo_oper = MongoOperations()
        X_train = mongo_oper.read_data_from_mongo("X_train")
        y_train = mongo_oper.read_data_from_mongo("y_train")
        X_test = mongo_oper.read_data_from_mongo("X_test")
        y_test = mongo_oper.read_data_from_mongo("y_test")

        model = None
        if config.name_of_model == "KNeighborsClassifier":
            model = KNN()
        elif config.name_of_model == "LogisticRegression":
            model = LogisticRegressionModel()
        elif config.name_of_model == "RandomForestClassifier":
            model = RandomForestModel()
        elif config.name_of_model == "SVM":
            model = SVMModel()
        elif config.name_of_model == "GaussianNB":
            model = GaussianNBModel()
        elif config.name_of_model == "BaggingClassifier":
            model = BaggingModel()
        else:
            raise ValueError(f'Model not supported: {config.name_of_model}')
        
        y_train = y_train['chd']
        tuner = HyperparameterChoice(model, X_train, y_train, X_test, y_test)

        if config.fine_tuning == True:
            best_parameters = tuner.optimize()
            mlflow.sklearn.autolog()
            trained_model = model.train(X_train, y_train, **best_parameters)
        else: 
            mlflow.sklearn.autolog()
            trained_model = model.train(X_train, y_train)

        create_plot(model=trained_model, X_train=X_train, model_name=config.name_of_model)

        mongo_oper.save_algorithm_to_mongo(algorithm=trained_model, collection_name="trained_model", algorithm_name=config.name_of_model)
    
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise e
