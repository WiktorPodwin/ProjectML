from zenml import step
from src import PCAModel, LDAModel, DataTransforming
from mongo_ops import MongoOperations
import logging
from .config import DataTransformConfig
import mlflow
from zenml.client import Client

client = Client()
client.activate_stack("mlflow_stack_customer")
experiment_tracker = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def data_transform(config: DataTransformConfig) -> None:
    """
    Transforms data using decomposition algorithms
    """
    try:
        if config.fine_transforming == True:
            mongo_oper = MongoOperations()
            X_train = mongo_oper.read_data_from_mongo("X_train")
            y_train = mongo_oper.read_data_from_mongo("y_train")
            X_test = mongo_oper.read_data_from_mongo("X_test")

            if config.name_of_transformation == "PCA":
                mlflow.sklearn.autolog()
                pca = PCAModel()
                pca_transforming = DataTransforming(strategy=pca, X_train=X_train, X_test=X_test, additional_parameter=config.number_dimensions)
                X_train, X_test, algorithm = pca_transforming.pca_process()
                mlflow.sklearn.log_model(algorithm, "PCA")
                            
            elif config.name_of_transformation == "LDA":
                lda = LDAModel()
                lda_transforming = DataTransforming(strategy=lda, X_train=X_train, X_test=X_test, y_train=y_train, additional_parameter=config.number_dimensions)
                X_train, X_test, algorithm = lda_transforming.lda_process()
                mlflow.sklearn.log_model(algorithm, "LDA")
            else:
                raise ValueError(f"Algorithm not supported: {config.name_of_transformation}")
            mongo_oper.clear_collection("X_train")
            mongo_oper.clear_collection("X_test")
            mongo_oper.save_data_to_mongo(X_train, "X_train")
            mongo_oper.save_data_to_mongo(X_test, "X_test")
            mongo_oper.save_algorithm_to_mongo(algorithm=algorithm, collection_name="transformation_algorithm", algorithm_name=config.name_of_transformation)
            logging.info("Data transforming completed. ")
        else:
            logging.info("Data transforming disabled.")
    except Exception as e:
        logging.error(f"Error whila data transforming: {e}")
        raise e



