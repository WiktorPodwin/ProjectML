import logging
from zenml import step
from src import DataCleaning, DataPreProcessStrategy, DataSplitStrategy, DataStandardScaler, DataTransforming
from docker_services import MongoOperations, ProjectSparkSession
import mlflow
from zenml.client import Client

client = Client()
client.activate_stack("mlflow_stack_customer")
experiment_tracker = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def clean_df() -> None:
    """
    Cleans the data and divides it into train and test, and standardize the data
    """
    try:
        spark_session = ProjectSparkSession.initialize_spark_session()
        mongo_oper = MongoOperations()
        df = mongo_oper.read_data_from_mongo("Raw_data")

        data_preprocessing = DataPreProcessStrategy()
        data_preprocess = DataCleaning(data_preprocessing, df, spark_session)
        data_preprocessed = data_preprocess.apply_strategy()

        data_splitting = DataSplitStrategy()
        data_split = DataCleaning(data_splitting, data_preprocessed, spark_session)
        X_train, X_test, y_train, y_test = data_split.apply_strategy()

        data_standardizing = DataStandardScaler()
        data_standard = DataTransforming(data_standardizing, X_train, X_test)
        X_train, X_test, standard_scaler = data_standard.standardize()
        mlflow.sklearn.log_model(standard_scaler, "Standard_Scaler")

        mongo_oper.save_data_to_mongo(data_preprocessed, "Cleaned_data")
        mongo_oper.save_data_to_mongo(X_train, "X_train")
        mongo_oper.save_data_to_mongo(y_train, "y_train")
        mongo_oper.save_data_to_mongo(X_test, "X_test")
        mongo_oper.save_data_to_mongo(y_test, "y_test")
        mongo_oper.save_algorithm_to_mongo(algorithm=standard_scaler, collection_name="standard_scaler", algorithm_name="Standard Scaler")
        logging.info('Cleaning data completed.')
    except Exception as e:
        logging.error(f'Error while cleaning data: {e}')
        raise e
    finally:
        ProjectSparkSession.stop_spark_session(spark_session)