import logging
import pandas as pd
from zenml import step
from src import DataCleaning, DataPreProcessStrategy, DataSplitStrategy, DataStandardScaler, DataTransforming
from mongo_ops import MongoOperations
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
        Mongo_Operations = MongoOperations()
        df = Mongo_Operations.read_data_from_mongo("Raw_data")

        data_preprocessing = DataPreProcessStrategy()
        data_preprocess = DataCleaning(data_preprocessing, df)
        data_preprocessed = data_preprocess.apply_strategy()

        data_splitting = DataSplitStrategy()
        data_split = DataCleaning(data_splitting, data_preprocessed)
        X_train, X_test, y_train, y_test = data_split.apply_strategy()

        data_standardizing = DataStandardScaler()
        data_standard = DataTransforming(data_standardizing, X_train, X_test)
        X_train, X_test, standard_scaler = data_standard.standardize()
        mlflow.sklearn.log_model(standard_scaler, "Standard_Scaler")

        Mongo_Operations.save_data_to_mongo(data_preprocessed, "Cleaned_data")
        Mongo_Operations.save_data_to_mongo(X_train, "X_train")
        Mongo_Operations.save_data_to_mongo(y_train, "y_train")
        Mongo_Operations.save_data_to_mongo(X_test, "X_test")
        Mongo_Operations.save_data_to_mongo(y_test, "y_test")
        Mongo_Operations.save_algorithm_to_mongo(algorithm=standard_scaler, collection_name="standard_scaler", algorithm_name="Standard Scaler")
        logging.info('Cleaning data completed.')

    except Exception as e:
        logging.error(f'Error while cleaning data: {e}')
        raise e
