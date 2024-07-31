import logging
import pandas as pd
from zenml import step
from src import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from mongo_ops import MongoOperations

@step(enable_cache=False)
def clean_df() -> None:
    """
    Cleans the data and divides it into train and test

    """
    try:
        Mongo_Operations = MongoOperations()
        df = Mongo_Operations.read_data_from_mongo("Raw_data")
        pre_process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, pre_process_strategy)
        cleaned_data = data_cleaning.handle_data()

        divide_strategy = DataSplitStrategy()
        data_dividing = DataCleaning(cleaned_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_dividing.handle_data()
        Mongo_Operations.save_data_to_mongo(cleaned_data, "Cleaned_data")
        Mongo_Operations.save_data_to_mongo(X_train, "X_train")
        Mongo_Operations.save_data_to_mongo(y_train, "y_train")
        Mongo_Operations.save_data_to_mongo(X_test, "X_test")
        Mongo_Operations.save_data_to_mongo(y_test, "y_test")
        logging.info('Cleaning data completed.')

    except Exception as e:
        logging.error(f'Error while cleaning data: {e}')
        raise e
