import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple

@step(enable_cache=False)
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Cleans the data and divides it into train and test
    
    Args:
        df: raw data
    Returns:
        X_train: Traning data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing_labels
    """
    try:
        pre_process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, pre_process_strategy)
        cleaned_data = data_cleaning.handle_data()

        divide_strategy = DataSplitStrategy()
        data_dividing = DataCleaning(cleaned_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_dividing.handle_data()

        logging.info('Cleaning data completed.')
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f'Error while cleaning data: {e}')
        raise e
