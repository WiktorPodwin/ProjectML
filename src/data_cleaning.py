import logging 
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from typing import Union

class DataStrategy(ABC):
    """
    Abstract class defining stratety for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for processing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data.fillna(data['sbp'].median(), inplace=True)
            data.fillna(data['tobacco'].median(), inplace=True)
            data.fillna(data['ldl'].median(), inplace=True)
            data.fillna(data['adiposity'].median(), inplace=True)
            data.dropna(subset=['famhist'])
            data.fillna(data['typea'].median(), inplace=True)
            data.fillna(data['obesity'].median(), inplace=True)
            data.fillna(data['alcohol'].median(), inplace=True)
            data.fillna(data['age'].median(), inplace=True)
            data.dropna(subset=['chd'])

            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
            famhist_encoded = encoder.fit_transform(data[['famhist']])
            data = pd.concat([data, famhist_encoded], axis=1).drop(columns=['famhist', 'row.names'])
            data = data.select_dtypes(include=[np.number])
            return data
        except Exception as e:
            logging.error(f'Error in processing data: {e}')
            raise e
        
class DataSplitStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Devide data into train and test
        """
        try:
            X = data.drop(['chd'], axis=1)
            y = data['chd']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Error while dividing data: {e}')
            raise e
        

class DataCleaning:
    """
    Class for cleaning the data and deviding into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f'Error with handling data: {e}')
            raise e
