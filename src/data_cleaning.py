import logging 
import pandas as pd
from abc import ABC, abstractmethod
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import NumericType
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame
from typing import NamedTuple


class DataStrategy(ABC):
    """
    Abstract class defining stratety for handling data
    """

    @abstractmethod
    def handle_data(self):
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for processing data
    """

    def handle_data(self, data: pd.DataFrame, spark_session: SparkSession) -> SparkDataFrame:
        """
        Preprocess data

        Args:
            data: Dataset for preprocessing
            spark_session: Spark connection
        Returns:
            SparkDataFrame: Cleaned data
        """
        try:
            spark_data = spark_session.createDataFrame(data)
            spark_data = spark_data.drop("row_names")
            spark_data = spark_data.dropna()
            indexer = StringIndexer(inputCol="famhist", outputCol="famhist_index")
            data_indexed = indexer.fit(spark_data).transform(spark_data)
            data_prepared = data_indexed.drop("famhist")
            numeric_col = [field.name for field in data_prepared.schema.fields if isinstance(field.dataType, NumericType)]
            data_numeric = data_prepared.select(numeric_col)
            return data_numeric
        except Exception as e:
            logging.error(f'Error in processing data: {e}')
            raise e
        
class SplitData(NamedTuple):
    """
    A named tuple to hold the results of a data split operation.

    Attributes:
        X_train: The training features as a Spark DataFrame.
        X_test: The testing features as a Spark DataFrame.
        y_train: The training labels as a Spark DataFrame.
        y_test: The testing labels as a Spark DataFrame.
    """
    X_train: SparkDataFrame
    X_test: SparkDataFrame
    y_train: SparkDataFrame
    y_test: SparkDataFrame

class DataSplitStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test.
    """
    def handle_data(self, data: SparkDataFrame, spark_session: SparkSession) -> SplitData:
        """
        Devide data into train and test

        Args:
            data: The input Spark DataFrame containing the dataset
            spark_session: Spark connection

        Returns:
            SplitData: a named tuple containing the following Spark Dataframes:
                - X_train: The training features as a Spark DataFrame.
                - X_test: The testing features as a Spark DataFrame.
                - y_train: The training labels as a Spark DataFrame.
                - y_test: The testing labels as a Spark DataFrame.
        """
        try:
            
            train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
            X_train = train_data.drop("chd")
            X_test = test_data.drop("chd")
            y_train = train_data.select("chd")
            y_test = test_data.select("chd")
            return SplitData(X_train, X_test, y_train, y_test)
        except Exception as e:
            logging.error(f'Error while dividing data: {e}')
            raise e

class DataCleaning:
    """
    Class for cleaning the data and deviding into train and test
    """
    def __init__(self, strategy: DataStrategy, data: pd.DataFrame, spark_session: SparkSession = None):
        self.strategy = strategy
        self.data = data
        self.spark_session = spark_session

    def apply_strategy(self) -> SparkDataFrame:
        """
        Apply precess strategy on the data
        """
        try:
            return self.strategy.handle_data(self.data, self.spark_session)
        except Exception as e:
            logging.error(f'Error while precessing data: {e}')
            raise e
        