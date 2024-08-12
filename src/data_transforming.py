from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import logging 
import pandas as pd
from abc import abstractmethod, ABC
from typing import Tuple, Any
from pyspark.sql import DataFrame as SparkDataFrame

class DataStrategy(ABC):
    """
    Abstract class defining stratety for transforming data
    """

    @abstractmethod
    def handle_data(self):
        pass

class DataStandardScaler(DataStrategy):
    """
    Class for standardizing training and test dataset using 'StandardScaler'
    """     
    def handle_data(self, X_train: SparkDataFrame, X_test: SparkDataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Standardize data
        
        Args:
            X_train: Training data
            X_test: Testing  data
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]: 
                - Standarized training data 
                - Standarized test data
                - The "StandardScaler" used for the transformation
        """
        try:
            X_train = X_train.toPandas()
            X_test = X_test.toPandas()
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            logging.info("Successfully standardized data with Standard Scaler")
            return X_train_scaled_df, X_test_scaled_df, scaler
        except Exception as e:
            logging.error(f"Error while standardizing data: {e}")
            raise e

class PCAModel(DataStrategy):
    """
    Class for performing Principal Component Analysis "PCA" on data.
    """
    def handle_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, n_components: int) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
        """
        Strategy for PCA

        Args:
            X_train: Training data
            X_test: Testing  data
            n_components: number of the future dimensions
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, PCA]: 
                - Training data after PCA transformation
                - Test data after PCA transformation
                - The "PCA" used for the transformation
        """
        try:
            number_dimensions = min(n_components, len(X_train.iloc[0, :]), len(X_train))
            pca = PCA(n_components=number_dimensions)
            pca.fit(X_train)
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)
            columns=[f"column{x}" for x in range(number_dimensions)]
            X_train_pca = pd.DataFrame(X_train_pca, columns=columns)
            X_test_pca = pd.DataFrame(X_test_pca, columns=columns)
            logging.info("Successfully transformed data with PCA")
            return X_train_pca, X_test_pca, pca
        except Exception as e:
            logging.error(f"Error while transforming data with PCA: {e}")
            raise e

class LDAModel(DataStrategy):
    """
    Class for performing Linear Discriminant Analysis (LDA) on data
    """
    def handle_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, n_components: int) -> Tuple[pd.DataFrame, pd.DataFrame, LinearDiscriminantAnalysis]:
        """
        Strategy for LDA

        Args:
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            n_components: number of the future dimensions
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, LinearDiscriminantAnalysis]:
                - Training data after LDA transformation
                - Test data after LDA transformation
                - The "LDA" used for the transformation
        """
        try:
            number_dimensions = min(n_components, len(X_train.iloc[0, :]), (y_train["chd"].nunique() - 1))
            if number_dimensions == 0:
                number_dimensions = 1
            lda = LinearDiscriminantAnalysis(n_components=number_dimensions)
            lda.fit(X_train, y_train)
            X_train_lda = lda.transform(X_train)
            X_test_lda = lda.transform(X_test)
            columns=[f"column{x}" for x in range(number_dimensions)]
            X_train_lda = pd.DataFrame(X_train_lda, columns=columns)
            X_test_lda = pd.DataFrame(X_test_lda, columns=columns)
            logging.info("Successfully transformed data with LDA")
            return X_train_lda, X_test_lda, lda
        except Exception as e:
            logging.error(f"Error while transforming data with LDA: {e}")
            raise e


class DataTransforming:
    """
    Class for cleaning the data, deviding into train and test and standardizing data
    """
    def __init__(self, strategy: DataStrategy, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series = None, additional_parameter: Any = None):
        self.strategy = strategy
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.additional_parameter = additional_parameter
        
    def standardize(self) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Apply Data Standardization on the data
        """
        try:
            return self.strategy.handle_data(self.X_train, self.X_test)
        except Exception as e:
            logging.error(f'Error while standardizing data: {e}')
            raise e
        
    def pca_process(self) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
        """
        Apply PCA on the data
        """
        try:
            return self.strategy.handle_data(self.X_train, self.X_test, self.additional_parameter)
        except Exception as e:
            logging.error(f"Error while appying PCA: {e}")
            raise e
    
    def lda_process(self) -> Tuple[pd.DataFrame, pd.DataFrame, LinearDiscriminantAnalysis]:
        """
        Apply LDA on the data
        """
        try:
            return self.strategy.handle_data(self.X_train, self.X_test, self.y_train, self.additional_parameter)
        except Exception as e:
            logging.error(f"Error while applying LDA: {e}")
            raise e
        