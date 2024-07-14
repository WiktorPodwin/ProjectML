import logging
from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

class Model(ABC):
    """
    Abstract class defining model for prediction
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        pass


class KNN(Model):
    """
    K-Nearest Neighbors model
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> KNeighborsClassifier:
        """
        Trains the model
        
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            classifier: The trained classifier
        """
        try:
            n_neighbors = kwargs.get('n_neighbors', 2)
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
            classifier.fit(X_train, y_train)
            logging.info("Model training completed.")
            return classifier
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e