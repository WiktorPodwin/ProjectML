import logging
import pandas as pd
from zenml import step
from src.model_development import KNN
from .config import ModelNameConfig
from sklearn.base import ClassifierMixin


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig
) -> ClassifierMixin:
    """
    Trains the model on the ingested data
    
    Args:
        X_train: Traning data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing_labels
        config: Model name
    """
    try:
        model = None
        if config.model_name == "KNeighborsClassifier":
            model = KNN()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f'Model not supported: {config.model_name}')
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
