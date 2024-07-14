import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import Accuracy, Recall, F1, RocAuc, ConfMatrix
from sklearn.base import ClassifierMixin
import numpy as np

@step
def evaluate_model(model: ClassifierMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> Tuple[
                       Annotated[float, "accracy"],
                       Annotated[float, "recall"],
                       Annotated[float, "f1_score"],
                       Annotated[float, "roc_auc"],
                       Annotated[np.ndarray, "conf_matrix"]]:
    """
    Evaluates the model on the ingested data
    
    Args:
        X_test: Testing data 
        y_test: Testing labels
    Returns:
        accuracy: Accuracy value
        recall: Recall value
        f1_score: F1 Score value
        roc_auc: Roc-Auc value
        conf_matrix: Matrix of confusion
    """
    try:
        prediction = model.predict(X_test)

        accuracy_class = Accuracy()
        accuracy_val = accuracy_class.calculate_scores(y_test, prediction)

        recall_class = Recall()
        recall_val = recall_class.calculate_scores(y_test, prediction)

        f1_class = F1()
        f1_val = f1_class.calculate_scores(y_test, prediction)

        roc_auc_class = RocAuc()
        roc_auc_val = roc_auc_class.calculate_scores(y_test, prediction)

        conf_matrix_class = ConfMatrix()
        conf_matrix_val = conf_matrix_class.calculate_scores(y_test, prediction)

        return accuracy_val, recall_val, f1_val, roc_auc_val, conf_matrix_val
    except Exception as e:
        logging.error(f"Error while evaluating model: {e}")
        raise e
