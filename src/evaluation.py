import logging
from abc import ABC, abstractmethod
import numpy as np 
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate score for the model
         Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class Accuracy(Evaluation):
    """
    Evaluation strategy that uses accuracy
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the accuracy for the model
         Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            float: Value of the accuracy
        """
        try:
            logging.info("Calculating accuracy")
            accuracy_val = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy: {accuracy_val}")
            return accuracy_val
        except Exception as e:
            logging.error(f"Error while calculating accuracy: {e}")
            raise e

class Recall(Evaluation):
    """
    Evaluation strategy that uses recall
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall for the model
         Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            float: Value of the recall
        """
        try:
            logging.info("Calculating recall")
            recall_val = recall_score(y_true, y_pred, average="binary")
            logging.info(f"Recall: {recall_val}")
            return recall_val
        except Exception as e:
            logging.error(f"Error while calculating recall: {e}")
            raise e

class F1(Evaluation):
    """
    Evaluation strategy that uses F1 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the F1 Score for the model
         Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            float: Value of the F1 Score
        """
        try:
            logging.info("Calculating F1 Score")
            f1_val = f1_score(y_true, y_pred, average="binary")
            logging.info(f"F1 Score: {f1_val}")
            return f1_val
        except Exception as e:
            logging.error(f"Error while calculating F1 Score: {e}")
            raise e

class RocAuc(Evaluation):
    """
    Evaluation strategy that uses Roc-Auc
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Roc-Auc for the model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            float: Value of the Roc-Auc
        """
        try:
            logging.info("Calculating Roc-Auc")
            roc_auc_val = roc_auc_score(y_true, y_pred)
            logging.info(f"Roc-Auc: {roc_auc_val}")
            return roc_auc_val
        except Exception as e:
            logging.error(f"Error while calculating Roc-Auc: {e}")
            raise e
        
class ConfMatrix(Evaluation):
    """
    Evaluation strategy that uses Confusion Matrix
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate Confusion Matrix for the model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            float: Value of the Confusion Matrix
        """
        try:
            logging.info("Calculating Confusion Matrix")
            conf_mat_val = confusion_matrix(y_true, y_pred)
            logging.info(f"Confusion Matrix: {conf_mat_val}")
            return conf_mat_val
        except Exception as e:
            logging.error(f"Error while calculating Confusion Matrix: {e}")
            raise e