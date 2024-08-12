import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
import logging
import os
import numpy as np

class PlotFeatureImportances:
    """
    Class for ploting feature importances
    """
    def __init__(self, model: ClassifierMixin, feature_names: list, save_path: str, model_name: str) -> None:
        """
        model: Trained model
        feature_names: List of columns names
        save_path: Path to save a plot
        model_name: Name of trained model
        """
        self.model = model
        self.feature_names = feature_names
        self.save_path = save_path
        self.model_name = model_name

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances from model using function feature_importances

        Return:
            np.ndarray: Array with feature importances 
        """
        try:
            feature_importances = self.model.feature_importances_
            return feature_importances
        except Exception as e:
            logging.error(f"Error while calculating feature importances")
            raise e
        
    def get_coef(self) -> np.ndarray:
        """
        Get feature importances from model using function coef_

        Return:
            np.ndarray: Array with feature importances 
        """
        try:
            coeficients = self.model.coef_[0]
            return coeficients
        except Exception as e:
            logging.error(f"Error while calculating feature importances")
            raise e
        
    def get_feature_importances_bagging(self) -> np.ndarray:
        """
        Get feature importances from Bagging model using function feature_importances

        Returns:
            np.ndarray: Array with feature importances
        """
        try:
            feature_importances = np.mean([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
            if len(feature_importances) != len(self.feature_names):
                differece = len(self.feature_names) - len(feature_importances)
                for _ in range(differece):
                    feature_importances = np.append(feature_importances, [0])

            return feature_importances
        except Exception as e:
            logging.error(f"Error while calculating feature importances")
            raise e
        
    def handle_plot(self, feature_importances: np.ndarray) -> None:
        """
        Plots feature importances and saves into file

        Args:
            feature_importances: Array with feature importances
        """
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            data = pd.DataFrame({"feature": self.feature_names, "importance": feature_importances})
            data = data.sort_values(by="importance", ascending=False)
            plt.figure(figsize=(12, 8))
            plt.barh(data["feature"], data["importance"], color="skyblue")
            plt.xlabel('Importance')
            plt.title(f'Feature Importances for model: {self.model_name}')
            plt.gca().invert_yaxis()
            plt.savefig(self.save_path)
            logging.info("Sucessfully feature importances saved into 'plot/feature_importances.png'")
        except Exception as e:
            logging.error(f"Error while saving plot to file: {e}")
            raise e

def create_plot(model: ClassifierMixin, X_train: pd.DataFrame, model_name: str):    
    if len(X_train.iloc[0, :]) == 1:
        logging.info("To low number of dimensions to load feature importance plot")
        return
    plot_feature_importances = PlotFeatureImportances(model=model, feature_names=X_train.columns, save_path="plot/feature_importances.png", model_name=model_name)
    if model_name == "LogisticRegression" or model_name == "SVM":
        feature_importances = plot_feature_importances.get_coef()
    elif model_name == "RandomForestClassifier":
        feature_importances = plot_feature_importances.get_feature_importance()
    elif model_name == "BaggingClassifier":
        feature_importances = plot_feature_importances.get_feature_importances_bagging()
    else:
        logging.info(f"Feature importances not supported in model: {model_name}")
        return
    plot_feature_importances.handle_plot(feature_importances)