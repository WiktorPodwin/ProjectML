import logging
from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import pandas as pd
import optuna
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score

class Model(ABC):
    """
    Abstract class defining model for prediction
    """
    @abstractmethod
    def model_train(self, X_train, y_train):
        """
        Trains the model
        
        Args:
            X_train: Training data
            y_train: Training labels
        """
        pass

    @abstractmethod
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        """
        Optimize the best hyperparameters for the model
        
        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
        """
        pass

class LogisticRegressionModel(Model):
    """
    Logistic Regression Model
    """
    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> ClassifierMixin:
        """
        Trains the model
        
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            ClassifierMixin: The trained classifier
        """
        try:
            classifier = LogisticRegression(**kwargs)
            classifier.fit(X_train, y_train)
            return classifier
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e

    def optimize(self, trial, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimize the best hyperparameters for the model
        
        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
        Returns:
            float: accuracy score of the trained model on the test data
        """
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))
        tol = trial.suggest_float("tol", 1e-5, 1e-2, log=True)
        classifier = self.model_train(X_train, y_train, C=C, solver=solver, tol=tol)
        return classifier.score(X_test, y_test)


class KNN(Model):
    """
    K-Nearest Neighbors model
    """

    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> ClassifierMixin:
        """
        Trains the model
        
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            ClassifierMixin: The trained classifier
        """
        try:
            classifier = KNeighborsClassifier(**kwargs)
            classifier.fit(X_train, y_train)
            return classifier
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e
        
        
    def optimize(self, trial, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimize the best hyperparameters for the model
        
        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
        Returns:
            float: accuracy score of the trained model on the test data
        """
        n_neighbors = trial.suggest_int("n_neighbors", 3, 15)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        leaf_size = None
        if algorithm == "ball_tree" or "kd_tree":
            leaf_size = trial.suggest_int("leaf_size", 10, 50)
        classifier = self.model_train(X_train, y_train, n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size)
        return classifier.score(X_test, y_test)

class RandomForestModel(Model):
    """
    The Random Forest Claassifier
    """
    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series, **args) -> ClassifierMixin:
        """
        Trains the model
        
         Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            ClassifierMixin: Trained classifier
        """
        try:
            classifier = RandomForestClassifier(**args)
            classifier.fit(X_train, y_train)
            return classifier
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e
        
    def optimize(self, trial, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
         """
        Optimize the best hyperparameters for the model
        
        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
        Returns:
            float: accuracy score of the trained model on the test data
        """
         n_estimators = trial.suggest_int("n_estimators", 5, 150)
         max_depth = trial.suggest_int("max_depth", 1, 30)
         criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
         classifier = self.model_train(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
         return classifier.score(X_test, y_test)

class SVMModel(Model):
    """
    The SVM Classifier
    """
    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series, **args) -> ClassifierMixin:
        """
        Trains the model
        
        Args:  
            X_train: Training data
            y_train: Training labels
        Returns:
            ClassifierMixin: Trained classifier
        """
        try:
            y_train = y_train.values.ravel()
            classifier = SVC(**args)
            classifier.fit(X_train, y_train)
            logging.info("Model training comlpeted")
            return classifier
        except Exception as e:
            logging.error(f"Error in model trtaining: {e}")
            raise e
    def optimize(self, trial, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimize the best hyperparameters for the model

        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
        Returns:
            float: accuracy score of the trained model on the test data
        """
        C = trial.suggest_float("C", 5e-2, 10, log=True)
        tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        classifier = self.model_train(X_train, y_train, C=C, tol=tol, gamma=gamma)
        return classifier.score(X_test, y_test)
    
class GaussianNBModel(Model):
    """
    The Gaussian Naive Bayes Classifier
    """
    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series, **args) -> ClassifierMixin:
        """
        Trains the model
        
        Args:  
            X_train: Training data
            y_train: Training labels
        Returns:
            ClassifierMixin: Trained classifier
        """
        try:
            classifier = GaussianNB(**args)
            classifier.fit(X_train, y_train)
            logging.info("Model training completed")
            return classifier
        except Exception as e:
            logging.error(f"Error inmodel training: {e}")
            raise e
    
    def optimize(self, trial, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimize the best hyperparameters for the model

        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
        Returns:
            float: accuracy score of the trained model on the test data
        """
        var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-8, log=True)
        classifier = self.model_train(X_train, y_train, var_smoothing=var_smoothing)
        return classifier.score(X_test, y_test)
    
class BaggingModel(Model):
    """
    The Bagging Classifier
    """
    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series, **args) -> ClassifierMixin:
        """
        Trains the model

        Args:  
            X_train: Training data
            y_train: Training labels
        Returns:
            ClassifierMixin: Trained classifier
        """
        try:
            classifier = BaggingClassifier(**args)
            classifier.fit(X_train, y_train)
            logging.info("Model training completed")
            return classifier
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e
    
    def optimize(self, trial, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimize the best hyperparameters for the model

        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
        Returns:
            float: accuracy score of the trained model on the test data
        """
        try:
            n_estimators = trial.suggest_int("n_estimators", 10, 100)
            oob_score = trial.suggest_categorical("oob_score", [True, False])
            max_samples = trial.suggest_float("max_samples", 0.2, 1.0)
            max_features = trial.suggest_float("max_features", 0.2, 1.0)
            classifier = self.model_train(X_train, y_train, n_estimators=n_estimators, oob_score=oob_score, max_samples=max_samples, max_features=max_features)
            scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring="accuracy")
            return scores.mean()
        except Exception as e:
            logging.error(f"Error while tuning BaggingClassifier: {e}")
            raise e

class HyperparameterChoice:
    """
    Class for choosing best hyperparameters.
    """
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize(self, n_trials=14):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.X_train, self.y_train, self.X_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params


