import logging
import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client
from typing import Tuple
from typing_extensions import Annotated
from src import Accuracy, Recall, F1, RocAuc, ConfMatrix
from sklearn.base import ClassifierMixin
import numpy as np
from mongo_ops import MongoOperations

client = Client()
client.activate_stack("mlflow_stack_customer")

experiment_tracker = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def evaluate_model() -> None:
    """
    Evaluates the model on the ingested data
    """
    try:
        Mongo_Operations = MongoOperations()
        X_test = Mongo_Operations.read_data_from_mongo("X_test")
        y_test = Mongo_Operations.read_data_from_mongo("y_test")
        model = Mongo_Operations.read_model_from_mongo("Trained_model")
        prediction = model.predict(X_test)

        accuracy_class = Accuracy()
        accuracy_val = accuracy_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("accuracy", accuracy_val)

        recall_class = Recall()
        recall_val = recall_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("recall", recall_val)

        f1_class = F1()
        f1_val = f1_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("f1 score", f1_val)

        roc_auc_class = RocAuc()
        roc_auc_val = roc_auc_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("roc-auc", roc_auc_val)

        conf_matrix_class = ConfMatrix()
        conf_matrix_val = conf_matrix_class.calculate_scores(y_test, prediction)
        t_n, f_p, f_n, t_p = conf_matrix_val.ravel()

        mlflow.log_metric("tn", t_n)
        mlflow.log_metric("fp", f_p)
        mlflow.log_metric("fn", f_n)
        mlflow.log_metric("tp", t_p)
        
        data = {"Accuracy": accuracy_val, 
                "Recall": recall_val,
                "F1 Score": f1_val,
                "Roc-auc": roc_auc_val,
                "Confusion matrix": str(conf_matrix_val)}
        Mongo_Operations.save_data_to_mongo(data=data, collection_name='Evaluation')
        
    except Exception as e:
        logging.error(f"Error while evaluating model: {e}")
        raise e
