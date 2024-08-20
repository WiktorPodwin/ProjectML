import logging
import mlflow
from zenml import step
from zenml.client import Client
from src import Accuracy, Recall, F1, RocAuc, ConfMatrix, pytorch_prediction
from docker_services import MongoOperations
from .config import ModelNameConfig
import numpy as np

client = Client()
experiment_tracker = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def evaluate_model(config: ModelNameConfig) -> None:
    """
    Evaluates the model on the ingested data
    """
    try:
        mongo_oper = MongoOperations()
        X_test = mongo_oper.read_data_from_mongo("X_test")
        y_test = mongo_oper.read_data_from_mongo("y_test")
        model = mongo_oper.read_algorithm_from_mongo("trained_model")
        config = ModelNameConfig()

        if config.name_of_model == "PyTorchNeuralNetwork":
            prediction = pytorch_prediction(model, X_test)
        else:
            prediction = model.predict(X_test)

        if np.issubdtype(prediction.dtype, np.floating):
            prediction = [0 if val < 0.5 else 1 for val in prediction]
            
        accuracy_class = Accuracy()
        accuracy_val = accuracy_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("validation_accuracy", accuracy_val)

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
        mongo_oper.save_data_to_mongo(data=data, collection_name='evaluation')
        
    except Exception as e:
        logging.error(f"Error while model evaluating: {e}")
        raise e
