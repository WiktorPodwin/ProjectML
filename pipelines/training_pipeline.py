from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.model_evaluate import evaluate_model

@pipeline(enable_cache=False, enable_step_logs=True)
def train_pipeline(data_path: str):
    ingest_df(data_path)
    clean_df(after="ingest_df")
    train_model(after="clean_df")
    evaluate_model(after="train_model")