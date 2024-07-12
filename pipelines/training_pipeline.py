from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluate_model import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    ingested_data = ingest_df(data_path)
    clean_df(ingested_data)
    train_model(ingested_data)
    evaluate_model(ingested_data)


