from zenml import pipeline
from steps import ingest_df, clean_df, train_model, evaluate_model, data_transform


@pipeline(enable_cache=False, enable_step_logs=True)
def train_pipeline(data_path: str):
    ingest_df(data_path)
    clean_df(after="ingest_df")
    data_transform(after="clean_df")
    train_model(after="data_transform")
    evaluate_model(after="train_model")