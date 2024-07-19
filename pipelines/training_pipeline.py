from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.model_evaluate import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    ingested_data = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(ingested_data)
    classifier = train_model(X_train, X_test, y_train, y_test)
    accuracy, recall, f1, roc_auc, conf_matrix = evaluate_model(classifier, X_test, y_test)



