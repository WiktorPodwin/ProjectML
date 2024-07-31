import logging
from mongo_ops import MongoOperations
import numpy as np

def deployment_trigger_prepare() -> float:
    """
    Prepare accuracy to comparison
    """
    try:
        MongoOper = MongoOperations()
        accuracy_df = MongoOper.read_data_from_mongo(
            collection_name="Evaluation", 
            column_name="Accuracy"
            )
        accuracy = float(accuracy_df.iloc[0, 0])
        return accuracy
    except Exception as e:
        logging.error(f"Error while importing accuracy: {e}")
        raise e

def predictor_prepare() -> np.ndarray:
    """
    Prepare data to prediction
    """
    try:
        MongoOper = MongoOperations()
        data = MongoOper.read_data_from_mongo(collection_name="Cleaned_data")
        data = data.sample(n=100)
        data_array = np.array(data.to_dict(orient="records"))
        return data_array
    except Exception as e:
        logging.error(f"Error while preparing data to prediction: {e}")
        raise e