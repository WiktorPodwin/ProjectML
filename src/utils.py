import logging
from docker_services import MongoOperations
import pandas as pd
from steps import DataTransformConfig

def deployment_trigger_prepare() -> float:
    """
    Prepare accuracy to comparison; imports model accuracy 

    Returns:
        float: value of the model accuracy
    """
    try:
        mongo_oper = MongoOperations()
        accuracy_df = mongo_oper.read_data_from_mongo(
            collection_name="Evaluation", 
            column_name="Accuracy"
            )
        accuracy = float(accuracy_df.iloc[0, 0])
        return accuracy
    except Exception as e:
        logging.error(f"Error while importing accuracy: {e}")
        raise e

def predictor_prepare() -> pd.DataFrame:
    """
    Prepare data to prediction

    Returns:
        pd.DataFrame: Prepared data
    """
    try:
        mongo_oper = MongoOperations()
        data = mongo_oper.read_data_from_mongo(collection_name="Cleaned_data")
        standard_scaler = mongo_oper.read_algorithm_from_mongo("standard_scaler")
        X_data = data.drop(columns=["chd"])
        y_data = data['chd']
        X_data = pd.DataFrame(standard_scaler.transform(X_data), columns=X_data.columns)
        config = DataTransformConfig()
        if config.fine_transforming == True:
            transform_algorithm = mongo_oper.read_algorithm_from_mongo("transformation_algorithm")
            X_data_transformed = transform_algorithm.transform(X_data)
            X_data = pd.DataFrame(X_data_transformed, columns=[f"column{x}" for x in range(len(X_data_transformed[0]))])
        data = pd.concat([X_data, y_data], axis=1)
        data = data.sample(n=100)    
        return data
    except Exception as e:
        logging.error(f"Error while preparing data to prediction: {e}")
        raise e
    