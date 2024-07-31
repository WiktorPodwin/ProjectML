from pymongo import MongoClient
import logging
import pandas as pd
import pickle
from sklearn.base import ClassifierMixin
from typing import Union
import gzip

class MongoOperations:
    """
    Class to handle data in MongoDB
    """
    def __init__(self, host: str = "localhost", port: int = 27017, database: str = "Classifier"):
        """
        Initialize the MongoOperation class with a MongoDB connection
        
        Args:
            host: MongoDB host
            port: MongoDB port
            database: MongoDB database
        """
        self.client = MongoClient(host, port)
        self.database = self.client[database]

    def __del__(self):
        """
        Ensure the MongoDB connection is closed
        """
        if hasattr(self, 'client'):
            self.client.close()

    def save_data_to_mongo(self, data: Union[pd.Series, pd.DataFrame, dict], collection_name: str) -> None:            
        """
        Saving data into collection

        Args:
            df: ingested data
            collection_name: name of the collection 
        Returns:
            None 
        """
        try:
            collection = self.database[collection_name]
            if isinstance(data, pd.DataFrame):
                final_data = data.to_dict(orient="records")
                collection.insert_many(final_data)
            elif isinstance(data, pd.Series):
                data = data.reset_index()
                data.columns = ['index', 'value']
                data.drop(columns=["index"], inplace=True)
                final_data = data.to_dict(orient="records")
                collection.insert_many(final_data)
            elif isinstance(data, dict):
                collection.insert_one(data)
            logging.info("Successfully added data into MongoDB collection.")
        except Exception as e:
            logging.error(f"Error while saving data into MongoDB: {e}")
            raise e


    def read_data_from_mongo(self, collection_name: str, column_name: str = None) -> Union[pd.DataFrame, pd.Series]:
        """
        Reading data from collection
        
        Args:
            collection_name: MongoDB collection
            column_name: Column to filter from colelction
        Returns:
            df: Data in Data Frame
        """
        try:
            collection = self.database[collection_name]
            if column_name is None:
                results = collection.find()
            else:
                results = collection.find({}, {column_name: 1, '_id': 0})
            df = pd.DataFrame(list(results))
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])            
            return df
        except Exception as e:
            logging.error(f"Error while loading data from MongoDB: {e}")
            raise e
        
    def save_model_to_mongo(self, model: ClassifierMixin, collection_name: str, model_name) -> None:
        """
        Saving model into collection
        
        Args:
            model: Trained model
            collection_name: MongoDB collection
            model_name: Model name
        """
        try:
            pickled_model = pickle.dumps(model)
            compressed_model = gzip.compress(pickled_model)
            collection = self.database[collection_name]
            collection.insert_one({"model": compressed_model, "name": model_name})
            logging.info(f"Successfully added model into MongoDB collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error while saving model into MongoDB: {e}")
            raise e

    def read_model_from_mongo(self, collection_name: str) -> ClassifierMixin:
        """
        Reading model from MongoDB
        
        Args:
            collection_name: MongoDB collection
        Returns:
            ClassifierMixin: Trained model
        """
        try:
            collection = self.database[collection_name]
            result = collection.find_one()
            pickled_model = result.get("model")
            return pickle.loads(gzip.decompress(pickled_model))
        except Exception as e:
            logging.error(f"Error while loading model from the collection: {collection_name}")
            raise e
        
    def delete_old_data(self) -> None:
        """
        Delete old data from all collections in MongoDB
        """
        try:
            collections_list = self.database.list_collection_names()
            for collection_name in collections_list:
                collection = self.database[collection_name]
                if collection.count_documents({}) > 0:
                    collection.delete_many({})
        except Exception as e:
            logging.error(f"Error while deleting data from MongoDB: {e}")
            raise e