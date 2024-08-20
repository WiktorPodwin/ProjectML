from pymongo import MongoClient
import logging
import pandas as pd
import pickle
from typing import Union
import gzip
from pyspark.sql import DataFrame as SparkDataFrame
from sklearn.base import ClassifierMixin, TransformerMixin
from tensorflow.keras.models import Sequential
import torch.nn as nn


class MongoOperations:
    """
    Class to handle data in MongoDB
    """
    def __init__(self, host: str = "localhost", port: int = 27017, database: str = "classifier"):
        """
        Initializes the MongoOperation class with a MongoDB connection
        
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

    def save_data_to_mongo(self, data: Union[pd.Series, pd.DataFrame, SparkDataFrame, dict], collection_name: str) -> None:            
        """
        Saving data into MongoDB collection

        Args:
            data: ingested data
            collection_name: name of the collection 
        """
        try:
            collection = self.database[collection_name]
            if isinstance(data, SparkDataFrame):
                data = data.toPandas()
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
            else:
                raise TypeError("Wrong type of input data, the possible types "
                                "of data: (pd.Series, pd.DataFrame, DataFrame, dict).")
            
            logging.info(f"Successfully added data into MongoDB collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error while saving data into MongoDB: {e}")
            raise e


    def read_data_from_mongo(self, collection_name: str, column_name: str = None) -> Union[pd.DataFrame, pd.Series]:
        """
        Reading data from MongoDB collection
        
        Args:
            collection_name: MongoDB collection
            column_name: Column to filter from colelction
        Returns:
             Union[pd.DataFrame, pd.Series]: Loaded data as pd.DataFrame or pd.Series
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
        
    def save_algorithm_to_mongo(self, algorithm: Union[ClassifierMixin, Sequential, TransformerMixin, nn.Module], collection_name: str, algorithm_name: str) -> None:
        """
        Saving algorithm into MongoDB collection
        
        Args:
            algorithm: Algorithm to save into MongoDB
            collection_name: MongoDB collection
            algorithm_name: Algorithm name
        """
        try:
            pickled_algorithm = pickle.dumps(algorithm)
            compressed_algorithm = gzip.compress(pickled_algorithm)
            collection = self.database[collection_name]
            collection.insert_one({"algorithm": compressed_algorithm, "name": algorithm_name})
            logging.info(f"Successfully added algorithm into MongoDB collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error while saving algorithm into MongoDB: {e}")
            raise e

    def read_algorithm_from_mongo(self, collection_name: str) -> Union[ClassifierMixin, Sequential, TransformerMixin, nn.Module]:
        """
        Reading algorithm from MongoDB
        
        Args:
            collection_name: MongoDB collection
        Returns:
            Union[ClassifierMixin, Sequential, TransformerMixin, nn.Module]: Loaded algorithm from MongoDB
        """
        try:
            collection = self.database[collection_name]
            result = collection.find_one()
            pickled_algorithm = result.get("algorithm")
            return pickle.loads(gzip.decompress(pickled_algorithm))
        except Exception as e:
            logging.error(f"Error while loading algorithm from the collection: {collection_name}")
            raise e
        
    def delete_old_data(self) -> None:
        """
        Deletes old data from all collections in MongoDB
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
        
    def clear_collection(self, collection_name: str) -> None:
        """
        Deletes all documents in collection

        Args:
            collection_name: MongoDB collection
        """
        try:
            collection = self.database[collection_name]
            collection.delete_many({})
        except Exception as e:
            logging.error(f"Error while deleting documents from collection: {collection_name}: {e}")
            raise e