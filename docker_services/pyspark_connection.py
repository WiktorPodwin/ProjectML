from pyspark.sql import SparkSession
import logging

class ProjectSparkSession:
    """
    Class for interacting with Apache Spark
    """
    
    def initialize_spark_session():
        """
        Connects with Spark using SparkSession
        """
        try:
            spark = SparkSession.builder \
                .appName("ProjectML") \
                .config("spark.mongodb.input.uri", "mongodb://mongo-db:27017/Classifier") \
                .config("spark.mongodb.output.uri", "mongodb://mongo-db:27017/Classifier") \
                .getOrCreate()
            logging.info("Successfully started Spark session.")
            return spark
        except Exception as e:
            logging.error(f"Error while connecting with Spark: {e}")
            raise e
        

    def stop_spark_session(spark):
        """
        Stops Spark session

        Args:
            spark: Running Spark session
        """
        try:
            spark.stop()
            logging.info("Succesfully stopped Spark session.")
        except Exception as e:
            logging.error(f"Error while stoping Spark session: {e}")
            raise e