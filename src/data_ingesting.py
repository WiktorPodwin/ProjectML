import logging
from docker_services import ProjectSparkSession
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

class IngestData:
    """
    Ingesting the data from the data_path.
    """
    def __init__(self, data_path: str, spark_session = SparkSession) -> None:
        self.data_path = data_path
        self.spark_session = spark_session
        """
        Args:
            data_path: Path to data
            spark_session: Spark connection
        """

    def get_data(self) -> SparkDataFrame:
        """
        Ingesting the data from the data_path.

        Returns:
            SparkDataFrame: the ingested data
        """
        logging.info(f'Ingesting data from {self.data_path}')
        return self.spark_session.read.csv(self.data_path, header=True, inferSchema=True, sep=",")
    