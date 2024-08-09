import logging
from zenml import step
from src import IngestData
from docker_services import MongoOperations, ProjectSparkSession

@step(enable_cache=False, enable_step_logs=True)
def ingest_df(data_path: str):
    """
    Ingesting the data from the data path.

    Args:
        data_path: path to data
    """
    try:
        spark_session = ProjectSparkSession.initialize_spark_session()
        ingest_data = IngestData(data_path, spark_session)
        df = ingest_data.get_data()
        Mongo_Operations = MongoOperations()
        Mongo_Operations.delete_old_data()
        Mongo_Operations.save_data_to_mongo(data=df, collection_name="Raw_data")
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}", exc_info=True)
        raise e
    finally:
        ProjectSparkSession.stop_spark_session(spark_session)