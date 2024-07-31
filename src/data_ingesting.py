import logging
import pandas as pd

class IngestData:
    """
    Ingesting the data from the data_path.
    """
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        """
        Args:
            data_path: path to data
        """

    def get_data(self) -> pd.DataFrame:
        """
        Ingesting the data from the data_path.

        Returns:
            pd.DataFrame: the ingested data
        """
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)
    