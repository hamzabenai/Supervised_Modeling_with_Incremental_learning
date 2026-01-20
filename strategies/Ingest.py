# strategies/Ingest.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
import logging


class IngestClass(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def handle_ingest(self, source: str) -> pd.DataFrame:
        pass
      
      
class IngestStrategy(IngestClass):
    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)

    def handle_ingest(self, source: str) -> Union[pd.DataFrame, str]:
        self.logger.info(f"Ingesting data from {source} path")
        if source.endswith('.csv'):
            data = pd.read_csv(source)
        elif source.endswith('xlsx'):
            data = pd.read_excel(source)
        else:
            raise ValueError("Unsupported file format")
        
        self.logger.info(f"Data ingested successfully from {source}")
        self.logger.debug(f"Checking data size :")
        
        records = data.shape[0]
        if records <= 2_000:
            size = "very small"
        elif records <= 10_000:
            size = "small"
        elif records <= 100_000:
            size = "medium"
        else:
            size = "large"
        
        self.logger.debug(f"Data size is {size} with {records} records")
        self.logger.info(f'Checking for NUll values')
        records = data.shape[0]
        null_perc = data.isnull().sum().sum() / records
        if null_perc >= 0.5:
            self.logger.warning(f'High percentage of null values detected')
            self.logger.info(f'Percentage of null values :\n{null_perc*100} %')
            self.logger.info(f'you are risking to lose a lot of data during cleaning')
        return data, size