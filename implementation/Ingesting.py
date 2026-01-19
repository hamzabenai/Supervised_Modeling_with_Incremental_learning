# implementation/Ingesting.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd 
import logging
from typing import Union, Tuple
from strategies.Ingest import IngestStrategy

def ingest_data(source: str) -> Tuple[pd.DataFrame, str]:
    try:
        ingest_strategy = IngestStrategy()
        data, size = ingest_strategy.handle_ingest(source)
        return data, size
    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        raise RuntimeError(f"Data ingestion failed: {e}")