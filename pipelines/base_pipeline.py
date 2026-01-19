# pipelines/base_pipeline.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from implementation.Cleaning import basic_clean_data
from implementation.Training import train_model, retrain_model
from implementation.Exporting import export_model
from implementation.Predicting import predict_data
from implementation.Ingesting import ingest_data
import logging
from typing import List
import pandas as pd
from pandas import Series
import pickle



def base_pipeline(source: str, model: object, target: str, export_path: str, ids: List[str] = None, buffer_path: str = None) -> float:
    # Ingest Data
    data, size = ingest_data(source)

    # Clean Data
    data, _= basic_clean_data(data, size, target, ids)

    # Train Model
    trained_model, score = train_model(data, target, model, 500, buffer_path)
    export_model(trained_model, export_path)
    logging.info(f"Model exported to {export_path}")

    return score

def predict_pipeline(data_source: str, trained_model_path: str, ids: List[str] = None) -> pd.DataFrame:
    X, size = ingest_data(data_source)
    data, id= basic_clean_data(X, size, target=None, ids=ids)

    # Load Trained Model
    with open(trained_model_path, "rb") as f:
        trained_model = pickle.load(f)
   # Make Predictions
    predictions = predict_data(data, trained_model)
    df = pd.DataFrame({'id': id, 'prediction': predictions})
    return df


def train_more_pipeline(source: str, trained_model_path: str, buffer_path: str, target: str, ids: List[str] = None) -> float:
    
    data, size = ingest_data(source)
    data, _= basic_clean_data(data, size, target, ids)
    with open(trained_model_path, "rb") as f:
        trained_model = pickle.load(f)
    
    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)

    updated_model, score = retrain_model(data, trained_model, 500, buffer)
    export_model(updated_model, trained_model_path)
    logging.info(f"Updated model exported to {trained_model_path}")
    return score