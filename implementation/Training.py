# implementation/Training.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
import pandas as pd
from strategies.Train import TrainStrategy
from strategies.ReplayBuffer import ReplayBuffer
from river import linear_model, ensemble
from implementation.Exporting import export_model


def create_river_model(model_name: str, target: str):
    """Factory to create River models with a target attribute."""
    if model_name == 'RandomForestRegressor':
        model = ensemble.AdaptiveRandomForestRegressor(seed=42, leaf_prediction= 'mean')
    elif model_name == 'RandomForestClassifier':
        model = ensemble.AdaptiveRandomForestClassifier(seed=42)
    else:
        raise ValueError(f"Unknown model: {model_name}.")
    model.target = target
    return model

def train_model(data: pd.DataFrame, target: str, model_name: str, chunk_size: int = 100, buffer_path: str = None):
    buffer = ReplayBuffer(max_per_class=20)
    logger = logging.getLogger(__name__)
    model = create_river_model(model_name, target)
    trainer = TrainStrategy(model, chunk_size, buffer, logger)
    trained_model, score = trainer.handle_train(data)
    export_model(buffer, buffer_path)
    return trained_model, score

def retrain_model(data: pd.DataFrame, trained_model: object, chunk_size: int = 100, buffer: object = None):
    logger = logging.getLogger(__name__)
    trainer = TrainStrategy(trained_model, chunk_size, buffer, is_train_more=True, logger = logger)
    updated_model, score = trainer.handle_train(data)
    return updated_model, score