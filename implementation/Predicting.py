# implementation/Predicting.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import logging
from strategies.Predict import PredictStrategy

def predict_data(data: pd.DataFrame, trained_model: object) -> pd.Series:
    
        
    logger = logging.getLogger(__name__)
    
    # Create predictor
    predictor = PredictStrategy(trained_model, chunk_size=100, logger=logger)
    
    # Make predictions
    predictions = predictor.handle_predict(data)
    
    return predictions