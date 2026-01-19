# implementation/Cleaning.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import logging
from strategies.Clean import MissingValueStrategy, DimensionReductionStrategy, StreamPreprocessingStrategy, SplittingStrategy, RemoveOutliersStrategy, RemoveIdentifiersStrategy, EncodingStrategy
from typing import List, Tuple, Union

def basic_clean_data(data: pd.DataFrame, size: str, target: str, ids: List[str] = None):
    logger = logging.getLogger(__name__)
    remove_id_strategy = RemoveIdentifiersStrategy(logger)
    missing_strategy = MissingValueStrategy(logger)
    outliers_strategy = RemoveOutliersStrategy(logger)  
    encode_strategy = EncodingStrategy(logger)
    # splitting_strategy = SplittingStrategy(logger)
    
    data = remove_id_strategy.handle_data(data, ids)
    data['id'] = data.index
    data = missing_strategy.handle_data(data, size=size)
    data = outliers_strategy.handle_data(data)
    data = encode_strategy.handle_data(data, target)
    id = data['id']
    data = data.drop(columns=['id'])
    
    return data, id