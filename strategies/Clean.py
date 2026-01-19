# strategies/Clean.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union
import logging 
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from river import preprocessing, compose
from sklearn.preprocessing import LabelEncoder


class CleanClass(ABC):
  def __init__(self, logger: logging.Logger = None):
    self.logger = logger or logging.getLogger(__name__)
  
  @abstractmethod
  def handle_data(self, data: pd.DataFrame, size : str = None) -> pd.DataFrame:
    pass

class RemoveIdentifiersStrategy(CleanClass):
  def __init__(self, logger: logging.Logger = None):
    super().__init__(logger)

  def handle_data(self, data: pd.DataFrame, ids: List[str] = None) -> pd.DataFrame:
    self.logger.info(f"Removing identifier columns: {ids}")
    if ids is not None:
        data = data.drop(columns=ids, errors='ignore')
    else: 
      for column in data.columns:
        if data[column].nunique() >= data.shape[0] * 0.9:
          data = data.drop(columns=[column])
          self.logger.info(f"Dropped identifier column: {column}")
    return data

class MissingValueStrategy(CleanClass):
  def __init__(self, logger: logging.Logger = None):
    super().__init__(logger)

  def handle_data(self, data: pd.DataFrame, size : str) -> pd.DataFrame:
    self.logger.info(f"checking the null values:")
    records = data.shape[0]
    for column in data.columns:
      nb_nulls = data[column].isnull().sum()
      if size in 'very small':
        if nb_nulls <= 0.1* records:
          data = data.dropna(subset=[column])
        elif nb_nulls <= 0.3* records:
          if data[column].dtype in [object, 'category']:
            mode = data[column].mode()[0]
            data[column].fillna(mode, inplace=True)
          else:
            mean = data[column].mean()
            data[column].fillna(mean, inplace=True)
        else:
          data = data.drop(columns=[column])
      if size == 'small':
        if nb_nulls <= 0.2* records:
          data = data.dropna(subset=[column])
        elif nb_nulls <= 0.4* records:
          if data[column].dtype in [object, 'category']:
            mode = data[column].mode()[0]
            data[column].fillna(mode, inplace=True)
          else:
            mean = data[column].mean()
            data[column].fillna(mean, inplace=True)
        else:
          data = data.drop(columns=[column])
      if size == 'medium':
        if nb_nulls <= 0.3 * records:
          if data[column].dtype in [object, 'category']:
            mode = data[column].mode()[0]
            data[column].fillna(mode, inplace=True)
          else:
            mean = data[column].mean()
            data[column].fillna(mean, inplace=True)
        elif nb_nulls <= 0.4 * records:
          data = data.dropna(subset=[column])
        else:
          data = data.drop(columns=[column])
      if size == 'large':
        if nb_nulls <= 0.1* records:
          if data[column].dtype in [object, 'category']:
            mode = data[column].mode()[0]
            data[column].fillna(mode, inplace=True)
          else: 
            mean = data[column].mean()
            data[column].fillna(mean, inplace=True)
        elif nb_nulls <= 0.45* records:
          data = data.dropna(subset=[column])
        else:
          data = data.drop(columns=[column])
    return data

class EncodingStrategy(CleanClass):
  def __init__(self, logger: logging.Logger = None):
    super().__init__(logger)
    
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    self.logger.info(f"Encoding Categorical Columns: {target}")
    if target is not None:
      X = data.drop(columns=[target])
    else:
      X = data.copy()
    cat_col = X.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in cat_col:
      X[col] = le.fit_transform(X[col])
    if target is not None:
      return pd.concat([X, data[target]], axis=1)
    else:
      return X

class StreamCleanClass(ABC):
  def __init__(self, logger: logging.Logger = None):
    self.logger = logger or logging.getLogger(__name__)
  
  @abstractmethod
  def learn_chunk(self, data: pd.DataFrame):
    pass
  
  @abstractmethod
  def transform_chunk(self, data: pd.DataFrame) -> pd.DataFrame:
    pass

class StreamPreprocessingStrategy(StreamCleanClass):
  def __init__(self, logger):
    self.logger = logger
    self.scaler = preprocessing.StandardScaler()

  def learn_chunk(self, records):
    for x in records:
      # All features are numeric after LabelEncoder
      self.scaler.learn_one(x)

  def transform_chunk(self, records):
    out = []
    for x in records:
      # Transform all features (they're all numeric)
      x_proc = self.scaler.transform_one(x)
      out.append(dict(x_proc))
    return out


class DimensionReductionStrategy(CleanClass):
  def __init__(self, logger: logging.Logger = None):
    super().__init__(logger)
    
  def handle_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
    self.logger.info('Applying PCA for Dimensionality Reduction ...')
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    input_data = data.drop(columns=[target])
    data_pca = pca.fit_transform(input_data)
    pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])])
    data = pd.concat([pca_df, data[target]], axis=1)
    return data

class SplittingStrategy(CleanClass):
  def __init__(self, logger: logging.Logger = None):
    super().__init__(logger)
    
  def handle_data(self, data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    from sklearn.model_selection import train_test_split
    self.logger.info('Splitting data into train and test sets ...')
    X = data.drop(columns=[target])
    y = data[target]
    return X, y
  
class RemoveOutliersStrategy(CleanClass):
  def __init__(self, logger: logging.Logger = None):
    super().__init__(logger)
    
  def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
    try: 
      for column in data.columns:
        if data[column].dtype in [np.float64, np.float32, np.int64, np.int32]:
          Q1 = data[column].quantile(0.25)
          Q3 = data[column].quantile(0.75)
          IQR = Q3 - Q1
          lower_bound = Q1 - 1.5 * IQR
          upper_bound = Q3 + 1.5 * IQR
          data = data[(data[column] > lower_bound) & (data[column] < upper_bound)]
      return data
    except Exception as e:
      raise RuntimeError(f"Error in IQR outlier detection: {e}")