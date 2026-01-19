# strategies/Evaluate.py
import pandas as pd 
import logging
from typing import Union, Tuple
from abc import ABC, abstractmethod
from river import metrics

class EvaluateClass(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def update(self, y_true, y_pred):
        pass

    @abstractmethod
    def get(self) -> float:
        pass
    
class RegressionEvaluateStrategy(EvaluateClass):
    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)
        self.metric = metrics.R2()

    def update(self, y_true, y_pred):
        self.metric.update(y_true, y_pred)

    def get(self) -> float:
        return self.metric.get()
      
class ClassificationEvaluateStrategy(EvaluateClass):
    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)
        self.metric = metrics.Accuracy()

    def update(self, y_true, y_pred):
        self.metric.update(y_true, y_pred)

    def get(self) -> float:
        return self.metric.get()