# strategies/Export.py
import logging
import pickle
from abc import ABC, abstractmethod


class ExportStrategy(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def handle_export(self, model: object, file_path: str) -> None:
        pass

class PickleExportStrategy(ExportStrategy):
    def __init__(self,model: object, logger: logging.Logger = None):
      super().__init__(logger)
      self.model = model

    def handle_export(self, file_path: str) -> None:
      model = self.model
      self.logger.info(f"Exporting model to {file_path} using pickle.")
      with open(file_path, 'wb') as file:
        pickle.dump(model, file)
      self.logger.info("Model exported successfully.")