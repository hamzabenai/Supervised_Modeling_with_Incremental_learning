# implementation/Exporting.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd 
import logging
from strategies.Export import PickleExportStrategy


def export_model(model: object, file_path: str) -> None:
  logger = logging.getLogger(__name__)
  exporter = PickleExportStrategy(model, logger)
  exporter.handle_export(file_path)