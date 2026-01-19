# strategies/Predict.py
import pandas as pd
import logging
from abc import ABC, abstractmethod

class PredictClass(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def handle_predict(self, X: pd.DataFrame) -> pd.Series:
        pass


class PredictStrategy(PredictClass):
    def __init__(self, trained_model: object, chunk_size: int = 100, logger: logging.Logger = None):
        super().__init__(logger)
        self.model = trained_model
        self.chunk_size = chunk_size  # You can adjust chunk size as needed

    def handle_predict(self, X: pd.DataFrame) -> pd.Series:
        self.logger.info("Making predictions with River model...")
        self.logger.info(f"Input shape: {X.shape}")

        model = self.model

        if not hasattr(model, "preprocessor"):
            raise AttributeError("Trained model has no attached preprocessor")

        preprocessor = model.preprocessor
        predictions = []
        def chunk_generator(df: pd.DataFrame):
                    for start in range(0, df.shape[0], self.chunk_size):
                        yield df.iloc[start:start + self.chunk_size]

        for chunk in chunk_generator(X):
            X_chunk = chunk

            # Convert ONCE
            X_records = X_chunk.to_dict(orient="records")

            # Preprocessing (River-style)
            X_proc = preprocessor.transform_chunk(X_records)

            for x in X_proc:
                y_pred = model.predict_one(x)
                predictions.append(y_pred)

        result = pd.Series(predictions, index=X.index)
        self.logger.info(f"Predictions completed. Shape: {result.shape}")
        return result
