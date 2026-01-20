# strategies/Train.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import logging
from abc import ABC, abstractmethod
from strategies.Clean import StreamPreprocessingStrategy
from strategies.Evaluate import RegressionEvaluateStrategy, ClassificationEvaluateStrategy
from river.base import Regressor, Classifier

class TrainClass(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def handle_train(self, data: pd.DataFrame) -> object:
        pass

class TrainStrategy(TrainClass):
    def __init__(
        self,
        model,
        chunk_size: int = 100,
        replay_buffer=None,
        is_train_more: bool = False,
        logger: logging.Logger = None
    ):
        super().__init__(logger)
        self.model = model
        self.chunk_size = chunk_size
        self.replay_buffer = replay_buffer
        self.is_train_more = is_train_more


    def handle_train(self, data: pd.DataFrame) -> object:
        preprocessor = StreamPreprocessingStrategy(self.logger)
        model = self.model
        target = model.target

        if isinstance(model, Regressor):
            evaluator = RegressionEvaluateStrategy(self.logger)
        elif isinstance(model, Classifier):
            evaluator = ClassificationEvaluateStrategy(self.logger)
        else:
            raise TypeError("Model must be a River Regressor or Classifier")

        def chunk_generator(df: pd.DataFrame):
            for start in range(0, df.shape[0], self.chunk_size):
                yield df.iloc[start:start + self.chunk_size]

        warmup = 1000
        n_seen = 0

        # 🔁 REPLAY PHASE (ONLY during train-more)
        if self.is_train_more and self.replay_buffer and not self.replay_buffer.is_empty():
            self.logger.info("Replaying past samples to prevent forgetting")
            for x_old, y_old in self.replay_buffer.replay():
                model.learn_one(x_old, y_old)

        # 🚀 NORMAL TRAINING LOOP
        for chunk in chunk_generator(data):
            y_chunk = chunk[target].tolist()
            X_chunk = chunk.drop(columns=[target])
            X_records = X_chunk.to_dict(orient="records")

            # preprocessing
            preprocessor.learn_chunk(X_records)
            X_proc = preprocessor.transform_chunk(X_records)

            for x, y in zip(X_proc, y_chunk):
                model.learn_one(x, y)

                # ✅ store for future replay
                if self.replay_buffer is not None:
                    self.replay_buffer.add(x, y)

                if n_seen > warmup:
                    y_pred = model.predict_one(x)
                    evaluator.update(y, y_pred)

                n_seen += 1

        score = evaluator.get()
        model.preprocessor = preprocessor

        return model, score
