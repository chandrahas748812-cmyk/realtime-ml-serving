"""Model wrapper: versioned artifact loading + prediction."""

import os
from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/fraud_model.joblib"))


class FraudModel:
    def __init__(self):
        self.bundle = joblib.load(MODEL_PATH)
        self.model = self.bundle["model"]
        self.version = self.bundle.get("version", "unknown")
        self.threshold = self.bundle.get("threshold", 0.5)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


_model = None


def get_model() -> FraudModel:
    global _model
    if _model is None:
        _model = FraudModel()
    return _model
