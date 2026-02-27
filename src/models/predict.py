import pandas as pd
import numpy as np

from src.models.registry import load_model
from src.features.build import build_features
from src.data.preprocess import preprocess
from src.utils.logger import get_logger

log = get_logger(__name__)

_model = None
_artifacts = None

def load(model_dir: str = "artifacts/models") -> None:
    """Load model and artifacts into memory. Call once at startup"""
    global _model, _artifacts
    _model, _artifacts, metadata = load_model(model_dir)
    log.info("model loaded: %s, auc=%.4f", metadata["model_type"], metadata["metrics"]["roc_auc"])

def predict(df: pd.DataFrame) -> dict:
    """Predict on raw input data.

    Args:
        df: Raw data — same format as what comes out of ingest,
            but without the target column.

    Returns:
        Dict with predictions and probabilities.
    """
    from src.features.build import _engineer_features, CATEGORICAL_COLS, NUMERIC_COLS, ENGINEERED_NUMERIC

    if _model is None or _artifacts is None:
        raise RuntimeError("Model not loaded. Call predict.load() first.")
    
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype(str).str.strip().astype("category")
    
    X = df.copy()
    X = _engineer_features(X)
    all_numeric = NUMERIC_COLS + ENGINEERED_NUMERIC

    X[CATEGORICAL_COLS] = _artifacts["encoder"].transform(X[CATEGORICAL_COLS].astype(str))
    X[all_numeric] = _artifacts["scaler"].transform(X[all_numeric])
    X = X[_artifacts["feature_columns"]]

    probabilities = _model.predict_proba(X)[:, 1]
    predictions = _model.predict(X)

    log.info("predicted %d rows", X.shape[0])

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    }
    