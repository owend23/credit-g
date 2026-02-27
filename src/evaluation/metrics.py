import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    log_loss,
    confusion_matrix
)

from src.utils.logger import get_logger

log = get_logger(__name__)

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> dict:
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba)
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])

    for name, value in metrics.items():
        if isinstance(value, float):
            log.info("%s: %.4f", name, value)
    
    return metrics