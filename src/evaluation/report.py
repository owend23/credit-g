from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from src.utils.logger import get_logger

log = get_logger(__name__)

def generate_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model,
    feature_names: list[str],
    output_dir: str = "artifacts/reports"
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.savefig(output_dir / "roc_curve.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.savefig(output_dir / "precision_recall.png", bbox_inches="tight")
    plt.close(fig)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices]
    )
    ax.set_title("Feature Importance")
    ax.invert_yaxis()
    fig.savefig(output_dir / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    log.info("saved reports to %s", output_dir)