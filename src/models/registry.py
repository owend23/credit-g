import json
from pathlib import Path
from datetime import datetime

import joblib

from src.utils.logger import get_logger

log = get_logger(__name__)

def register_model(
    model,
    metrics: dict,
    feature_artifacts: dict,
    output_dir: str = "artifacts/models"
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "model.joblib")

    joblib.dump(feature_artifacts, output_dir / "feature_artifacts.joblib")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "feature_columns": feature_artifacts["feature_columns"],
        "model_type": type(model).__name__,
        "model_params": model.get_params()
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info("registered model to %s", output_dir)
    return output_dir

def load_model(model_dir: str = "artifacts/models") -> tuple:
    model_dir = Path(model_dir)

    model = joblib.load(model_dir / "model.joblib")
    feature_artifacts = joblib.load(model_dir / "feature_artifacts.joblib")

    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    log.info("loaded model from %s", model_dir)
    return model, feature_artifacts, metadata