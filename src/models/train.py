import xgboost as xgb
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "eval_metric": "auc",
    "random_state": 42,
    "early_stopping_rounds": 10
}

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None
) -> xgb.XGBClassifier:
    params = {**DEFAULT_PARAMS, **(params or {})}

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    best_iteration = model.best_iteration
    best_score = model.best_score

    log.info("training complete: best_iteration=%d, best_score=%.4f", best_iteration, best_score)
    return model