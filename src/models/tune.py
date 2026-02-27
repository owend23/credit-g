import optuna
import xgboost as xgb 
from sklearn.metrics import roc_auc_score
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    seed: int = 42
) -> dict:
    def objective(trial):
        params = {
            "n_estimators": 500,
            "early_stopping_rounds": 10,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "eval_metric": "auc",
            "random_state": seed,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    log.info("best trial: %d", study.best_trial.number)
    log.info("best auc: %.4f", study.best_value)
    for param, value in study.best_params.items():
        log.info("  %s: %s", param, value)

    return study.best_params
