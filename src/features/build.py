import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from src.utils.logger import get_logger
from src.data.preprocess import preprocess

log = get_logger(__name__)

NUMERIC_COLS = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_COLS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

ENGINEERED_NUMERIC = ["credit_per_month", "credit_to_age"]
ENGINEERED_BINARY = ["high_commitment"]

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["credit_per_month"] = df["credit_amount"] / df["duration"].clip(lower=1)
    df["credit_to_age"] = df["credit_amount"] / df["age"].clip(lower=1)
    df["high_commitment"] = (df["installment_commitment"] >= 4).astype(int)
    return df

def build_features(
    df: pd.DataFrame,
    target_col: str = "target",
    fit: bool = True,
    artifacts: dict | None = None
) -> tuple[pd.DataFrame, pd.Series, dict]:
    if not fit and artifacts is None:
        raise ValueError("Must provide artifacts when fit=False")
    
    y = df[target_col].copy()
    df = df.drop(columns=[target_col])

    df = _engineer_features(df)

    all_numeric = NUMERIC_COLS + ENGINEERED_NUMERIC
    all_features = all_numeric + ENGINEERED_BINARY + CATEGORICAL_COLS

    if fit:
        artifacts = {}

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(df[CATEGORICAL_COLS].astype(str))
        artifacts["encoder"] = encoder

        scaler = RobustScaler()
        scaler.fit(df[all_numeric])
        artifacts["scaler"] = scaler

        artifacts["feature_columns"] = all_features
        log.info("fitted transformers on %d rows", df.shape[0])

    df[CATEGORICAL_COLS] = artifacts["encoder"].transform(df[CATEGORICAL_COLS].astype(str))
    df[all_numeric] = artifacts["scaler"].transform(df[all_numeric])

    X = df[artifacts["feature_columns"]]

    log.info("built features: %d rows, %d features, fit=%s", X.shape[0], X.shape[1], fit)
    return X, y, artifacts