import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns={"class": "target"})

    df["target"] = df["target"].map({"good": 1, "bad": 0}).astype(int)

    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype(str).str.strip().astype("category")
    
    n_before = df.shape[0]
    df = df.drop_duplicates()
    n_dropped = n_before - df.shape[0]
    if n_dropped > 0:
        log.info("dropped %d duplicate rows", n_dropped)
    
    df = df.dropna(subset=["target"])

    log.info("preprocessed data: rows=%d, cols=%d", df.shape[0], df.shape[1])
    return df