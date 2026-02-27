import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

log = get_logger(__name__)

def split(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=seed,
        stratify=df[target_col]
    )

    adjusted_val_size = val_size / (1 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=train_val[target_col]
    )

    log.info(
        "split data: train=%d, val=%d, test=%d",
        train.shape[0], val.shape[0], test.shape[0]
    )

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        pos_rate = subset[target_col].mean()
        log.info("%s positive rate: %.3f", name, pos_rate)

    return train, val, test