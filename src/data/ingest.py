from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from src.utils.logger import get_logger

log = get_logger(__name__)

RAW_PATH = Path("data/raw/data.parquet")

def ingest(
    source_id: int = 31,
    sample_frac: float = 1.0,
    seed: int = 42,
    cache_path: Path = RAW_PATH,
) -> pd.DataFrame:
    if cache_path.exists():
        log.info("loading from cache: %s", cache_path)
        df = pd.read_parquet(cache_path)
    else:
        log.info("fetching from OpenML, dataset_id=%d", source_id)
        data = fetch_openml(data_id=source_id, as_frame=True)
        df = data.frame

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log.info("cached raw data to %s", cache_path)
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
        log.info("sampled %.0f%% of data, rows=%d", sample_frac * 100, len(df))
    
    log.info("ingested data: rows=%d, cols=%d", df.shape[0], df.shape[1])
    return df