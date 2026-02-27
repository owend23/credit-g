import pandas as pd
import pandera.pandas as pa

from src.utils.logger import get_logger

log = get_logger(__name__)

def get_raw_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        columns={
            "class": pa.Column(nullable=False, checks=pa.Check.isin(["good", "bad"])),
            "credit_amount": pa.Column(float, checks=pa.Check.gt(0)),
            "duration": pa.Column(checks=pa.Check.gt(0)),
            "age": pa.Column(checks=[pa.Check.ge(18), pa.Check.le(100)])
        },
        checks=pa.Check(lambda df: len(df) > 0, error="DataFrame is empty"),
        strict=False,
        coerce=True
    )

def validate(df: pd.DataFrame, schema: pa.DataFrameSchema, stage: str = "raw") -> pd.DataFrame:
    log.info("validating %s data: rows=%d", stage, len(df))
    validated = schema.validate(df, lazy=True)
    log.info("validation passed: %s", stage)
    return validated