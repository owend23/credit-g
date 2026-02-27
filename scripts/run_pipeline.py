from src.utils.logger import setup_logging, get_logger
from src.data.ingest import ingest
from src.data.validate import validate, get_raw_schema
from src.data.preprocess import preprocess
from src.data.split import split
from src.features.build import build_features
from src.models.train import train
from src.models.tune import tune
from src.evaluation.metrics import compute_metrics
from src.evaluation.report import generate_report
from src.models.registry import register_model

setup_logging()
log = get_logger("pipeline")

def main():
    log.info("starting pipeline")
    df = ingest()

    df = validate(df, get_raw_schema(), stage="raw")

    df = preprocess(df)

    train_df, val_df, test_df = split(df)

    X_train, y_train, artifacts = build_features(train_df, fit=True)
    X_val, y_val, _ = build_features(val_df, fit=False, artifacts=artifacts)
    X_test, y_test, _ = build_features(test_df, fit=False, artifacts=artifacts)

    best_params = tune(X_train, y_train, X_val, y_val, n_trials=50)

    model = train(X_train, y_train, X_val, y_val, params=best_params)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    generate_report(y_test, y_pred, y_proba, model, X_test.columns.tolist())

    register_model(model, metrics, artifacts)

    log.info("pipeline complete")


if __name__ == "__main__":
    main()