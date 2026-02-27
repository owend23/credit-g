# Credit Risk Classification Pipeline

End-to-end machine learning pipeline that predicts whether a credit applicant is a **good** or **bad** risk, built on the [German Credit dataset](https://www.openml.org/d/31) (1,000 applicants, 20 features). The pipeline covers every stage from raw data ingestion through model serving via a REST API.

## Pipeline Overview

```
Ingest → Validate → Preprocess → Feature Engineering → Tune → Train → Evaluate → Register → Serve
```

| Stage | Module | What it does |
|---|---|---|
| **Ingest** | `src/data/ingest.py` | Fetches the German Credit dataset from OpenML and caches it locally as Parquet |
| **Validate** | `src/data/validate.py` | Enforces schema constraints with Pandera (non-null target, positive amounts, valid age range) |
| **Preprocess** | `src/data/preprocess.py` | Encodes target (`good`→1, `bad`→0), cleans categoricals, drops duplicates |
| **Split** | `src/data/split.py` | Stratified train/val/test split (68/15/17) preserving class distribution |
| **Features** | `src/features/build.py` | Engineers 3 new features, applies OrdinalEncoder + RobustScaler, outputs 23 features total |
| **Tune** | `src/models/tune.py` | Bayesian hyperparameter search over 8 XGBoost params using Optuna (50 trials, maximizing ROC-AUC) |
| **Train** | `src/models/train.py` | Trains XGBClassifier with early stopping on the validation set |
| **Evaluate** | `src/evaluation/metrics.py` | Computes ROC-AUC, accuracy, precision, recall, F1, log loss, and confusion matrix |
| **Report** | `src/evaluation/report.py` | Generates confusion matrix, ROC curve, precision-recall curve, and feature importance plots |
| **Register** | `src/models/registry.py` | Persists model, feature artifacts, and metadata (joblib + JSON) |
| **Serve** | `src/serving/app.py` | FastAPI REST API with `/predict` and `/health` endpoints |

## Dataset

The [German Credit dataset](https://www.openml.org/d/31) contains 1,000 credit applications with 20 attributes:

- **13 categorical features**: checking account status, credit history, loan purpose, savings, employment duration, personal status, housing type, job type, etc.
- **7 numeric features**: loan duration, credit amount, installment rate, residence duration, age, number of existing credits, number of dependents
- **Target**: binary classification — `good` (70%) or `bad` (30%) credit risk

## Feature Engineering

Three features are engineered on top of the raw 20:

| Feature | Formula | Rationale |
|---|---|---|
| `credit_per_month` | `credit_amount / duration` | Monthly repayment burden |
| `credit_to_age` | `credit_amount / age` | Debt relative to borrower maturity |
| `high_commitment` | `installment_commitment >= 4` | Binary flag for high installment rate |

Categorical features are encoded with `OrdinalEncoder` (unknown categories mapped to -1). Numeric features are scaled with `RobustScaler` to handle outliers.

## Model

**XGBoost Classifier** with Optuna-tuned hyperparameters:

| Hyperparameter | Search Range | Tuned Value |
|---|---|---|
| `max_depth` | 3–10 | 7 |
| `learning_rate` | 0.01–0.3 | 0.060 |
| `subsample` | 0.6–1.0 | 0.711 |
| `colsample_bytree` | 0.6–1.0 | 0.611 |
| `min_child_weight` | 1–10 | 2 |
| `reg_alpha` | 0.001–10 | 6.053 |
| `reg_lambda` | 0.001–10 | 2.499 |

Training uses 500 estimators with early stopping (patience=10) monitored on AUC.

## Results

Evaluated on a held-out test set (200 samples):

| Metric | Score |
|---|---|
| ROC-AUC | 0.756 |
| Accuracy | 0.700 |
| Precision | 0.700 |
| Recall | 1.000 |
| F1 | 0.824 |
| Log Loss | 0.585 |

Reports (confusion matrix, ROC curve, precision-recall curve, feature importance) are saved to `artifacts/reports/`.

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── ingest.py              # OpenML data fetching + Parquet caching
│   │   ├── validate.py            # Pandera schema validation
│   │   ├── preprocess.py          # Target encoding, deduplication, type casting
│   │   └── split.py               # Stratified train/val/test splitting
│   ├── features/
│   │   └── build.py               # Feature engineering + encoding + scaling
│   ├── models/
│   │   ├── train.py               # XGBoost training with early stopping
│   │   ├── tune.py                # Optuna hyperparameter optimization
│   │   ├── predict.py             # Inference with feature pipeline
│   │   └── registry.py            # Model serialization and loading
│   ├── evaluation/
│   │   ├── metrics.py             # Classification metrics computation
│   │   └── report.py              # Matplotlib evaluation plots
│   ├── serving/
│   │   └── app.py                 # FastAPI REST API (/predict, /health)
│   └── utils/
│       └── logger.py              # Structured logging setup
├── scripts/
│   ├── run_pipeline.py            # Pipeline orchestration (entry point)
│   └── serve.py                   # Uvicorn server launcher
├── docker/
│   ├── Dockerfile.train           # Training container
│   └── Dockerfile.serve           # Serving container
├── data/raw/                      # Cached raw Parquet data
├── artifacts/
│   ├── models/                    # model.joblib, feature_artifacts.joblib, metadata.json
│   └── reports/                   # Evaluation plots (PNG)
├── .github/workflows/ci.yml      # CI: lint + pipeline smoke test
├── pyproject.toml                 # Dependencies and project metadata
└── .gitignore
```

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Run the pipeline

```bash
uv run python scripts/run_pipeline.py
```

This will ingest the data, validate, preprocess, engineer features, run 50 Optuna trials, train the final model, compute metrics, generate reports, and save everything to `artifacts/`.

### Serve the model

```bash
uv run python scripts/serve.py
```

The API starts on `http://localhost:8000`. Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "checking_status": "no checking",
    "duration": 24,
    "credit_history": "existing paid",
    "purpose": "new car",
    "credit_amount": 3500,
    "savings_status": "<100",
    "employment": "1<=X<4",
    "installment_commitment": 3,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "car",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "yes"
  }'
```

Response:

```json
{"prediction": 1, "probability": 0.8234, "label": "good"}
```

### Docker

```bash
# Train
docker build -f docker/Dockerfile.train -t credit-g-train .
docker run credit-g-train

# Serve
docker build -f docker/Dockerfile.serve -t credit-g-serve .
docker run -p 8000:8000 credit-g-serve
```

## Tech Stack

- **ML**: XGBoost, scikit-learn, Optuna
- **Data**: pandas, Pandera, OpenML
- **Serving**: FastAPI, Uvicorn, Pydantic
- **Visualization**: matplotlib, seaborn
- **Tooling**: uv, ruff, GitHub Actions
