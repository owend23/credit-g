from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.predict import load, predict
from src.utils.logger import get_logger

log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load()
    log.info("model ready")
    yield

app = FastAPI(title="Credit Risk API", lifespan=lifespan)

class CreditApplication(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: float
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(application: CreditApplication):
    try:
        df = pd.DataFrame([application.model_dump()])
        result = predict(df)

        pred = result["predictions"][0]
        proba = result["probabilities"][0]

        return PredictionResponse(
            prediction=pred,
            probability=round(proba, 4),
            label="good" if pred == 1 else "bad",
        )
    except Exception as e:
        log.error("prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}