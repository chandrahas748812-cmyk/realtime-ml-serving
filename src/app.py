"""FastAPI inference service."""

import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .features import engineer_features, get_cached_prediction, set_cached_prediction
from .model import get_model
from .monitoring import Monitor

# Filled in at startup from the training baseline
monitor: Monitor | None = None


class PredictRequest(BaseModel):
    amount: float = Field(gt=0)
    merchant_category: str = "unknown"
    hour_of_day: int = Field(ge=0, le=23)
    days_since_last: float = 0
    avg_amount_30d: float = 0


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    latency_ms: float
    cached: bool
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global monitor
    model = get_model()
    bundle = model.bundle
    monitor = Monitor(
        baseline_means=bundle.get("baseline_means", [0] * 8),
        baseline_stds=bundle.get("baseline_stds", [1] * 8),
    )
    yield


app = FastAPI(title="Real-Time Fraud Scoring API", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    event = req.model_dump()
    model = get_model()

    cached_proba = get_cached_prediction(event)
    if cached_proba is not None:
        proba, was_cached = cached_proba, True
        features = engineer_features(event)  # still log features for drift
    else:
        features = engineer_features(event)
        proba = float(model.predict_proba(features)[0])
        set_cached_prediction(event, proba)
        was_cached = False

    latency_ms = (time.perf_counter() - t0) * 1000
    monitor.log(features, proba, latency_ms)

    return PredictResponse(
        fraud_probability=round(proba, 4),
        is_fraud=proba >= model.threshold,
        latency_ms=round(latency_ms, 2),
        cached=was_cached,
        model_version=model.version,
    )


@app.get("/health")
def health():
    model = get_model()
    return {"status": "ok", "model_version": model.version}


@app.get("/metrics")
def metrics():
    return {
        "latency": monitor.latency_stats(),
        "drift": monitor.drift_status(),
        "model_version": get_model().version,
    }
