import json
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import config
from .db import Prediction, get_db, init_db
from .monitor import drift_from_db
from .train import train


class Transaction(BaseModel):
    amount: float = Field(ge=0, examples=[120.5])
    hour: int = Field(ge=0, le=23, examples=[14])
    distance_from_home_km: float = Field(ge=0, examples=[4.2])
    merchant_risk: float = Field(ge=0, le=1, examples=[0.1])
    txn_count_24h: int = Field(ge=0, examples=[2])
    is_foreign: int = Field(ge=0, le=1, examples=[0])
    card_age_days: int = Field(ge=0, examples=[900])


class PredictionOut(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str
    latency_ms: float


class ModelStore:
    model = None
    meta: dict = {}

    def load(self) -> None:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        self.model = mlflow.xgboost.load_model(f"models:/{config.MODEL_NAME}@champion")
        self.meta = json.loads(config.META_PATH.read_text())


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if not config.META_PATH.exists():  # fresh clone: train once so the API can start
        train()
    store.load()
    yield


app = FastAPI(title="Fraud Detection MLOps API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": store.model is not None}


@app.get("/model")
def model_info():
    return store.meta


@app.post("/predict", response_model=PredictionOut)
def predict(tx: Transaction, db: Session = Depends(get_db)):
    start = time.perf_counter()
    row = pd.DataFrame([tx.model_dump()])[config.FEATURES]
    proba = float(store.model.predict_proba(row)[0, 1])
    latency = (time.perf_counter() - start) * 1000
    is_fraud = proba >= store.meta["threshold"]
    version = str(store.meta["version"])
    db.add(
        Prediction(
            **tx.model_dump(), fraud_probability=proba, is_fraud=is_fraud,
            latency_ms=latency, model_version=version,
        )
    )
    db.commit()
    return PredictionOut(
        fraud_probability=round(proba, 5), is_fraud=is_fraud,
        threshold=round(store.meta["threshold"], 4), model_version=version,
        latency_ms=round(latency, 2),
    )


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Prediction.id))) or 0
    flagged = db.scalar(select(func.count(Prediction.id)).where(Prediction.is_fraud.is_(True))) or 0
    lat = db.scalars(select(Prediction.latency_ms).order_by(Prediction.id.desc()).limit(1000)).all()
    recent = db.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(10)).all()
    return {
        "total_predictions": total,
        "flagged": flagged,
        "flag_rate": round(flagged / total, 4) if total else 0,
        "avg_latency_ms": round(float(np.mean(lat)), 2) if lat else 0,
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 2) if lat else 0,
        "recent": [
            {
                "id": r.id,
                "amount": r.amount,
                "hour": r.hour,
                "distance_from_home_km": r.distance_from_home_km,
                "fraud_probability": round(r.fraud_probability, 4),
                "is_fraud": r.is_fraud,
                "model_version": r.model_version,
            }
            for r in recent
        ],
    }


@app.get("/drift")
def drift():
    return drift_from_db()


def _retrain() -> None:
    train(drift_mix=True, seed=int(time.time()) % 100000)
    store.load()


@app.post("/retrain", status_code=202)
def retrain(background: BackgroundTasks, x_api_key: str = Header(default="")):
    if x_api_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    background.add_task(_retrain)
    return {"status": "retraining started"}
