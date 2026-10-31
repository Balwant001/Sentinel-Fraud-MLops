import os

os.environ["DATABASE_URL"] = "sqlite:///./test_predictions.db"

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.data import make_transactions

LEGIT = dict(amount=24.5, hour=14, distance_from_home_km=3.2, merchant_risk=0.08,
             txn_count_24h=2, is_foreign=0, card_age_days=1450)
RISKY = dict(amount=1850, hour=3, distance_from_home_km=640, merchant_risk=0.85,
             txn_count_24h=11, is_foreign=1, card_age_days=35)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_risky_scores_higher_than_legit(client):
    low = client.post("/predict", json=LEGIT).json()
    high = client.post("/predict", json=RISKY).json()
    assert 0 <= low["fraud_probability"] <= 1
    assert high["fraud_probability"] > low["fraud_probability"]
    assert high["is_fraud"] is True and low["is_fraud"] is False


def test_predict_validates_input(client):
    assert client.post("/predict", json={**LEGIT, "hour": 99}).status_code == 422


def test_stats_counts_predictions(client):
    client.post("/predict", json=LEGIT)
    assert client.get("/stats").json()["total_predictions"] >= 1


def test_retrain_requires_api_key(client):
    assert client.post("/retrain").status_code == 401


def test_synthetic_data_fraud_rate_is_realistic():
    rate = make_transactions(20000, seed=1)["is_fraud"].mean()
    assert 0.005 < rate < 0.10
