import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
REPORT_DIR = BASE / "reports"
REFERENCE_PATH = DATA_DIR / "reference.csv"
META_PATH = DATA_DIR / "meta.json"

MODEL_NAME = "fraud-detector"
EXPERIMENT = "fraud-detection"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(BASE / 'mlflow.db').as_posix()}"
)

_db = os.getenv("DATABASE_URL", f"sqlite:///{(BASE / 'predictions.db').as_posix()}")
if _db.startswith("postgres://"):
    _db = _db.replace("postgres://", "postgresql+psycopg2://", 1)
elif _db.startswith("postgresql://"):
    _db = _db.replace("postgresql://", "postgresql+psycopg2://", 1)
DATABASE_URL = _db

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "dev-key")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

DRIFT_SHARE_THRESHOLD = 0.25  # share of drifted features that triggers retraining

FEATURES = [
    "amount",
    "hour",
    "distance_from_home_km",
    "merchant_risk",
    "txn_count_24h",
    "is_foreign",
    "card_age_days",
]
