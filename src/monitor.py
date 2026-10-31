"""Drift monitoring: compare recent live traffic with the training reference data.

CLI:  python -m src.monitor            # print drift report
      python -m src.monitor --retrain  # retrain automatically if drift detected
"""
import json
import sys

import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import select

from . import config
from .db import Prediction, SessionLocal

MIN_ROWS = 50


def check_drift(current: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict:
    if len(current) < MIN_ROWS:
        return {"status": "insufficient_data", "rows": len(current), "min_rows": MIN_ROWS}
    if reference is None:
        reference = pd.read_csv(config.REFERENCE_PATH)

    features = {}
    for col in config.FEATURES:
        stat, p = ks_2samp(reference[col], current[col])
        features[col] = {
            "ks_stat": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "drifted": bool(p < 0.01 and stat > 0.1),
        }
    share = sum(f["drifted"] for f in features.values()) / len(features)
    return {
        "status": "ok",
        "rows": len(current),
        "drifted_share": round(share, 3),
        "threshold": config.DRIFT_SHARE_THRESHOLD,
        "dataset_drift": share >= config.DRIFT_SHARE_THRESHOLD,
        "features": features,
        "report_path": _evidently_report(reference, current),
    }


def _evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> str | None:
    """Optional HTML report; needs `pip install evidently==0.4.40`."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        config.REPORT_DIR.mkdir(exist_ok=True)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference[config.FEATURES], current_data=current[config.FEATURES])
        path = config.REPORT_DIR / "drift_report.html"
        report.save_html(str(path))
        return str(path)
    except Exception:
        return None


def drift_from_db(limit: int = 500) -> dict:
    with SessionLocal() as db:
        cols = [getattr(Prediction, f) for f in config.FEATURES]
        rows = db.execute(select(*cols).order_by(Prediction.id.desc()).limit(limit)).all()
    return check_drift(pd.DataFrame([tuple(r) for r in rows], columns=config.FEATURES))


if __name__ == "__main__":
    result = drift_from_db()
    print(json.dumps(result, indent=2))
    if "--retrain" in sys.argv and result.get("dataset_drift"):
        from .train import train

        train(drift_mix=True)
