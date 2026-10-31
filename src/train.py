"""Training pipeline: data -> XGBoost -> MLflow tracking -> model registry ('champion' alias)."""
import json
from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import config
from .data import make_transactions


def train(n_samples: int = 20000, seed: int = 42, drift_mix: bool = False) -> dict:
    """drift_mix=True simulates retraining on fresh data that includes the new regime.
    In production this step would pull recently labelled data from your warehouse."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    if client.get_experiment_by_name(config.EXPERIMENT) is None:
        client.create_experiment(
            config.EXPERIMENT, artifact_location=(config.BASE / "mlruns").as_uri()
        )
    mlflow.set_experiment(config.EXPERIMENT)

    if drift_mix:
        df = pd.concat(
            [
                make_transactions(n_samples // 2, seed),
                make_transactions(n_samples // 2, seed + 1, drift=True),
            ],
            ignore_index=True,
        )
    else:
        df = make_transactions(n_samples, seed)

    X, y = df[config.FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    params = dict(
        n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, eval_metric="aucpr", random_state=seed,
        scale_pos_weight=float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
    )

    with mlflow.start_run(run_name=f"xgb-{datetime.now():%Y%m%d-%H%M%S}") as run:
        model = XGBClassifier(**params).fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        # pick the decision threshold that maximises F1 on the held-out set
        prec, rec, thr = precision_recall_curve(y_te, proba)
        f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        threshold = float(thr[int(np.argmax(f1s[:-1]))])
        pred = (proba >= threshold).astype(int)

        metrics = {
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
            "fraud_rate": float(y.mean()),
        }
        mlflow.log_params({**params, "threshold": round(threshold, 4), "n_samples": len(df)})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model, artifact_path="model", registered_model_name=config.MODEL_NAME
        )

        versions = client.search_model_versions(f"name='{config.MODEL_NAME}'")
        version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(config.MODEL_NAME, "champion", str(version))

        config.DATA_DIR.mkdir(exist_ok=True)
        X_tr.sample(min(5000, len(X_tr)), random_state=seed).to_csv(
            config.REFERENCE_PATH, index=False
        )
        meta = {
            "version": version,
            "run_id": run.info.run_id,
            "threshold": threshold,
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_train": len(X_tr),
        }
        config.META_PATH.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    train()
