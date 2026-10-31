"""Synthetic card-transaction generator (swap for a real dataset such as the
Kaggle credit-card fraud data by returning a DataFrame with FEATURES + is_fraud)."""
import numpy as np
import pandas as pd

from .config import FEATURES


INTERCEPT = -7.0


def make_transactions(n: int, seed: int = 42, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.6, 1.0, n)
    hour = rng.integers(0, 24, n)
    distance = rng.exponential(15, n)
    risk = rng.beta(2, 8, n)
    count = rng.poisson(6 if drift else 3, n)
    foreign = rng.binomial(1, 0.08, n)
    age = rng.integers(10, 3000, n)
    if drift:  # covariate shift: bigger amounts, farther away, more activity
        amount = amount * 2.5
        distance = distance * 2.0

    night = (hour < 5).astype(int)
    logit = (
        INTERCEPT + 0.003 * amount + 0.03 * distance + 9.0 * risk + 0.3 * count
        + 2.5 * foreign + 2.5 * night - 0.0008 * age
    )
    # label = noisy threshold on the risk score (small noise keeps the problem learnable but not trivial)
    fraud = ((logit + rng.normal(0, 0.7, n)) > 0).astype(int)

    df = pd.DataFrame(
        {
            "amount": amount.round(2),
            "hour": hour,
            "distance_from_home_km": distance.round(1),
            "merchant_risk": risk.round(3),
            "txn_count_24h": count,
            "is_foreign": foreign,
            "card_age_days": age,
        }
    )[FEATURES]
    df["is_fraud"] = fraud
    return df
