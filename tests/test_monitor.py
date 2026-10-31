import pytest

from src.data import make_transactions
from src.monitor import check_drift
from src.train import train


@pytest.fixture(scope="module", autouse=True)
def trained():
    train(n_samples=6000)


def test_no_drift_on_same_distribution():
    assert check_drift(make_transactions(400, seed=7))["dataset_drift"] is False


def test_drift_detected_on_shifted_traffic():
    result = check_drift(make_transactions(400, seed=7, drift=True))
    assert result["dataset_drift"] is True
    assert result["features"]["amount"]["drifted"] is True


def test_insufficient_data():
    assert check_drift(make_transactions(10))["status"] == "insufficient_data"
