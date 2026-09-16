"""Tests covering the data, feature and scoring steps."""

import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.data import make_dataset
from churn.features.build_features import add_engineered_features, build_preprocessor, feature_frame
from churn.models.evaluate import compute_metrics, lift_at_k, recall_at_k
from churn.models.predict_model import risk_band, score


@pytest.fixture(name="frame")
def fixture_frame():
    """A small synthetic table with the same columns as the real dataset."""
    rows = 40
    rng = np.random.default_rng(0)
    data = {config.ID_COLUMN: range(rows), config.TARGET: [0, 1] * (rows // 2)}
    for column in config.NUMERIC_FEATURES:
        data[column] = rng.integers(0, 10, rows).astype(float)
    for column in config.CATEGORICAL_FEATURES:
        data[column] = rng.choice(["a", "b"], rows)
    return pd.DataFrame(data)


def test_harmonise_categories_merges_spellings():
    frame = pd.DataFrame(
        {
            "PreferredLoginDevice": ["Mobile Phone", "Phone"],
            "PreferredPaymentMode": ["COD", "Cash on Delivery"],
        }
    )
    out = make_dataset.harmonise_categories(frame)
    assert out["PreferredLoginDevice"].nunique() == 1
    assert out["PreferredPaymentMode"].nunique() == 1


def test_drop_duplicate_customers():
    frame = pd.DataFrame({config.ID_COLUMN: [1, 1, 2], config.TARGET: [0, 0, 1]})
    assert len(make_dataset.drop_duplicate_customers(frame)) == 2


def test_split_is_stratified(frame):
    train, test = make_dataset.split(frame, test_size=0.25)
    assert len(train) + len(test) == len(frame)
    assert train[config.TARGET].mean() == pytest.approx(test[config.TARGET].mean(), abs=0.05)


def test_engineered_features_are_finite(frame):
    out = add_engineered_features(frame)
    for column in config.ENGINEERED_FEATURES:
        assert column in out
    assert np.isfinite(out[config.ENGINEERED_FEATURES].to_numpy()).all()


def test_engineered_features_survive_zero_tenure():
    """A brand-new customer has tenure 0; the ratios must not divide by zero."""
    frame = pd.DataFrame(
        {
            "Tenure": [0.0],
            "OrderCount": [3.0],
            "CouponUsed": [1.0],
            "CashbackAmount": [10.0],
            "DaySinceLastOrder": [2.0],
        }
    )
    out = add_engineered_features(frame)
    assert np.isfinite(out[config.ENGINEERED_FEATURES].to_numpy()).all()
    assert out["IsNewCustomer"].iloc[0] == 1


def test_preprocessor_handles_unseen_category(frame):
    preprocessor = build_preprocessor().fit(feature_frame(frame))
    unseen = frame.copy()
    unseen[config.CATEGORICAL_FEATURES[0]] = "never-seen-before"
    assert preprocessor.transform(feature_frame(unseen)).shape[0] == len(frame)


def test_preprocessor_imputes_missing_values(frame):
    preprocessor = build_preprocessor().fit(feature_frame(frame))
    holes = frame.copy()
    holes.loc[0, "Tenure"] = np.nan
    assert np.isfinite(preprocessor.transform(feature_frame(holes))).all()


def test_lift_and_recall_at_k_on_a_perfect_ranking():
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.linspace(1, 0, 100)
    assert lift_at_k(y_true, y_score, 0.10) == pytest.approx(10.0)
    assert recall_at_k(y_true, y_score, 0.10) == pytest.approx(1.0)


def test_compute_metrics_keys():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]
    metrics = compute_metrics(y_true, y_score)
    assert set(metrics) == {
        "pr_auc",
        "roc_auc",
        "lift_at_10pct",
        "recall_at_10pct",
        "precision",
        "recall",
        "f1",
        "brier",
    }
    assert metrics["roc_auc"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "probability,expected", [(0.95, "HIGH"), (0.60, "HIGH"), (0.40, "MEDIUM"), (0.05, "LOW")]
)
def test_risk_band(probability, expected):
    assert risk_band(probability) == expected


def test_score_returns_one_row_per_customer(frame):
    class _Stub:
        """Minimal stand-in for a fitted estimator."""

        @staticmethod
        def predict_proba(features):
            return np.column_stack([np.full(len(features), 0.3), np.full(len(features), 0.7)])

    out = score(frame, _Stub())
    assert len(out) == len(frame)
    assert set(out["risk_band"]) == {"HIGH"}
    assert out["rank"].is_unique
