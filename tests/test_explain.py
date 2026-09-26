"""Tests for the SHAP explainability step (part 3)."""

import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.features.build_features import feature_frame
from churn.models import explain, train_model


@pytest.fixture(name="fitted")
def fixture_fitted():
    """A fitted LightGBM pipeline plus the matrices the explainer needs."""
    rng = np.random.default_rng(3)
    rows = 150
    data = {config.ID_COLUMN: range(rows)}
    for column in config.NUMERIC_FEATURES:
        data[column] = rng.integers(0, 20, rows).astype(float)
    for column in config.CATEGORICAL_FEATURES:
        data[column] = rng.choice(["a", "b"], rows)
    customers = pd.DataFrame(data)
    customers[config.TARGET] = (customers["Complain"] > 12).astype(int)

    pipeline = train_model.build_pipeline("lightgbm", n_estimators=40)
    pipeline.fit(feature_frame(customers), customers[config.TARGET])
    preprocessor = pipeline.named_steps["preprocess"]
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    features = pd.DataFrame(preprocessor.transform(feature_frame(customers)), columns=names)
    return pipeline, features, customers


def test_display_frame_restores_original_units(fitted):
    pipeline, features, customers = fitted
    display = explain.display_frame(pipeline.named_steps["preprocess"], features)
    assert display.shape == features.shape
    # Scaled values are centred on zero; the real Tenure values are not.
    assert display["Tenure"].max() == pytest.approx(customers["Tenure"].max(), abs=0.01)
    assert set(display["Complain"].unique()) <= set(customers["Complain"].unique())


def test_shapley_values_are_additive(fitted):
    """SHAP's guarantee: base value + contributions = the model's raw output."""
    pipeline, features, _ = fitted
    estimator = pipeline.named_steps["model"]
    _, explanation = explain.shapley_values(estimator, features)
    assert explanation.values.shape == features.shape
    raw = estimator.predict_proba(features, raw_score=True)
    assert np.allclose(explanation.values.sum(axis=1) + explanation.base_values, raw, atol=1e-6)


def test_reason_codes_rank_by_magnitude_and_state_direction(fitted, tmp_path, monkeypatch):
    pipeline, features, customers = fitted
    monkeypatch.setattr(explain, "REASONS_FILE", tmp_path / "reasons.csv")
    estimator = pipeline.named_steps["model"]
    _, explanation = explain.shapley_values(estimator, features)
    frame = explain.reason_codes(estimator, explanation, features, customers)

    assert len(frame) == len(customers)
    assert (explain.REASONS_FILE).exists()
    assert frame["churn_probability"].is_monotonic_decreasing
    for reason in frame["reason_1"]:
        assert ("raises risk" in reason) or ("lowers risk" in reason)
    # The first reason must never be weaker than the second.
    first = frame["reason_1"].str.extract(r"([+-]\d+\.\d+)")[0].astype(float).abs()
    second = frame["reason_2"].str.extract(r"([+-]\d+\.\d+)")[0].astype(float).abs()
    assert (first >= second - 1e-9).all()


def test_fairness_audit_reports_one_row_per_group(fitted, tmp_path, monkeypatch):
    pipeline, features, customers = fitted
    monkeypatch.setattr(explain, "FAIRNESS_FILE", tmp_path / "fairness.csv")
    estimator = pipeline.named_steps["model"]
    _, explanation = explain.shapley_values(estimator, features)
    probabilities = estimator.predict_proba(features)[:, 1]
    audit = explain.fairness_audit(explanation, features, customers, probabilities)

    assert set(audit["attribute"]) <= set(explain.SENSITIVE_ATTRIBUTES)
    assert audit.groupby("attribute")["customers"].sum().eq(len(customers)).all()
    # Disparate impact is a ratio to the most-flagged group, so it never exceeds 1.
    assert (audit["disparate_impact"] <= 1.0).all()
    assert audit["disparate_impact"].max() == pytest.approx(1.0)


def test_prepare_rejects_a_non_tree_model(monkeypatch):
    """A linear champion must fail loudly rather than silently mis-explain."""
    monkeypatch.setattr(
        explain,
        "load_model",
        lambda alias="champion": train_model.build_pipeline("logistic_regression"),
    )
    with pytest.raises(TypeError, match="TreeExplainer"):
        explain.prepare()
