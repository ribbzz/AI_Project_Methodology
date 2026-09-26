"""Tests for the training step and the evaluation plots."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.features.build_features import feature_frame
from churn.models import train_model
from churn.visualization import visualize


@pytest.fixture(name="frames")
def fixture_frames():
    """Train and test tables where churn depends on complaints and tenure."""
    rng = np.random.default_rng(1)

    def make(rows):
        data = {config.ID_COLUMN: range(rows)}
        for column in config.NUMERIC_FEATURES:
            data[column] = rng.integers(0, 10, rows).astype(float)
        for column in config.CATEGORICAL_FEATURES:
            data[column] = rng.choice(["a", "b"], rows)
        frame = pd.DataFrame(data)
        frame[config.TARGET] = ((frame["Complain"] > 6) | (frame["Tenure"] < 2)).astype(int)
        return frame

    return make(200), make(80)


@pytest.fixture(name="tracking")
def fixture_tracking(tmp_path):
    """Isolated MLflow store so tests never touch the project runs."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # The artefact root must be redirected too, or runs land in the project's
    # own mlruns/ directory even though the database is temporary.
    mlflow.create_experiment("test", artifact_location=str(tmp_path / "mlruns"))
    mlflow.set_experiment("test")
    yield
    mlflow.set_tracking_uri(config.TRACKING_URI)


@pytest.mark.parametrize("name", sorted(train_model.MODELS))
def test_build_pipeline_fits_every_candidate(frames, name):
    train, _ = frames
    pipeline = train_model.build_pipeline(name)
    pipeline.fit(feature_frame(train), train[config.TARGET])
    probabilities = pipeline.predict_proba(feature_frame(train))
    assert probabilities.shape == (len(train), 2)


def test_build_pipeline_applies_overrides():
    pipeline = train_model.build_pipeline("lightgbm", learning_rate=0.3)
    assert pipeline.named_steps["model"].learning_rate == 0.3


def test_train_one_logs_run(frames, tracking):  # pylint: disable=unused-argument
    train, test = frames
    result = train_model.train_one("logistic_regression", train, test)
    run = mlflow.get_run(result["run_id"])
    assert run.data.params["model"] == "logistic_regression"
    assert "pr_auc" in run.data.metrics
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(result["run_id"], "plots")}
    assert "plots/lift.png" in artifacts
    assert result["pr_auc"] > 0.5


def test_register_best_sets_champion_alias(frames, tracking, monkeypatch):
    # pylint: disable=unused-argument
    monkeypatch.setattr(config, "REGISTERED_MODEL_NAME", "test-model")
    train, test = frames
    results = [train_model.train_one("logistic_regression", train, test)]
    version = train_model.register_best(results)
    champion = mlflow.MlflowClient().get_model_version_by_alias("test-model", "champion")
    assert champion.version == version


def test_plots_are_written(tmp_path):
    y_true = np.array([0, 1] * 20)
    y_score = np.linspace(0, 1, 40)
    assert visualize.plot_confusion(y_true, y_score > 0.5, tmp_path / "c.png").exists()
    assert visualize.plot_curves(y_true, y_score, tmp_path / "r.png").exists()
    assert visualize.plot_lift(y_true, y_score, tmp_path / "l.png").exists()
    names = [f"f{i}" for i in range(5)]
    assert visualize.plot_importance(names, np.arange(5), tmp_path / "i.png").exists()
