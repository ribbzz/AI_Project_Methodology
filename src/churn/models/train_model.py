"""Model training with MLflow tracking and model registry.

Train one model::

    python -m churn.models.train_model --model lightgbm

Train every candidate and register the best of them::

    python -m churn.models.train_model --all --register

Run a small LightGBM hyper-parameter sweep as nested runs::

    python -m churn.models.train_model --sweep

Each run holds the parameters, the metrics, the evaluation plots and the
serialised model with its input signature.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn import config
from churn.features.build_features import build_preprocessor, feature_frame
from churn.models.evaluate import compute_metrics
from churn.visualization import visualize

logger = logging.getLogger(__name__)

#: Candidate estimators with their default parameters. ``class_weight``
#: handles the imbalance without resampling, which keeps probabilities meaningful.
MODELS = {
    "logistic_regression": (
        LogisticRegression,
        {"max_iter": 2000, "class_weight": "balanced", "random_state": config.RANDOM_STATE},
    ),
    "random_forest": (
        RandomForestClassifier,
        {
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": config.RANDOM_STATE,
        },
    ),
    "lightgbm": (
        LGBMClassifier,
        {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "class_weight": "balanced",
            "verbose": -1,
            "random_state": config.RANDOM_STATE,
        },
    ),
}

#: Grid explored by ``--sweep``. Kept deliberately small: the brief is about
#: tracking practice, not about squeezing out the last point of PR-AUC.
SWEEP_GRID = {"learning_rate": [0.02, 0.05, 0.1], "num_leaves": [15, 31]}


def build_pipeline(name: str, **overrides) -> Pipeline:
    """Preprocessing and estimator assembled as one pipeline.

    Args:
        name: Key of :data:`MODELS`.
        **overrides: Estimator parameters replacing the defaults.
    """
    estimator_class, defaults = MODELS[name]
    estimator = estimator_class(**{**defaults, **overrides})
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def _log_plots(pipeline: Pipeline, y_true, y_score, threshold: float) -> None:
    """Render the evaluation figures and attach them to the active run."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        y_pred = (y_score >= threshold).astype(int)
        figures = [
            visualize.plot_confusion(y_true, y_pred, tmp / "confusion.png"),
            visualize.plot_curves(y_true, y_score, tmp / "curves.png"),
            visualize.plot_lift(y_true, y_score, tmp / "lift.png"),
        ]
        estimator = pipeline.named_steps["model"]
        names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        values = getattr(estimator, "feature_importances_", None)
        if values is None:
            values = getattr(estimator, "coef_", [[]])[0]
        if len(values):
            figures.append(visualize.plot_importance(names, values, tmp / "importance.png"))
        for figure in figures:
            mlflow.log_artifact(str(figure), "plots")


def train_one(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = 0.5,
    *,
    run_name: str | None = None,
    nested: bool = False,
    **overrides,
) -> dict:
    """Train one candidate, log everything to MLflow and return its metrics."""
    x_train, y_train = feature_frame(train), train[config.TARGET]
    x_test, y_test = feature_frame(test), test[config.TARGET]

    with mlflow.start_run(run_name=run_name or name, nested=nested) as run:
        pipeline = build_pipeline(name, **overrides)
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)
        y_score = probabilities[:, 1]
        metrics = compute_metrics(y_test, y_score, threshold)

        mlflow.log_params(
            {"model": name, "threshold": threshold, "n_train": len(train), "n_test": len(test)}
        )
        mlflow.log_params(
            {
                f"model__{key}": value
                for key, value in pipeline.named_steps["model"].get_params().items()
                if isinstance(value, (int, float, str, bool)) or value is None
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"dataset": config.RAW_FILE.name, "model_family": name})

        # cloudpickle rather than the skops default: the pipeline contains a
        # LightGBM booster and numpy dtypes that skops refuses to serialise.
        # pyfunc_predict_fn makes the served REST endpoint return the churn
        # probability rather than a hard 0/1 label - a label cannot be ranked.
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(x_test, probabilities),
            input_example=x_test.head(3),
            serialization_format="cloudpickle",
            pyfunc_predict_fn="predict_proba",
        )
        _log_plots(pipeline, y_test, y_score, threshold)

        logger.info(
            "%-28s pr_auc=%.4f roc_auc=%.4f lift@10%%=%.2f",
            run_name or name,
            metrics["pr_auc"],
            metrics["roc_auc"],
            metrics["lift_at_10pct"],
        )
        return {"name": run_name or name, "run_id": run.info.run_id, **metrics}


def sweep(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """Run the LightGBM grid as nested runs under one parent run."""
    results = []
    keys = list(SWEEP_GRID)
    with mlflow.start_run(run_name="lightgbm_sweep"):
        mlflow.set_tag("sweep_grid", str(SWEEP_GRID))
        for values in itertools.product(*SWEEP_GRID.values()):
            params = dict(zip(keys, values))
            label = "lightgbm_" + "_".join(f"{k}={v}" for k, v in params.items())
            results.append(
                train_one("lightgbm", train, test, threshold, run_name=label, nested=True, **params)
            )
        best = max(results, key=lambda r: r["pr_auc"])
        mlflow.log_metric("best_pr_auc", best["pr_auc"])
        mlflow.set_tag("best_child", best["name"])
    return results


def register_best(results: list[dict], metric: str = "pr_auc") -> str:
    """Register the winning run and give it the ``champion`` alias.

    Returns:
        The version number assigned in the registry.
    """
    best = max(results, key=lambda r: r[metric])
    version = mlflow.register_model(
        f"runs:/{best['run_id']}/model", config.REGISTERED_MODEL_NAME
    ).version
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", version)
    client.set_model_version_tag(config.REGISTERED_MODEL_NAME, version, "selected_on", metric)
    client.set_model_version_tag(config.REGISTERED_MODEL_NAME, version, "run_name", best["name"])
    client.update_model_version(
        config.REGISTERED_MODEL_NAME,
        version,
        description=(
            f"{best['name']} - {metric}={best[metric]:.4f}, "
            f"lift@10%={best['lift_at_10pct']:.2f}. Selected automatically by train_model.py."
        ),
    )
    logger.info(
        "registered %s v%s (%s) with alias 'champion'",
        config.REGISTERED_MODEL_NAME,
        version,
        best["name"],
    )
    return version


def setup_tracking() -> None:
    """Point MLflow at the project store and select the experiment."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.EXPERIMENT_NAME, artifact_location=str(config.MLRUNS_DIR))
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def main(
    model: str = "lightgbm",
    run_all: bool = False,
    run_sweep: bool = False,
    register: bool = False,
    threshold: float = 0.5,
) -> None:
    """Entry point used by the CLI and by the MLflow project."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_tracking()
    train = pd.read_parquet(config.TRAIN_FILE)
    test = pd.read_parquet(config.TEST_FILE)

    results = []
    if run_sweep:
        results += sweep(train, test, threshold)
    if run_all:
        results += [train_one(name, train, test, threshold) for name in MODELS]
    if not results:
        results.append(train_one(model, train, test, threshold))

    table = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    print(table.drop(columns="run_id").to_string(index=False))

    if register:
        register_best(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn models with MLflow tracking.")
    parser.add_argument(
        "--model", default="lightgbm", choices=sorted(MODELS), help="Single model to train."
    )
    parser.add_argument(
        "--all", dest="run_all", action="store_true", help="Train every candidate model."
    )
    parser.add_argument(
        "--sweep", dest="run_sweep", action="store_true", help="Run the LightGBM grid."
    )
    parser.add_argument(
        "--register", action="store_true", help="Register the best run in the model registry."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold for point metrics."
    )
    main(**vars(parser.parse_args()))
