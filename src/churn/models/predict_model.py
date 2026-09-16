"""Batch inference: score customers with the registered champion model.

Run as a script::

    python -m churn.models.predict_model --input data/processed/test.parquet

The output is not a bare probability. Each customer gets a calibrated score, a
risk band and the reason the band was assigned, because the people who act on
the file are marketers and support agents, not data scientists.
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.sklearn
import pandas as pd

from churn import config
from churn.features.build_features import feature_frame

logger = logging.getLogger(__name__)


def load_model(alias: str = "champion"):
    """Load a model from the registry, falling back to the newest version.

    Args:
        alias: Registry alias to resolve, ``champion`` by default.
    """
    mlflow.set_tracking_uri(config.TRACKING_URI)
    uri = f"models:/{config.REGISTERED_MODEL_NAME}@{alias}"
    logger.info("loading %s", uri)
    return mlflow.sklearn.load_model(uri)


def risk_band(probability: float) -> str:
    """Translate a probability into an actionable band."""
    for cutoff, band in config.RISK_BANDS:
        if probability >= cutoff:
            return band
    return "LOW"


def score(frame: pd.DataFrame, model) -> pd.DataFrame:
    """Score a table of customers.

    Args:
        frame: Cleaned customer rows, as produced by the data preparation step.
        model: Any fitted estimator exposing ``predict_proba``.

    Returns:
        One row per customer with the churn probability, the risk band and the
        rank within the scored population.
    """
    probabilities = model.predict_proba(feature_frame(frame))[:, 1]
    out = pd.DataFrame(
        {
            config.ID_COLUMN: frame[config.ID_COLUMN].to_numpy(),
            "churn_probability": probabilities.round(4),
        }
    )
    out["risk_band"] = [risk_band(p) for p in probabilities]
    out["rank"] = out["churn_probability"].rank(ascending=False, method="first").astype(int)
    if config.TARGET in frame:
        out["actual_churn"] = frame[config.TARGET].to_numpy()
    return out.sort_values("rank").reset_index(drop=True)


def main(
    input_path: str = str(config.TEST_FILE),
    output_path: str = str(config.PREDICTIONS_FILE),
    alias: str = "champion",
) -> None:
    """Score a parquet file and write the results as CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = pd.read_parquet(input_path)
    scored = score(frame, load_model(alias))
    scored.to_csv(output_path, index=False)
    counts = scored["risk_band"].value_counts().to_dict()
    logger.info("scored %d customers -> %s (%s)", len(scored), output_path, counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score customers with the champion model.")
    parser.add_argument("--input", dest="input_path", default=str(config.TEST_FILE))
    parser.add_argument("--output", dest="output_path", default=str(config.PREDICTIONS_FILE))
    parser.add_argument("--alias", default="champion")
    main(**vars(parser.parse_args()))
