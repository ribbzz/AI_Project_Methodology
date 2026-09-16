"""Feature engineering: derived columns plus the fitted preprocessing pipeline.

Run as a script::

    python -m churn.features.build_features

The same code path is used at training time and at inference time, which is
what keeps the two from drifting apart.
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config

logger = logging.getLogger(__name__)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the derived behavioural ratios.

    Absolute counts say less about churn than the same counts put in context:
    ten orders is a lot for a new customer and very little for an old one.

    Args:
        frame: A cleaned table produced by :mod:`churn.data.make_dataset`.

    Returns:
        A copy of ``frame`` with the columns listed in
        :data:`churn.config.ENGINEERED_FEATURES` appended.
    """
    frame = frame.copy()
    tenure = frame["Tenure"].fillna(0)
    orders = frame["OrderCount"].fillna(0)

    frame["OrdersPerTenure"] = orders / (tenure + 1.0)
    frame["CouponRate"] = frame["CouponUsed"].fillna(0) / (orders + 1.0)
    frame["CashbackPerOrder"] = frame["CashbackAmount"].fillna(0) / (orders + 1.0)
    frame["IsNewCustomer"] = (tenure <= 1).astype(int)
    # How long the customer has been silent relative to their own history.
    frame["RecencyRatio"] = frame["DaySinceLastOrder"].fillna(0) / (tenure + 1.0)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def build_preprocessor() -> ColumnTransformer:
    """Assemble the column-wise preprocessing pipeline.

    Numeric columns are median-imputed and standardised; categorical columns
    are mode-imputed and one-hot encoded with unknown categories ignored, so
    that an unseen category at scoring time cannot crash the pipeline.
    """
    numeric = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the model input columns, with engineered features added."""
    frame = add_engineered_features(frame)
    numeric = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
    # Floats everywhere: the MLflow signature must accept a missing value in
    # any numeric column at serving time, which an integer column cannot hold.
    frame[numeric] = frame[numeric].astype("float64")
    return frame[numeric + config.CATEGORICAL_FEATURES]


def main() -> None:
    """Fit the preprocessor on the training split and persist it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train = pd.read_parquet(config.TRAIN_FILE)
    preprocessor = build_preprocessor()
    preprocessor.fit(feature_frame(train))
    joblib.dump(preprocessor, config.PREPROCESSOR_FILE)
    names = preprocessor.get_feature_names_out()
    logger.info("fitted preprocessor on %d rows -> %d features", len(train), len(names))
    logger.info("saved %s", config.PREPROCESSOR_FILE)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Fit and save the preprocessing pipeline.").parse_args()
    main()
