"""Data preparation: read the raw Excel export, clean it and split it.

Run as a script::

    python -m churn.data.make_dataset

The step is deliberately separate from feature engineering: it only produces a
tidy table that still looks like the source data, so that the cleaning rules
can be reviewed by someone who knows the business rather than the model.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

logger = logging.getLogger(__name__)


def load_raw(path=config.RAW_FILE, sheet: str = config.RAW_SHEET) -> pd.DataFrame:
    """Read the raw dataset.

    Args:
        path: Location of the Excel workbook shipped in ``data/raw``.
        sheet: Worksheet holding the observations.

    Returns:
        The raw table, untouched apart from being loaded.
    """
    logger.info("reading %s (sheet %s)", path, sheet)
    return pd.read_excel(path, sheet_name=sheet)


def harmonise_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate spellings of the same category.

    The source file uses both ``Mobile Phone`` and ``Phone`` for the login
    device, and both ``COD`` and ``Cash on Delivery`` for the payment mode.
    Left alone they would become separate one-hot columns and split the signal.
    """
    frame = frame.copy()
    for column, mapping in config.CATEGORY_FIXES.items():
        if column in frame:
            frame[column] = frame[column].replace(mapping)
    return frame


def drop_duplicate_customers(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per customer id."""
    before = len(frame)
    frame = frame.drop_duplicates(subset=config.ID_COLUMN, keep="first")
    if before != len(frame):
        logger.warning("dropped %d duplicate customer rows", before - len(frame))
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning rule and return the tidy table."""
    frame = harmonise_categories(frame)
    frame = drop_duplicate_customers(frame)
    frame = frame[frame[config.TARGET].notna()]
    numeric = [c for c in config.NUMERIC_FEATURES if c in frame]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame[config.TARGET] = frame[config.TARGET].astype(int)
    logger.info("clean table: %d rows, churn rate %.4f", len(frame), frame[config.TARGET].mean())
    return frame.reset_index(drop=True)


def split(frame: pd.DataFrame, test_size: float = config.TEST_SIZE):
    """Stratified train/test split.

    Stratification matters here because the positive class is only about 17%
    of the data; a plain random split can shift the base rate between the two
    sets and make the metrics hard to compare.
    """
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=config.RANDOM_STATE,
        stratify=frame[config.TARGET],
    )
    logger.info(
        "train %d rows (churn %.4f) / test %d rows (churn %.4f)",
        len(train),
        train[config.TARGET].mean(),
        len(test),
        test[config.TARGET].mean(),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main(test_size: float = config.TEST_SIZE) -> None:
    """Run the full data preparation step and write the outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = clean(load_raw())
    frame.to_parquet(config.CLEAN_FILE, index=False)
    train, test = split(frame, test_size)
    train.to_parquet(config.TRAIN_FILE, index=False)
    test.to_parquet(config.TEST_FILE, index=False)
    logger.info("wrote %s, %s, %s", config.CLEAN_FILE, config.TRAIN_FILE, config.TEST_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and split the raw churn dataset.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=config.TEST_SIZE,
        help="Share of the data held out for testing.",
    )
    main(**vars(parser.parse_args()))
