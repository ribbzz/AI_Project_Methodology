"""Send a sample scoring request to a running MLflow inference server.

Start the server first::

    mlflow models serve -m "models:/ecommerce-churn-classifier@champion" -p 5001 --no-conda

then::

    python scripts/score_request.py --port 5001
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

from churn import config
from churn.features.build_features import feature_frame


def sample_customers(rows: int = 6) -> pd.DataFrame:
    """Pick half churners and half non-churners from the test split."""
    test = pd.read_parquet(config.TEST_FILE)
    half = max(1, rows // 2)
    churners = test[test[config.TARGET] == 1].head(half)
    stayers = test[test[config.TARGET] == 0].head(rows - half)
    return pd.concat([churners, stayers], ignore_index=True)


def build_payload(customers: pd.DataFrame) -> dict:
    """Serialise customers in the MLflow ``dataframe_split`` format."""
    frame = feature_frame(customers)
    return {
        "dataframe_split": {
            "columns": list(frame.columns),
            "data": frame.astype(object).where(frame.notna(), None).values.tolist(),
        }
    }


def main(port: int = 5001, rows: int = 6) -> None:
    """POST sample customers to ``/invocations`` and print their churn scores."""
    customers = sample_customers(rows)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/invocations",
        data=json.dumps(build_payload(customers)).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        predictions = json.loads(response.read())["predictions"]

    # The model is served with predict_proba: each row is [p(stay), p(churn)].
    result = pd.DataFrame(
        {
            config.ID_COLUMN: customers[config.ID_COLUMN],
            "actual_churn": customers[config.TARGET],
            "churn_probability": [round(row[1], 4) for row in predictions],
        }
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call the local inference server.")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--rows", type=int, default=6)
    main(**vars(parser.parse_args()))
