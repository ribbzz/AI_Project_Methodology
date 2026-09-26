"""Explainable AI for the churn model, using SHAP (graded project part 3).

Run as a script::

    python -m churn.models.explain                 # every output, default customer
    python -m churn.models.explain --customer 51803

The champion model is a LightGBM pipeline, so Shapley values are computed
exactly with :class:`shap.TreeExplainer` rather than approximated. Every figure
is written to ``reports/figures/shap/`` and the per-customer reason codes to
``reports/predictions_with_reasons.csv``.

Part 1 of this project promised marketing "the top three reasons" beside every
risk score; this module is what produces them.
"""

# pylint: disable=unreachable
# shap's plotting functions are inferred by pylint as never returning, so every
# statement after one of them is falsely reported as unreachable code.

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from churn import config  # noqa: E402
from churn.features.build_features import feature_frame  # noqa: E402
from churn.models.predict_model import load_model, risk_band  # noqa: E402

logger = logging.getLogger(__name__)

#: Where every SHAP artefact is written.
SHAP_DIR = config.FIGURES_DIR / "shap"
REASONS_FILE = config.REPORTS_DIR / "predictions_with_reasons.csv"
FAIRNESS_FILE = config.REPORTS_DIR / "fairness_audit.csv"
#: Attributes audited for disparate treatment. Gender is protected; marital
#: status is sensitive and is a legitimate churn driver, so it is reported
#: rather than judged.
SENSITIVE_ATTRIBUTES = ("Gender", "MaritalStatus")
#: Features whose dependence plots are always produced, if present.
DEPENDENCE_FEATURES = ("Tenure", "CashbackAmount", "Complain", "DaySinceLastOrder")


def _save(path, tight=True):
    """Save the current matplotlib figure and close it."""
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close("all")
    logger.info("wrote %s", path.name)
    return path


def prepare(alias: str = "champion", frame: pd.DataFrame | None = None):
    """Load the champion pipeline and the transformed matrix SHAP needs.

    The registered model is a pipeline of preprocessing + estimator. SHAP's
    TreeExplainer works on the estimator, so the two halves are separated here
    and the feature names of the transformed space are recovered.

    Args:
        alias: Registry alias to explain.
        frame: Customers to explain; the test split by default.

    Returns:
        Tuple of (estimator, transformed DataFrame, display DataFrame in the
        original units, original customer rows).
    """
    pipeline = load_model(alias)
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "booster_") and not hasattr(estimator, "estimators_"):
        raise TypeError(
            f"TreeExplainer needs a tree model; the champion is {type(estimator).__name__}."
        )
    customers = pd.read_parquet(config.TEST_FILE) if frame is None else frame
    preprocessor = pipeline.named_steps["preprocess"]
    matrix = preprocessor.transform(feature_frame(customers))
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    features = pd.DataFrame(matrix, columns=names)
    return (
        estimator,
        features,
        display_frame(preprocessor, features),
        customers.reset_index(drop=True),
    )


def display_frame(preprocessor, features: pd.DataFrame) -> pd.DataFrame:
    """Undo the scaling so the plots show real units, not z-scores.

    Shapley values are computed on the scaled matrix the model actually sees;
    only the values *displayed* beside them are converted back, so a reader
    sees "Tenure = 1" instead of "Tenure = -1.094".
    """
    display = features.copy()
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    # The scaler sits behind an imputer, so it carries no column names; the
    # numeric block keeps the order declared in config.
    numeric = [
        c for c in config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES if c in features.columns
    ]
    display[numeric] = scaler.inverse_transform(features[numeric]).round(2)
    return display


def shapley_values(estimator, features: pd.DataFrame):
    """Build a TreeExplainer and compute Shapley values for the churn class.

    LightGBM binary classifiers return either one matrix (the positive class)
    or one per class depending on the version, so both shapes are normalised
    here to a single :class:`shap.Explanation` for "customer churns".
    """
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(features)
    if explanation.values.ndim == 3:  # (rows, features, classes)
        explanation = explanation[:, :, 1]
    logger.info(
        "Shapley values: %s rows x %s features",
        explanation.values.shape[0],
        explanation.values.shape[1],
    )
    return explainer, explanation


def plot_single_customer(explanation, index: int, customer_id) -> None:
    """Waterfall and force plots for one customer (a specific data point)."""
    shap.plots.waterfall(explanation[index], max_display=14, show=False)
    plt.title(f"Waterfall - customer {customer_id}", fontsize=10)
    _save(SHAP_DIR / f"waterfall_customer_{customer_id}.png")

    shap.plots.force(explanation[index], matplotlib=True, show=False)
    _save(SHAP_DIR / f"force_customer_{customer_id}.png", tight=False)


def plot_all_points(explainer, explanation, features: pd.DataFrame, limit: int = 400) -> None:
    """Interactive force plot stacking every customer in one figure."""
    subset = slice(0, min(limit, len(features)))
    force = shap.force_plot(
        (
            explainer.expected_value
            if np.ndim(explainer.expected_value) == 0
            else explainer.expected_value[-1]
        ),
        explanation.values[subset],
        features.iloc[subset],
    )
    path = SHAP_DIR / "force_all_customers.html"
    shap.save_html(str(path), force)
    logger.info("wrote %s (%d customers)", path.name, min(limit, len(features)))


def plot_global(explanation, display: pd.DataFrame) -> None:
    """Summary plot per class, mean SHAP bar, beeswarm and dependence plots."""
    # A binary model has one set of Shapley values: positive pushes towards
    # churn, negative towards staying. The "no churn" class is its mirror.
    for label, values in (("churn", explanation.values), ("no_churn", -explanation.values)):
        shap.summary_plot(values, display, show=False, max_display=15, plot_size=(9, 6))
        plt.title(f"SHAP summary - class '{label}'", fontsize=11)
        _save(SHAP_DIR / f"summary_class_{label}.png")

    shap.plots.bar(explanation, max_display=15, show=False)
    plt.title("Mean |SHAP| per feature", fontsize=11)
    _save(SHAP_DIR / "mean_shap_bar.png")

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("Beeswarm - distribution of Shapley values", fontsize=11)
    _save(SHAP_DIR / "beeswarm.png")

    for feature in DEPENDENCE_FEATURES:
        if feature not in display.columns:
            continue
        shap.dependence_plot(feature, explanation.values, display, show=False)
        plt.title(f"Dependence - {feature}", fontsize=11)
        _save(SHAP_DIR / f"dependence_{feature}.png")


def reason_codes(
    estimator, explanation, features: pd.DataFrame, customers: pd.DataFrame, top: int = 3
) -> pd.DataFrame:
    """Strongest drivers of each customer's score, as text a marketer can read.

    Drivers are ranked by the **size** of their contribution, not by its sign,
    and carry the direction with them. Ranking by signed value alone would
    hand a safe customer three "risk factors" that are merely the least
    protective of their features.

    Returns:
        One row per customer with the churn probability, the risk band and
        ``top`` reasons such as ``Tenure -3.21 (lowers risk)``.
    """
    values = explanation.values
    order = np.argsort(-np.abs(values), axis=1)[:, :top]
    names = np.array(features.columns)
    probabilities = estimator.predict_proba(features)[:, 1]
    rows = {
        config.ID_COLUMN: customers[config.ID_COLUMN].to_numpy(),
        "churn_probability": probabilities.round(4),
        "risk_band": [risk_band(p) for p in probabilities],
    }
    for rank in range(top):
        idx = order[:, rank]
        rows[f"reason_{rank + 1}"] = [
            f"{names[j]} {values[i, j]:+.2f} "
            f"({'raises' if values[i, j] > 0 else 'lowers'} risk)"
            for i, j in enumerate(idx)
        ]
    frame = pd.DataFrame(rows)
    if config.TARGET in customers:
        frame["actual_churn"] = customers[config.TARGET].to_numpy()
    frame = frame.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    frame.to_csv(REASONS_FILE, index=False)
    logger.info("wrote %s", REASONS_FILE.name)
    return frame


def fairness_audit(
    explanation, features: pd.DataFrame, customers: pd.DataFrame, probabilities=None
) -> pd.DataFrame:
    """Audit the model's behaviour across sensitive groups.

    Two questions are asked, one explanation-based and one outcome-based:
    how much does the model *lean on* a sensitive attribute (mean absolute
    Shapley value of its encoded columns), and does the resulting HIGH-risk
    flag fall unevenly on groups (selection rate, and its ratio to the best
    served group - the disparate impact measure used in the course lab).

    Returns:
        One row per group with its size, selection rate, disparate impact,
        statistical parity difference and actual churn rate.
    """
    if probabilities is None:
        probabilities = np.zeros(len(customers))
    flagged = np.array([risk_band(p) == "HIGH" for p in probabilities])
    rows = []
    for attribute in SENSITIVE_ATTRIBUTES:
        if attribute not in customers:
            continue
        encoded = [c for c in features.columns if c.startswith(f"{attribute}_")]
        leaning = (
            float(
                np.abs(explanation.values[:, [features.columns.get_loc(c) for c in encoded]]).mean()
            )
            if encoded
            else 0.0
        )
        groups = customers[attribute]
        rates = {g: flagged[groups == g].mean() for g in sorted(groups.dropna().unique())}
        best = max(rates.values()) if rates else 0.0
        for group, rate in rates.items():
            mask = groups == group
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "customers": int(mask.sum()),
                    "mean_abs_shap_of_attribute": round(leaning, 3),
                    "selection_rate": round(float(rate), 4),
                    "disparate_impact": round(float(rate / best), 3) if best else None,
                    "parity_difference": round(float(rate - flagged.mean()), 4),
                    "actual_churn_rate": (
                        round(float(customers.loc[mask, config.TARGET].mean()), 4)
                        if config.TARGET in customers
                        else None
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(FAIRNESS_FILE, index=False)
    logger.info("wrote %s", FAIRNESS_FILE.name)
    return frame


def main(customer: int | None = None, alias: str = "champion", limit: int = 400) -> None:
    """Produce every SHAP artefact required by part 3 of the brief."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    estimator, features, display, customers = prepare(alias)
    explainer, explanation = shapley_values(estimator, features)
    # Keep the Shapley values, show the readable units beside them.
    explanation.data = display.to_numpy()

    if customer is None:
        # Default to the highest-risk customer: the most interesting to explain.
        index = int(np.argmax(explanation.values.sum(axis=1)))
    else:
        matches = customers.index[customers[config.ID_COLUMN] == customer]
        if len(matches) == 0:
            raise SystemExit(f"customer {customer} is not in {config.TEST_FILE.name}")
        index = int(matches[0])
    customer_id = customers.loc[index, config.ID_COLUMN]
    logger.info("explaining customer %s (row %d)", customer_id, index)

    plot_single_customer(explanation, index, customer_id)
    plot_all_points(explainer, explanation, display, limit)
    plot_global(explanation, display)
    reasons = reason_codes(estimator, explanation, features, customers)
    print(reasons.head(8).to_string(index=False))
    audit = fairness_audit(
        explanation, features, customers, estimator.predict_proba(features)[:, 1]
    )
    print("\nFairness audit\n" + audit.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain the champion model with SHAP.")
    parser.add_argument(
        "--customer",
        type=int,
        default=None,
        help="CustomerID to explain (default: the highest-risk customer).",
    )
    parser.add_argument("--alias", default="champion", help="Registry alias to explain.")
    parser.add_argument(
        "--limit", type=int, default=400, help="Customers included in the all-points force plot."
    )
    main(**vars(parser.parse_args()))
