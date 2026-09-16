"""Metrics used to compare candidate models.

Accuracy is deliberately absent: with a 17% positive rate, a model that never
predicts churn is already 83% accurate.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def lift_at_k(y_true, y_score, k: float = 0.10) -> float:
    """Lift in the top ``k`` fraction of the ranking.

    Args:
        y_true: Binary ground truth.
        y_score: Predicted probability of churn.
        k: Share of the population that would actually be contacted.

    Returns:
        The churn rate inside the top ``k`` divided by the overall churn rate.
        A value of 2.5 means the contacted list is 2.5 times richer in
        churners than the base population.
    """
    y_true = np.asarray(y_true)
    cutoff = max(1, int(len(y_true) * k))
    top = y_true[np.argsort(-np.asarray(y_score))[:cutoff]]
    base = y_true.mean()
    return float(top.mean() / base) if base else 0.0


def recall_at_k(y_true, y_score, k: float = 0.10) -> float:
    """Share of all churners captured in the top ``k`` of the ranking."""
    y_true = np.asarray(y_true)
    cutoff = max(1, int(len(y_true) * k))
    top = y_true[np.argsort(-np.asarray(y_score))[:cutoff]]
    total = y_true.sum()
    return float(top.sum() / total) if total else 0.0


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    """Return the full metric set for one model on one split."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "lift_at_10pct": lift_at_k(y_true, y_score, 0.10),
        "recall_at_10pct": recall_at_k(y_true, y_score, 0.10),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
