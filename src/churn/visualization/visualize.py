"""Evaluation plots logged to MLflow as run artefacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def plot_confusion(y_true, y_pred, path: Path) -> Path:
    """Save a confusion matrix at the chosen operating point."""
    fig, axes = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes, colorbar=False)
    axes.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_curves(y_true, y_score, path: Path) -> Path:
    """Save the ROC and precision-recall curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[0])
    axes[0].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes[1])
    axes[1].set_title("Precision-recall")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lift(y_true, y_score, path: Path, deciles: int = 10) -> Path:
    """Save a decile lift chart.

    The operational question is not 'how accurate is the model' but 'how much
    richer in churners is the list we actually contact', which is what a lift
    chart shows directly.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))
    ranked = y_true[order]
    base = ranked.mean()
    chunks = np.array_split(ranked, deciles)
    lifts = [chunk.mean() / base if base else 0.0 for chunk in chunks]

    fig, axes = plt.subplots(figsize=(6, 4))
    axes.bar(range(1, deciles + 1), lifts, color="#11345f")
    axes.axhline(1.0, color="#c62828", linestyle="--", linewidth=1)
    axes.set_xlabel("Decile of predicted risk (1 = highest)")
    axes.set_ylabel("Lift vs. base rate")
    axes.set_title("Decile lift")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_importance(names, values, path: Path, top: int = 20) -> Path:
    """Save the top feature importances or coefficients."""
    values = np.asarray(values, dtype=float)
    idx = np.argsort(np.abs(values))[-top:]
    fig, axes = plt.subplots(figsize=(6, 0.3 * len(idx) + 1.5))
    axes.barh([names[i] for i in idx], values[idx], color="#11345f")
    axes.set_title(f"Top {len(idx)} features")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
