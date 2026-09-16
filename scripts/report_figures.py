"""Build the report figures straight from the MLflow tracking store.

Usage::

    python scripts/report_figures.py

Writes ``reports/figures/model_comparison.png`` and ``sweep_heatmap.png`` and
copies the champion run's evaluation plots next to them, so the report can be
regenerated without opening the MLflow UI.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402

from churn import config  # noqa: E402

BLUE = "#2a78d6"
BLUE_RAMP = ["#e6effa", "#b9d2f2", "#7fb0e8", "#2a78d6", "#1d58a3", "#123a6e"]
GREY = "#b9b7b0"
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"


def _style(axes) -> None:
    """Recessive axes: no top/right spines, light grid, muted ticks."""
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#d6d5d0")
    axes.tick_params(colors=MUTED, labelsize=8)
    axes.grid(axis="x", color="#ecebe7", linewidth=0.8)
    axes.set_axisbelow(True)


def model_comparison(runs, champion_run: str, path: Path) -> None:
    """Horizontal bars of PR-AUC for every run; the champion is emphasised."""
    runs = runs.dropna(subset=["metrics.pr_auc"]).sort_values("metrics.pr_auc")
    names = runs["tags.mlflow.runName"].str.replace("lightgbm_learning_rate=", "lgbm lr=")
    names = names.str.replace("_num_leaves=", ", leaves=")
    values = runs["metrics.pr_auc"].to_numpy()
    colours = [BLUE if rid == champion_run else GREY for rid in runs["run_id"]]

    fig, axes = plt.subplots(figsize=(7.2, 3.9), facecolor=SURFACE)
    axes.barh(names, values, color=colours, height=0.62, edgecolor=SURFACE, linewidth=2)
    for y, value in enumerate(values):
        axes.text(value + 0.004, y, f"{value:.3f}", va="center", fontsize=7.5, color=INK)
    axes.set_xlim(0.6, 1.04)
    axes.set_xlabel("PR-AUC on the test split", color=MUTED, fontsize=8.5)
    axes.set_title(
        "Model comparison - all tracked runs (champion in blue)",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _style(axes)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def sweep_heatmap(runs, path: Path) -> None:
    """PR-AUC of the LightGBM sweep as a learning-rate x leaves grid."""
    sweep = runs[runs["tags.mlflow.runName"].str.startswith("lightgbm_learning_rate")]
    rates = sorted(sweep["params.model__learning_rate"].astype(float).unique())
    leaves = sorted(sweep["params.model__num_leaves"].astype(int).unique())
    grid = np.full((len(leaves), len(rates)), np.nan)
    for _, row in sweep.iterrows():
        i = leaves.index(int(row["params.model__num_leaves"]))
        j = rates.index(float(row["params.model__learning_rate"]))
        grid[i, j] = row["metrics.pr_auc"]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    fig, axes = plt.subplots(figsize=(4.6, 2.6), facecolor=SURFACE)
    image = axes.imshow(grid, cmap=cmap, vmin=np.nanmin(grid) - 0.01, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            dark = image.norm(grid[i, j]) > 0.55
            axes.text(
                j,
                i,
                f"{grid[i, j]:.4f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if dark else INK,
            )
    axes.set_xticks(range(len(rates)), [str(r) for r in rates])
    axes.set_yticks(range(len(leaves)), [str(n) for n in leaves])
    axes.set_xlabel("learning_rate", color=MUTED, fontsize=8.5)
    axes.set_ylabel("num_leaves", color=MUTED, fontsize=8.5)
    axes.set_title("LightGBM sweep - PR-AUC (nested runs)", loc="left", fontsize=10, color=INK)
    axes.tick_params(colors=MUTED, labelsize=8, length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=axes, fraction=0.05, pad=0.03).ax.tick_params(labelsize=7, colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    """Query the store and write every figure."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    runs = mlflow.search_runs([experiment.experiment_id])
    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, "champion")

    model_comparison(runs, champion.run_id, config.FIGURES_DIR / "model_comparison.png")
    sweep_heatmap(runs, config.FIGURES_DIR / "sweep_heatmap.png")

    plots = client.download_artifacts(champion.run_id, "plots")
    for png in Path(plots).glob("*.png"):
        shutil.copy(png, config.FIGURES_DIR / f"champion_{png.name}")
    print("figures written to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
