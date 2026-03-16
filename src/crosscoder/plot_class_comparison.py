"""
Plot class distribution comparison between Wanda and AWQ crosscoder results.
For each model (e.g., blip2, qwen3vl), loads both wanda and awq feature_classification.csv
and produces a double-bar chart: one bar per class for AWQ, one for Wanda.
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .utils import get_features_dir, get_results_dir
from .visualize import (
    COLORS,
    ICLR_DPI,
    classify_for_plot,
    compute_adaptive_rho_thresholds,
)

plt.style.use("seaborn-v0_8-whitegrid")

# Larger font sizes for comparison plot
PLOT_TICK_SIZE = 14
PLOT_LABEL_SIZE = 15
PLOT_BAR_LABEL_SIZE = 12
PLOT_LEGEND_SIZE = 14

# Class order and short display names
CLASS_DISTRIBUTION_ORDER = [
    "uncompressed_only",
    "shared_aligned",
    "shared_redirected",
    "shared_attenuated",
    "shared_intermediate",
    "compressed_only",
]

CLASS_SHORT_NAMES = {
    "uncompressed_only": "Ucompressed",
    "shared_aligned": "S-Aligned",
    "shared_redirected": "S-Redirected",
    "shared_attenuated": "S-Attenuated",
    "shared_intermediate": "S-Intermediate",
    "compressed_only": "Compressed",
}


def load_class_counts(
    results_dir: Path,
) -> dict[str, int]:
    """
    Load feature_classification.csv and return counts per class using GMM-based
    rho thresholds (feature_classification.csv and all evaluation use GMM).
    """
    features_dir = get_features_dir(results_dir)
    csv_path = features_dir / "feature_classification.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    thresh = compute_adaptive_rho_thresholds(df)
    gmm_class = df.apply(
        lambda row: classify_for_plot(row["rho"], row["theta"], thresh), axis=1
    )
    raw_counts = gmm_class.value_counts()
    return {c: raw_counts.get(c, 0) for c in CLASS_DISTRIBUTION_ORDER}


def plot_class_distribution_comparison(
    wanda_counts: dict[str, int],
    awq_counts: dict[str, int],
    model: str,
    component: str,
    output_path: Path,
) -> None:
    """
    Plot a double-bar chart: for each class, two bars side by side (Wanda, AWQ).
    - Same color scheme as current class_distribution
    - AWQ bars: double diagonal hatching (//)
    - Wanda bars: solid fill
    - Thin bars, wide figure, no x-axis rotation
    """
    n_classes = len(CLASS_DISTRIBUTION_ORDER)
    short_labels = [CLASS_SHORT_NAMES[c] for c in CLASS_DISTRIBUTION_ORDER]
    colors = [COLORS[c] for c in CLASS_DISTRIBUTION_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = range(n_classes)
    bar_width = 0.22
    offset = bar_width / 2

    wanda_vals = [wanda_counts.get(c, 0) for c in CLASS_DISTRIBUTION_ORDER]
    awq_vals = [awq_counts.get(c, 0) for c in CLASS_DISTRIBUTION_ORDER]

    # Wanda bars: left of each pair, solid
    bars_wanda = ax.bar(
        [i - offset for i in x],
        wanda_vals,
        width=bar_width,
        color=colors,
        edgecolor="black",
        alpha=0.85,
        linewidth=0.8,
    )

    # AWQ bars: right of each pair, same colors + hatching
    bars_awq = ax.bar(
        [i + offset for i in x],
        awq_vals,
        width=bar_width,
        color=colors,
        edgecolor="black",
        alpha=0.85,
        linewidth=0.8,
        hatch="//",
    )

    # Legend with grey patches to avoid reusing bar colors (e.g. red)
    legend_grey = "#7F8C8D"
    legend_handles = [
        mpatches.Patch(facecolor=legend_grey, edgecolor="black", alpha=0.85, label="Wanda"),
        mpatches.Patch(facecolor=legend_grey, edgecolor="black", alpha=0.85, hatch="//", label="AWQ"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=PLOT_LEGEND_SIZE)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=0, ha="center", fontsize=PLOT_TICK_SIZE)
    ax.set_xlabel("Feature class", fontsize=PLOT_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Count", fontsize=PLOT_LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE, size=5, width=1.25)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels above bars
    for bar, val in zip(bars_wanda, wanda_vals):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(val)),
                ha="center",
                va="bottom",
                fontsize=PLOT_BAR_LABEL_SIZE,
            )
    for bar, val in zip(bars_awq, awq_vals):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(val)),
                ha="center",
                va="bottom",
                fontsize=PLOT_BAR_LABEL_SIZE,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def run_comparison(model: str, component: str, token_type: str = "cls") -> None:
    """Load wanda and awq results for the given model/component and plot comparison."""
    results_base = config.CROSSCODER_RESULTS_DIR

    wanda_dir = results_base / f"{model}__wanda__{component}__{token_type}"
    awq_dir = results_base / f"{model}__awq__{component}__{token_type}"

    wanda_counts = load_class_counts(wanda_dir)
    awq_counts = load_class_counts(awq_dir)

    if not wanda_counts and not awq_counts:
        print(f"No data found for {model} / {component}. Skipping.")
        return

    out_dir = results_base / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"class_distribution_{model}_{component}_{token_type}.png"

    plot_class_distribution_comparison(
        wanda_counts=wanda_counts,
        awq_counts=awq_counts,
        model=model,
        component=component,
        output_path=output_path,
    )
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot class distribution comparison (Wanda vs AWQ) for crosscoder results."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "blip2", "qwen3vl"],
        help="Model to plot (default: all)",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="V_P",
        choices=["V", "P", "V_P"],
        help="Component (default: V_P)",
    )
    parser.add_argument(
        "--token-type",
        type=str,
        default="cls",
        choices=["cls", "patch"],
        help="Token type (default: cls)",
    )
    args = parser.parse_args()

    models = config.MODELS if args.model == "all" else [args.model]

    for model in models:
        run_comparison(model=model, component=args.component, token_type=args.token_type)


if __name__ == "__main__":
    main()
