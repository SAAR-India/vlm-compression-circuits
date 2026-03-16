"""
NOTICE-style heatmap and bar plots for activation patching results.
"""

from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOKEN_LABELS = ["[CLS]", "correct_object", "or", "incorrect_object", "?"]
YAXIS_LABEL_MAP = {"correct_object": "correct_obj", "incorrect_object": "incorrect_obj"}


def plot_notice_heatmaps(
    mlp_data: np.ndarray,
    attn_data: np.ndarray,
    xattn_data: Optional[np.ndarray],
    token_labels: List[str],
    model_label: str,
    save_path,
):
    """NOTICE-style per-layer, per-token heatmaps for MLP, Self-Attention, Cross-Attention."""
    has_xattn = xattn_data is not None and np.any(xattn_data != 0)
    n_panels = 3 if has_xattn else 2
    num_tokens, num_layers = mlp_data.shape
    boundaries, boundary_ends = {}, {}
    prev = None
    for i, lab in enumerate(token_labels):
        if lab != prev:
            boundaries[lab] = i
            prev = lab
    for lab in TOKEN_LABELS:
        idx = [i for i, l in enumerate(token_labels) if l == lab]
        if idx:
            boundary_ends[lab] = idx[-1]
    all_data = [mlp_data, attn_data] + ([xattn_data] if has_xattn else [])
    all_vals = np.concatenate([d.flatten() for d in all_data])
    vmin, vmax = all_vals.min(), all_vals.max()
    fig_width = 7 * n_panels + 1.5
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 6), sharey=True)
    if n_panels == 2:
        axes = list(axes)
    panels = [("MLP", mlp_data), ("Self-Attention", attn_data)]
    if has_xattn:
        panels.append(("Cross-Attention", xattn_data))
    for idx, (title, data) in enumerate(panels):
        ax = axes[idx]
        im = ax.imshow(
            data, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax,
            interpolation="nearest", origin="upper"
        )
        ax.set_xlabel("Layers", fontsize=14, fontweight="bold")
        layer_ticks = list(range(0, num_layers, max(1, num_layers // 7)))
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels([str(t) for t in layer_ticks], fontsize=12)
        ytick_positions, ytick_labels_list = [], []
        for lab in TOKEN_LABELS:
            indices = [i for i, l in enumerate(token_labels) if l == lab]
            if indices:
                ytick_positions.append((indices[0] + indices[-1]) / 2)
                display_lab = YAXIS_LABEL_MAP.get(lab, lab)
                ytick_labels_list.append(display_lab)
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels_list, fontsize=12, rotation=90)
        if idx == 0:
            ax.set_ylabel("Token Position", fontsize=14, fontweight="bold")
        ax.set_title(f"{model_label} - {title} Patching", fontsize=15, fontweight="bold")
        if "correct_object" in boundaries and "correct_object" in boundary_ends:
            ax.axhline(y=boundaries["correct_object"] - 0.5, color="red", linestyle="--", linewidth=2)
            ax.axhline(y=boundary_ends["correct_object"] + 0.5, color="red", linestyle="--", linewidth=2)
        if "incorrect_object" in boundaries and "incorrect_object" in boundary_ends:
            ax.axhline(y=boundaries["incorrect_object"] - 0.5, color="blue", linestyle="--", linewidth=2)
            ax.axhline(y=boundary_ends["incorrect_object"] + 0.5, color="blue", linestyle="--", linewidth=2)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("logit diff", fontsize=12)
        cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_layer_importance_bars(
    mlp_data: np.ndarray,
    attn_data: np.ndarray,
    xattn_data: Optional[np.ndarray],
    model_label: str,
    save_path,
):
    """Per-layer component importance bar chart."""
    num_layers = mlp_data.shape[1]
    x = np.arange(num_layers)
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, mlp_data.sum(axis=0), w, label="MLP", color="coral", alpha=0.85)
    ax.bar(x, attn_data.sum(axis=0), w, label="Self-Attn", color="steelblue", alpha=0.85)
    if xattn_data is not None and np.any(xattn_data != 0):
        ax.bar(x + w, xattn_data.sum(axis=0), w, label="Cross-Attn", color="seagreen", alpha=0.85)
    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Δ Logit Diff", fontsize=14, fontweight="bold")
    ax.set_title(f"{model_label} — Per-Layer Component Importance", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_layers)], fontsize=11)
    ax.legend(fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_token_importance(
    mlp_data: np.ndarray,
    attn_data: np.ndarray,
    xattn_data: Optional[np.ndarray],
    token_labels: List[str],
    model_label: str,
    save_path,
):
    """Per-token component importance line plot."""
    num_tokens = mlp_data.shape[0]
    x = np.arange(num_tokens)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, mlp_data.sum(axis=1), label="MLP", color="coral", linewidth=2)
    ax.plot(x, attn_data.sum(axis=1), label="Self-Attn", color="steelblue", linewidth=2)
    if xattn_data is not None and np.any(xattn_data != 0):
        ax.plot(x, xattn_data.sum(axis=1), label="Cross-Attn", color="seagreen", linewidth=2)
    ax.set_xlabel("Token Position", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Δ Logit Diff", fontsize=14, fontweight="bold")
    ax.set_title(f"{model_label} — Per-Token Component Importance", fontsize=15, fontweight="bold")
    ax.legend(fontsize=13)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")
