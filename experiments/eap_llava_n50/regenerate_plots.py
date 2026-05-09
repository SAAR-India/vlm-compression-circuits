"""Regenerate plots for both variants from the merged patching_data.npz files.

Reads only the npz arrays; does not require shards or raw pickles.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.activation_patching.plotting import (  # noqa: E402
    plot_layer_importance_bars,
    plot_notice_heatmaps,
    plot_token_importance,
)


VARIANTS = {
    "baseline": "LLaVA-1.5-7B Uncompressed",
    "awq": "LLaVA-1.5-7B AWQ",
}


def regen(variant: str, label: str) -> None:
    vdir = HERE / f"llava__{variant}"
    npz = vdir / "patching_data.npz"
    if not npz.exists():
        print(f"SKIP {variant}: {npz} missing")
        return
    data = np.load(npz, allow_pickle=True)
    mlp = data["mlp"]
    attn = data["attn"]
    token_labels = list(data["token_labels"])

    plots_dir = vdir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_notice_heatmaps(mlp, attn, None, token_labels, label, plots_dir / "notice_heatmap.png")
    plot_layer_importance_bars(mlp, attn, None, label, plots_dir / "layer_importance.png")
    plot_token_importance(mlp, attn, None, token_labels, label, plots_dir / "token_importance.png")
    print(f"[{variant}] regenerated -> {plots_dir}")


if __name__ == "__main__":
    for v, lab in VARIANTS.items():
        regen(v, lab)
