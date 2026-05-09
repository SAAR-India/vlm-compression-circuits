"""Merge per-shard EAP results for a llava variant into the canonical layout.

Reads src/activation_patching/results/llava__{variant}__shard*/raw_sample_results.pkl,
concatenates all_results, re-runs _aggregate_by_semantic_group, writes:
  src/activation_patching/results/llava__{variant}/
    patching_data.npz, metrics/patching_metrics.json, plots/*.png

Usage:
    python scripts/merge_shards.py --variants baseline awq --with_compare
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.activation_patching import config as patching_config
from src.activation_patching.llava_patching import (
    TOKEN_LABELS,
    _aggregate_by_semantic_group,
)
from src.activation_patching.plotting import (
    plot_layer_importance_bars,
    plot_notice_heatmaps,
    plot_token_importance,
)


def merge_variant(variant: str) -> dict:
    results_dir = patching_config.PATCHING_RESULTS_DIR
    shard_dirs = sorted(results_dir.glob(f"llava__{variant}__shard*"))
    if not shard_dirs:
        raise SystemExit(f"no shards found for variant={variant}")
    print(f"[{variant}] merging {len(shard_dirs)} shards")

    all_results = []
    num_layers = None
    for d in shard_dirs:
        pkl = d / "raw_sample_results.pkl"
        if not pkl.exists():
            print(f"  WARN missing {pkl}")
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        nl = data["num_layers"]
        num_layers = num_layers or nl
        assert nl == num_layers, f"layer mismatch {nl} vs {num_layers}"
        all_results.extend(data["all_results"])
        print(f"  + {d.name}: {len(data['all_results'])} samples")

    if not all_results:
        raise SystemExit(f"no samples merged for {variant}")

    avg_mlp, avg_attn, group_labels = _aggregate_by_semantic_group(all_results, num_layers)

    out_path = results_dir / f"llava__{variant}"
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_path / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    per_layer = {
        "mlp": avg_mlp.sum(axis=0).tolist(),
        "self_attention": avg_attn.sum(axis=0).tolist(),
        "cross_attention": [],
    }
    per_component = {
        "mlp": float(avg_mlp.sum()),
        "self_attention": float(avg_attn.sum()),
        "cross_attention": 0.0,
    }
    metrics = {
        "model": "llava",
        "variant": variant,
        "model_label": f"LLaVA-1.5-7B {variant.upper() if variant != 'baseline' else 'Uncompressed'}",
        "num_samples": len(all_results),
        "num_layers": int(num_layers),
        "per_layer": per_layer,
        "per_component": per_component,
        "token_labels": group_labels,
    }
    with open(metrics_dir / "patching_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        out_path / "patching_data.npz",
        mlp=avg_mlp,
        attn=avg_attn,
        xattn=np.zeros_like(avg_mlp),
        token_labels=group_labels,
    )

    label = metrics["model_label"]
    plot_notice_heatmaps(avg_mlp, avg_attn, None, group_labels, label, plots_dir / "notice_heatmap.png")
    plot_layer_importance_bars(avg_mlp, avg_attn, None, label, plots_dir / "layer_importance.png")
    plot_token_importance(avg_mlp, avg_attn, None, group_labels, label, plots_dir / "token_importance.png")

    print(f"[{variant}] merged {len(all_results)} samples -> {out_path}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline", "wanda", "awq"])
    ap.add_argument("--with_compare", action="store_true",
                    help="also run baseline-vs-{wanda,awq}")
    args = ap.parse_args()

    all_metrics = {}
    for v in args.variants:
        all_metrics[v] = merge_variant(v)

    if args.with_compare and "baseline" in all_metrics:
        from src.activation_patching.metrics import compute_comparison_metrics
        for comp in ["wanda", "awq"]:
            if comp not in all_metrics:
                continue
            mu, mc = all_metrics["baseline"], all_metrics[comp]
            if "error" in mu or "error" in mc:
                continue
            comp_metrics = compute_comparison_metrics(mu, mc)
            comp_dir = patching_config.PATCHING_RESULTS_DIR / f"llava__{comp}" / "metrics"
            comp_dir.mkdir(parents=True, exist_ok=True)
            with open(comp_dir / "comparison_metrics.json", "w") as f:
                json.dump(comp_metrics, f, indent=2)
            agg = comp_metrics["aggregate"]
            print(
                f"{comp} vs baseline: "
                f"Jaccard={agg['jaccard_mean']:.3f} "
                f"Spearman={agg['spearman_mean']:.3f} "
                f"Stability={agg['stability_mean']:.3f}"
            )


if __name__ == "__main__":
    main()
