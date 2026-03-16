"""
Compute plan-aligned metrics (FSR, SSS, CSS, Jaccard) for crosscoder results.
Outputs to results/plan_metrics/ as CSV files.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config
from .classify import load_classification_results
from .metrics import (
    compute_counterfactual_sensitivity_shift,
    compute_feature_sharing_ratio,
    compute_jaccard_class_distributions,
    compute_semantic_stability_score,
)
from .visualize import compute_adaptive_rho_thresholds
from .utils import get_features_dir, get_metrics_dir, get_results_dir, load_json


def parse_config_dirname(name: str) -> Optional[dict]:
    """Parse dirname like blip2__wanda__V_P__cls into components."""
    parts = name.split("__")
    if len(parts) != 4:
        return None
    model, method, component, token_type = parts
    if model not in config.MODELS or method not in config.METHODS:
        return None
    if component not in config.COMPONENTS or token_type not in config.TOKEN_TYPES:
        return None
    return {"model": model, "method": method, "component": component, "token_type": token_type}


def discover_configs(results_dir: Path) -> List[dict]:
    """Discover configs from subdirs matching model__method__component__token_type."""
    configs = []
    for p in results_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = parse_config_dirname(p.name)
        if parsed is None:
            continue
        features_dir = p / "features"
        if not (features_dir / "feature_classification.csv").exists():
            continue
        configs.append(parsed)
    return configs


def compute_metrics_for_config(
    model: str,
    method: str,
    component: str,
    token_type: str,
    output_dir: Path,
) -> Optional[dict]:
    """Compute FSR (classification-based: shared/(shared+exclusive)), SSS, CSS for one config."""
    results_dir = get_results_dir(model, method, component, token_type)
    features_dir = get_features_dir(results_dir)

    classification_path = features_dir / "feature_classification.csv"
    merged_path = features_dir / "merged_classification.csv"

    if not classification_path.exists():
        return None

    classification_df = load_classification_results(str(classification_path))
    fsr = compute_feature_sharing_ratio(classification_df)
    sss = compute_semantic_stability_score(classification_df)

    css = {}
    if merged_path.exists():
        merged_df = pd.read_csv(merged_path)
        css = compute_counterfactual_sensitivity_shift(merged_df)

    row = {
        "model": model,
        "method": method,
        "component": component,
        "token_type": token_type,
        "fsr": fsr,
        "sss": sss,
    }
    for k, v in css.items():
        row[f"css_{k}"] = v
    return row


def compute_classification_thresholds(
    configs: List[dict],
) -> List[dict]:
    """
    Store GMM-based rho thresholds and theta thresholds used for classification.
    feature_classification.csv and all evaluation use these GMM-derived thresholds.
    """
    rows = []
    for c in configs:
        results_dir = get_results_dir(
            c["model"], c["method"], c["component"], c["token_type"]
        )
        features_dir = get_features_dir(results_dir)
        classification_path = features_dir / "feature_classification.csv"
        if not classification_path.exists():
            continue

        classification_df = load_classification_results(str(classification_path))
        thresh = compute_adaptive_rho_thresholds(classification_df)

        rows.append({
            "model": c["model"],
            "method": c["method"],
            "component": c["component"],
            "token_type": c["token_type"],
            "rho_uncompressed_only": thresh["rho_uncompressed_only"],
            "rho_compressed_only": thresh["rho_compressed_only"],
            "rho_shared_low": thresh["rho_shared_low"],
            "rho_shared_high": thresh["rho_shared_high"],
            "theta_aligned": config.THETA_ALIGNED,
            "theta_redirected": config.THETA_REDIRECTED,
        })
    return rows


def compute_jaccard_pairs(
    configs: List[dict],
    output_dir: Path,
) -> List[dict]:
    """Compute Jaccard for wanda vs awq pairs per (model, component, token_type)."""
    results_dir = config.CROSSCODER_RESULTS_DIR
    rows = []

    # Group by (model, component, token_type)
    groups = {}
    for c in configs:
        key = (c["model"], c["component"], c["token_type"])
        if key not in groups:
            groups[key] = []
        groups[key].append(c)

    for (model, component, token_type), group in groups.items():
        wanda = next((g for g in group if g["method"] == "wanda"), None)
        awq = next((g for g in group if g["method"] == "awq"), None)
        if wanda is None or awq is None:
            continue

        wanda_dir = results_dir / f"{model}__wanda__{component}__{token_type}"
        awq_dir = results_dir / f"{model}__awq__{component}__{token_type}"
        wanda_cls = get_features_dir(wanda_dir) / "feature_classification.csv"
        awq_cls = get_features_dir(awq_dir) / "feature_classification.csv"

        if not wanda_cls.exists() or not awq_cls.exists():
            continue

        df_a = load_classification_results(str(wanda_cls))
        df_b = load_classification_results(str(awq_cls))
        result = compute_jaccard_class_distributions(df_a, df_b)

        rows.append({
            "model": model,
            "component": component,
            "token_type": token_type,
            "config_a": f"{model}__wanda__{component}__{token_type}",
            "config_b": f"{model}__awq__{component}__{token_type}",
            "jaccard_macro": result["macro_mean"],
            "jaccard_per_class": json.dumps(result["per_class"]),
        })
    return rows


def compute_shared_geometry_plan_rows(configs: List[dict]) -> List[dict]:
    """Aggregate per-config shared geometry metrics for Wanda vs AWQ comparisons."""
    from .metrics import SHARED_CLASSES

    rows = []
    for c in configs:
        results_dir = get_results_dir(
            c["model"], c["method"], c["component"], c["token_type"]
        )
        metrics_dir = get_metrics_dir(results_dir)
        path = metrics_dir / "shared_geometry_metrics.json"
        if not path.exists():
            continue

        data = load_json(path)
        row = {
            "model": c["model"],
            "method": c["method"],
            "component": c["component"],
            "token_type": c["token_type"],
        }
        for cls in list(SHARED_CLASSES) + ["all_shared"]:
            sub = data.get(cls, {})
            if isinstance(sub, dict) and sub.get("n", 0) > 0:
                row[f"{cls}_angle_deg_mean"] = sub.get("angle_deg_mean")
                row[f"{cls}_norm_ratio_raw_mean"] = sub.get("norm_ratio_raw_mean")
                if cls == "all_shared":
                    lm = sub.get("linear_map")
                    if isinstance(lm, dict):
                        row["all_shared_sv_mean"] = lm.get("sv_mean")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute plan-aligned FSR, SSS, CSS, Jaccard for crosscoder results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.CROSSCODER_RESULTS_DIR / "plan_metrics",
        help="Output directory for CSV files (default: results/plan_metrics)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        help="Specific configs (model__method__component__token_type)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all existing result configs",
    )
    parser.add_argument(
        "--jaccard-pairs",
        action="store_true",
        default=True,
        help="Compute Jaccard for wanda vs awq pairs (default: True)",
    )
    parser.add_argument(
        "--no-jaccard-pairs",
        action="store_false",
        dest="jaccard_pairs",
        help="Skip Jaccard computation",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.configs:
        configs = []
        for c in args.configs:
            parsed = parse_config_dirname(c)
            if parsed is None:
                print(f"Warning: invalid config '{c}', skipping")
                continue
            results_dir = config.CROSSCODER_RESULTS_DIR / c
            if (results_dir / "features" / "feature_classification.csv").exists():
                configs.append(parsed)
            else:
                print(f"Warning: no feature_classification.csv for '{c}', skipping")
    elif args.all:
        configs = discover_configs(config.CROSSCODER_RESULTS_DIR)
    else:
        configs = discover_configs(config.CROSSCODER_RESULTS_DIR)

    if not configs:
        print("No configs to process.")
        return

    print(f"Processing {len(configs)} configs...")

    fsr_rows = []
    for c in configs:
        row = compute_metrics_for_config(
            model=c["model"],
            method=c["method"],
            component=c["component"],
            token_type=c["token_type"],
            output_dir=output_dir,
        )
        if row is not None:
            fsr_rows.append(row)
            print(f"  {c['model']}__{c['method']}__{c['component']}__{c['token_type']}")

    if fsr_rows:
        fsr_df = pd.DataFrame(fsr_rows)
        fsr_path = output_dir / "fsr_sss_css.csv"
        fsr_df.to_csv(fsr_path, index=False)
        print(f"Saved: {fsr_path}")

    threshold_rows = compute_classification_thresholds(configs)
    if threshold_rows:
        threshold_df = pd.DataFrame(threshold_rows)
        threshold_path = output_dir / "classification_thresholds.csv"
        threshold_df.to_csv(threshold_path, index=False)
        print(f"Saved: {threshold_path}")

    if args.jaccard_pairs:
        jaccard_rows = compute_jaccard_pairs(configs, output_dir)
        if jaccard_rows:
            jaccard_df = pd.DataFrame(jaccard_rows)
            jaccard_path = output_dir / "jaccard_class_distributions.csv"
            jaccard_df.to_csv(jaccard_path, index=False)
            print(f"Saved: {jaccard_path}")

    geometry_rows = compute_shared_geometry_plan_rows(configs)
    if geometry_rows:
        geometry_df = pd.DataFrame(geometry_rows)
        geometry_path = output_dir / "shared_geometry_metrics_summary.csv"
        geometry_df.to_csv(geometry_path, index=False)
        print(f"Saved: {geometry_path}")

    print("Done.")


if __name__ == "__main__":
    main()
