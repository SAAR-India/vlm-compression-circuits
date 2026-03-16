"""
Activation patching pipeline: Edge Activation Patching / NOTICE-style.

Runs patching on Visual-Counterfact (output/counterfactual_selected) for:
- blip2 (blip-vqa-base): uncompressed + compressed (wanda, awq)
- qwen3vl (Qwen3-VL-2B): uncompressed + compressed (when supported)
- llava (LLaVA-1.5-7B): when compressed checkpoints available

Stores per-layer, per-component metrics; computes Jaccard, Spearman rho, stability
between compressed and uncompressed; produces NOTICE-style heatmaps.

Usage:
  source .venv/bin/activate
  python -m src.activation_patching.main --model blip2 --num_samples 100
  python -m src.activation_patching.main --model blip2 --all_variants
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.activation_patching import config as patching_config
from src.activation_patching.blip_patching import run_blip_patching
from src.activation_patching.llava_patching import run_llava_patching
from src.activation_patching.metrics import compute_comparison_metrics
from src.activation_patching.qwen3vl_patching import run_qwen3vl_patching


def run_blip2_pipeline(
    num_samples: int = 100,
    output_dir: Path = None,
    variants: list = None,
    patch_batch_size: int = None,
    use_amp: bool = None,
) -> dict:
    """Run patching for BLIP-2 across baseline and compressed variants."""
    output_dir = output_dir or patching_config.PATCHING_RESULTS_DIR
    variants = variants or ["baseline", "wanda", "awq"]
    all_metrics = {}
    for v in tqdm(variants, desc="Variants", unit="variant"):
        print(f"\n{'='*60}")
        print(f"BLIP-2 Patching: {v}")
        print(f"{'='*60}")
        m = run_blip_patching(
            model_variant=v,
            num_samples=num_samples,
            output_dir=output_dir,
            patch_batch_size=patch_batch_size,
            use_amp=use_amp,
        )
        all_metrics[v] = m

    for comp_v in ["wanda", "awq"]:
        if comp_v in all_metrics and "baseline" in all_metrics:
            mu = all_metrics["baseline"]
            mc = all_metrics[comp_v]
            if "error" not in mu and "error" not in mc:
                comp_metrics = compute_comparison_metrics(mu, mc)
                comp_dir = output_dir / f"blip2__{comp_v}" / "metrics"
                comp_dir.mkdir(parents=True, exist_ok=True)
                with open(comp_dir / "comparison_metrics.json", "w") as f:
                    json.dump(comp_metrics, f, indent=2)
                print(f"\n  {comp_v} vs baseline: Jaccard={comp_metrics['aggregate']['jaccard_mean']:.3f}, "
                      f"Spearman={comp_metrics['aggregate']['spearman_mean']:.3f}, "
                      f"Stability={comp_metrics['aggregate']['stability_mean']:.3f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Activation patching for VLMs on Visual-Counterfact"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["blip2", "qwen3vl", "llava"],
        default="blip2",
        help="Model to patch (blip2 supported; qwen3vl/llava require architecture adapters)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for patching",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["baseline", "wanda", "awq"],
        default=None,
        help="Single variant to run (default: all)",
    )
    parser.add_argument(
        "--all_variants",
        action="store_true",
        help="Run baseline + wanda + awq and compute comparison metrics",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--patch_batch_size",
        type=int,
        default=None,
        help="Token patches per forward (default: 16; reduces L*3*T forwards to L*3*ceil(T/B))",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed precision (use FP32); may be slower but more stable",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else patching_config.PATCHING_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "blip2":
        variants = ["baseline", "wanda", "awq"] if args.all_variants else [args.variant or "baseline"]
        run_blip2_pipeline(
            num_samples=args.num_samples,
            output_dir=output_dir,
            variants=variants,
            patch_batch_size=args.patch_batch_size,
            use_amp=False if args.no_amp else None,
        )
    elif args.model == "qwen3vl":
        variants = ["baseline", "wanda", "awq"] if args.all_variants else [args.variant or "baseline"]
        all_metrics = {}
        for v in tqdm(variants, desc="Variants", unit="variant"):
            print(f"\n{'='*60}")
            print(f"Qwen3-VL Patching: {v}")
            print(f"{'='*60}")
            m = run_qwen3vl_patching(
                model_variant=v,
                num_samples=args.num_samples,
                output_dir=output_dir,
                patch_batch_size=args.patch_batch_size,
                use_amp=False if args.no_amp else None,
            )
            all_metrics[v] = m
        for comp_v in ["wanda", "awq"]:
            if comp_v in all_metrics and "baseline" in all_metrics:
                mu = all_metrics["baseline"]
                mc = all_metrics[comp_v]
                if "error" not in mu and "error" not in mc:
                    comp_metrics = compute_comparison_metrics(mu, mc)
                    comp_dir = output_dir / f"qwen3vl__{comp_v}" / "metrics"
                    comp_dir.mkdir(parents=True, exist_ok=True)
                    with open(comp_dir / "comparison_metrics.json", "w") as f:
                        json.dump(comp_metrics, f, indent=2)
                    print(f"\n  {comp_v} vs baseline: Jaccard={comp_metrics['aggregate']['jaccard_mean']:.3f}, "
                          f"Spearman={comp_metrics['aggregate']['spearman_mean']:.3f}, "
                          f"Stability={comp_metrics['aggregate']['stability_mean']:.3f}")
    elif args.model == "llava":
        variants = ["baseline", "wanda", "awq"] if args.all_variants else [args.variant or "baseline"]
        all_metrics = {}
        for v in tqdm(variants, desc="Variants", unit="variant"):
            print(f"\n{'='*60}")
            print(f"LLaVA-1.5-7B Patching: {v}")
            print(f"{'='*60}")
            m = run_llava_patching(
                model_variant=v,
                num_samples=args.num_samples,
                output_dir=output_dir,
                patch_batch_size=args.patch_batch_size,
                use_amp=False if args.no_amp else None,
            )
            all_metrics[v] = m
        for comp_v in ["wanda", "awq"]:
            if comp_v in all_metrics and "baseline" in all_metrics:
                mu = all_metrics["baseline"]
                mc = all_metrics[comp_v]
                if "error" not in mu and "error" not in mc:
                    comp_metrics = compute_comparison_metrics(mu, mc)
                    comp_dir = output_dir / f"llava__{comp_v}" / "metrics"
                    comp_dir.mkdir(parents=True, exist_ok=True)
                    with open(comp_dir / "comparison_metrics.json", "w") as f:
                        json.dump(comp_metrics, f, indent=2)
                    print(f"\n  {comp_v} vs baseline: Jaccard={comp_metrics['aggregate']['jaccard_mean']:.3f}, "
                          f"Spearman={comp_metrics['aggregate']['spearman_mean']:.3f}, "
                          f"Stability={comp_metrics['aggregate']['stability_mean']:.3f}")


if __name__ == "__main__":
    main()
