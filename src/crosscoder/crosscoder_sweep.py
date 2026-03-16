"""
Crosscoder hyperparameter sweep for Qwen3-VL and BLIP-VQA on Visual-Counterfact.
Sweeps expansion factor (4, 8) and TopK (400, 800); runs 100 epochs per config;
stores FVE_C, FVE_U, dead latents in results_sweep/sweep_results.csv. No plotting.
"""

import csv
from pathlib import Path

from . import config
from .activations import extract_activations_for_config
from .utils import (
    get_activations_dir,
    get_metrics_dir,
    get_results_dir,
    load_activations,
    load_json,
    save_activations,
)
from .train import train_crosscoder

# Sweep config (rest from config.py)
SWEEP_MODELS = ["qwen3vl", "blip2"]
SWEEP_METHODS = ["wanda", "awq"]
SWEEP_COMPONENTS = ["V_P"]
SWEEP_TOKEN_TYPE = "cls"
SWEEP_EXPANSION_FACTORS = [4, 8]
SWEEP_TOPKS = [400, 800]
SWEEP_NUM_EPOCHS = 100

RESULTS_SWEEP_DIR = config.CROSSCODER_RESULTS_DIR / "results_sweep"


def _ensure_activations(model: str, method: str, component: str, token_type: str) -> Path:
    """Extract activations if missing; return path to activations.pt."""
    results_dir = get_results_dir(model, method, component, token_type)
    activations_dir = get_activations_dir(results_dir)
    activations_path = activations_dir / "activations.pt"
    if activations_path.exists():
        return activations_path
    print(f"Extracting activations: {model}/{method}/{component}/{token_type}")
    activations_data = extract_activations_for_config(model, method, component, token_type)
    activations_dir.mkdir(parents=True, exist_ok=True)
    save_activations(activations_data, activations_path)
    return activations_path


def _sweep_results_dir(
    model: str,
    method: str,
    component: str,
    token_type: str,
    expansion: int,
    topk: int,
) -> Path:
    """Unique results dir for this sweep config (no overwrites)."""
    name = f"{model}__{method}__{component}__{token_type}__ef{expansion}__k{topk}"
    d = RESULTS_SWEEP_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_sweep():
    RESULTS_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_SWEEP_DIR / "sweep_results.csv"
    rows: list[dict] = []

    for model in SWEEP_MODELS:
        for method in SWEEP_METHODS:
            for component in SWEEP_COMPONENTS:
                token_type = SWEEP_TOKEN_TYPE
                activations_path = _ensure_activations(model, method, component, token_type)
                activations_data = load_activations(activations_path)

                for expansion in SWEEP_EXPANSION_FACTORS:
                    for topk in SWEEP_TOPKS:
                        sweep_dir = _sweep_results_dir(
                            model, method, component, token_type, expansion, topk
                        )
                        # Skip if already run (final metrics exist)
                        metrics_dir = get_metrics_dir(sweep_dir)
                        metrics_file = metrics_dir / "training_metrics.json"
                        if metrics_file.exists():
                            print(f"Skip (already run): {sweep_dir.name}")
                            hist = load_json(metrics_file)
                            last = len(hist.get("epochs", [])) - 1
                            if last >= 0:
                                fve_u = hist["val_fve_u"][last]
                                fve_c = hist["val_fve_c"][last]
                                dead = hist["dead_neurons"][last]
                            else:
                                fve_u = fve_c = dead = float("nan")
                            rows.append({
                                "model": model,
                                "method": method,
                                "component": component,
                                "token_type": token_type,
                                "expansion_factor": expansion,
                                "topk": topk,
                                "FVE_U": fve_u,
                                "FVE_C": fve_c,
                                "dead_latents": dead,
                            })
                            continue

                        print(f"Training: {sweep_dir.name}")
                        try:
                            result = train_crosscoder(
                                activations_data=activations_data,
                                model_name=model,
                                method=method,
                                component=component,
                                token_type=token_type,
                                num_epochs=SWEEP_NUM_EPOCHS,
                                results_dir=sweep_dir,
                                expansion_factor=expansion,
                                topk=topk,
                            )
                            fm = result["final_metrics"]
                            rows.append({
                                "model": model,
                                "method": method,
                                "component": component,
                                "token_type": token_type,
                                "expansion_factor": expansion,
                                "topk": topk,
                                "FVE_U": fm["fve_u"],
                                "FVE_C": fm["fve_c"],
                                "dead_latents": fm["dead_neurons"],
                            })
                        except Exception as e:
                            print(f"  Failed: {e}")
                            fve_u = fve_c = dead = float("nan")
                            metrics_file = get_metrics_dir(sweep_dir) / "training_metrics.json"
                            if metrics_file.exists():
                                hist = load_json(metrics_file)
                                last = len(hist.get("epochs", [])) - 1
                                if last >= 0:
                                    fve_u = hist["val_fve_u"][last]
                                    fve_c = hist["val_fve_c"][last]
                                    dead = hist["dead_neurons"][last]
                            rows.append({
                                "model": model,
                                "method": method,
                                "component": component,
                                "token_type": token_type,
                                "expansion_factor": expansion,
                                "topk": topk,
                                "FVE_U": fve_u,
                                "FVE_C": fve_c,
                                "dead_latents": dead,
                                "error": str(e),
                            })

    fieldnames = [
        "model", "method", "component", "token_type",
        "expansion_factor", "topk",
        "FVE_U", "FVE_C", "dead_latents", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    run_sweep()
