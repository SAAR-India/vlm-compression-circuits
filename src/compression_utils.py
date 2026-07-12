import gc
import os
import time
import json
from pathlib import Path
from typing import List, Tuple

import torch
import pandas as pd
from tabulate import tabulate

import compression_configs
from model_evals import EVAL_DATASETS
from run_compression_eval import apply_wanda, build_awq_state_dict, save_awq_checkpoint

def get_submodule(model, dotted_path: str):
    """Safely traverse a dot-separated path like 'model.vision_tower'."""
    m = model
    for attr in dotted_path.split("."):
        if not hasattr(m, attr):
            raise AttributeError(
                f"Module has no attribute '{attr}'. "
                f"Full path: '{dotted_path}'. "
                f"Available: {[n for n, _ in m.named_children()]}"
            )
        m = getattr(m, attr)
    return m


def get_module_paths(model_name: str, components: List[str]) -> List[str]:
    """Return the actual dotted attribute paths for given abstract components."""
    return [compression_configs.MODULE_MAP[model_name][c] for c in components]


def count_params(model, module_path: str = None) -> Tuple[int, int]:
    """Return (total_params, nonzero_params) for a module."""
    target = get_submodule(model, module_path) if module_path else model
    total = 0
    nonzero = 0
    for p in target.parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()
    return total, nonzero


def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "no GPU"
    used = torch.cuda.memory_allocated(0) / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return f"{used:.0f}/{total:.0f} MB"


def load_log() -> dict:
    if os.path.exists(compression_configs.LOG_FILE):
        with open(compression_configs.LOG_FILE) as f:
            return json.load(f)
    return {"done_compress": [], "done_eval": [], "timings": {}}


def save_log(log: dict):
    with open(compression_configs.LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def _sanitize_comp_label(comp_label: str) -> str:
    """Use in folder/job names: no '+', use '_' (e.g. V+P -> V_P)."""
    return comp_label.replace("+", "_")


def jid(model_name: str, method: str, comp_label: str) -> str:
    return f"{model_name}__{method}__{_sanitize_comp_label(comp_label)}"


def is_done(log: dict, stage: str, job_id: str) -> bool:
    return job_id in log.get(f"done_{stage}", [])


def mark_done(log: dict, stage: str, job_id: str, elapsed: float = 0):
    log.setdefault(f"done_{stage}", []).append(job_id)
    log.setdefault("timings", {})[f"{stage}_{job_id}"] = round(elapsed, 1)
    save_log(log)


def out_path(model_name: str, method: str, comp_label: str) -> str:
    return os.path.join(compression_configs.OUTPUT_DIR, f"{model_name}__{method}__{_sanitize_comp_label(comp_label)}")

def load_model(model_name: str):
    """Load a VLM in FP16 on the default GPU (e.g. single A6000)."""
    cfg = compression_configs.MODEL_CONFIGS[model_name]
    device_map = "cuda:0" if torch.cuda.is_available() else "auto"

    if model_name == "blip2":
        from transformers import BlipForQuestionAnswering, BlipProcessor
        model = BlipForQuestionAnswering.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map=device_map,
        )
        processor = BlipProcessor.from_pretrained(cfg["model_id"])
    elif model_name == "qwen3vl":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"])
    else:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], use_fast=False)

    return model, processor


def run_compression(quick: bool = False):
    os.makedirs(compression_configs.OUTPUT_DIR, exist_ok=True)
    log = load_log()

    models = ["llava15"] if quick else list(compression_configs.MODEL_CONFIGS.keys())
    methods = ["wanda"] if quick else compression_configs.METHODS
    combos = {"V": compression_configs.COMPONENT_COMBOS["V"]} if quick else compression_configs.COMPONENT_COMBOS

    total = len(models) * len(methods) * len(combos)
    n = 0

    for model_name in models:
        for method in methods:
            for comp_label, components in combos.items():
                n += 1
                job = jid(model_name, method, comp_label)

                if is_done(log, "compress", job):
                    print(f"[{n}/{total}] SKIP (done): {job}")
                    continue

                opath = out_path(model_name, method, comp_label)
                os.makedirs(opath, exist_ok=True)

                print(f"\n[{n}/{total}] COMPRESS: {job}")
                print(f"  Output: {opath}")
                t0 = time.time()

                flush_gpu()
                model, processor = load_model(model_name)
                print(f"  Loaded ({gpu_mem()})")

                paths = get_module_paths(model_name, components)
                for comp, path in zip(components, paths):
                    total_p, nz = count_params(model, path)
                    print(f"  Pre:  {comp} ({path}): {total_p/1e6:.1f}M params")

                if method == "wanda":
                    model = apply_wanda(model, model_name, components,
                                       compression_configs.METHOD_CONFIGS[method])
                    for comp, path in zip(components, paths):
                        total_p, nz = count_params(model, path)
                        sp = 1.0 - (nz / total_p) if total_p else 0
                        print(f"  Post: {comp} ({path}): {nz/1e6:.1f}M nonzero, {sp:.1%} sparse")
                    print(f"  Saving to {opath}...")
                    model.save_pretrained(opath, max_shard_size="2GB")
                    processor.save_pretrained(opath)
                    meta = {
                        "model": model_name, "method": method,
                        "components": components, "comp_label": comp_label,
                        "config": compression_configs.METHOD_CONFIGS[method],
                        "module_paths": paths,
                    }
                    with open(os.path.join(opath, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=2)
                elif method == "awq":
                    print(f"  Building AWQ state dict (packed INT4 + scale/zero_point)...")
                    state_dict, quantized_layers = build_awq_state_dict(
                        model, model_name, components, compression_configs.METHOD_CONFIGS[method]
                    )
                    n_quant = len(quantized_layers)
                    print(f"  Post: {n_quant} linear layer(s) stored as INT4 (packed) + scales/zeros")
                    print(f"  Saving to {opath}...")
                    save_awq_checkpoint(
                        opath,
                        model,
                        model_name,
                        state_dict,
                        quantized_layers,
                        processor,
                        components,
                        comp_label,
                        paths,
                        compression_configs.METHOD_CONFIGS[method],
                    )

                del model, processor
                flush_gpu()

                elapsed = time.time() - t0
                mark_done(log, "compress", job, elapsed)
                print(f"  Done in {elapsed:.0f}s")

def collect_results() -> pd.DataFrame:
    rows = []
    rdir = Path(compression_configs.RESULTS_DIR)
    if not rdir.exists():
        return pd.DataFrame()

    for run_dir in sorted(rdir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "scripts":
            continue

        parts = run_dir.name.split("__")
        if len(parts) != 3:
            continue
        model_name, method, comp_label = parts

        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue

        try:
            data = json.load(open(results_file))
            for task, metrics in data.get("results", {}).items():
                row = {"model": model_name, "method": method,
                       "components": comp_label, "task": task}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        row[k] = v
                rows.append(row)
        except Exception:
            pass

    return pd.DataFrame(rows)


def generate_table():
    df = collect_results()

    if df.empty:
        print("\nNo results yet. Expected table:\n")
        _template()
        return

    print("\n" + "=" * 90)
    print("RESULTS: Component-wise Compression")
    print("=" * 90)

    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        print(f"\n--- {model.upper()} ---\n")
        pivot = mdf.pivot_table(index=["method", "components"], columns="task",
                                values="accuracy", aggfunc="first")
        if not pivot.empty:
            print(tabulate(pivot, headers="keys", tablefmt="github", floatfmt=".2f"))

    csv = os.path.join(compression_configs.RESULTS_DIR, "results.csv")
    df.to_csv(csv, index=False)
    print(f"\nSaved: {csv}")


def _template():
    header = ["Model", "Method", "Components"]
    header += list(compression_configs.EVAL_DATASETS.keys())
    rows = []
    for model in ["blip2", "qwen3vl", "llava15"]:
        rows.append([model, "FP16", "—"] + ["—"] * len(EVAL_DATASETS))
        for method in ["wanda", "awq"]:
            for comp in ["V", "V_P"]:
                rows.append([model, method, comp] + ["—"] * len(EVAL_DATASETS))
        rows.append([""] * (3 + len(EVAL_DATASETS)))

    print(tabulate(rows, headers=header, tablefmt="github"))
    print("\nV=Vision, P=Projector/Q-Former (no LLM compression)")
