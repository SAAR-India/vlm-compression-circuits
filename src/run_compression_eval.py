"""
Vision + projector compression (Wanda, AWQ) for BLIP-VQA, Qwen3-VL-2B, and LLaVA-1.5-7B.
Models from preprocessing/config.py: BLIP_VQA_MODEL_ID, QWEN3VL_2B_MODEL_ID, LLAVA15_7B_MODEL_ID.
Q-VLM is separate: python src/run_qvlm_compression.py --model llava15 --combo V+P

Usage:
    python src/run_compression_eval.py --stage compress
    python src/run_compression_eval.py --stage eval [--batch_size 64]
    python src/run_compression_eval.py --stage table
    python src/run_compression_eval.py --stage all [--quick]
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

# Print first N samples per dataset (input, prediction, ground truth) for format debugging
DEBUG_EVAL_SAMPLES = 5

import torch
import pandas as pd
from tabulate import tabulate


# =====================================================================
# VERIFIED MODULE MAPS (from inspect_models.py output)
# =====================================================================

# Model IDs from preprocessing/config.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.config import BLIP_VQA_MODEL_ID, LLAVA15_7B_MODEL_ID, QWEN3VL_2B_MODEL_ID

COMP_V = "vision"
COMP_P = "projector"

# Exact attribute paths on the model object (vision + projector only; LLM not compressed)
# BLIP-VQA has vision_model + text_encoder (no qformer)
MODULE_MAP = {
    "blip2": {
        COMP_V: "vision_model",
        COMP_P: "text_encoder",
    },
    "qwen3vl": {
        COMP_V: "model.visual",
        COMP_P: "model.visual.merger",
    },
    "llava15": {
        COMP_V: "model.vision_tower",
        COMP_P: "model.multi_modal_projector",
    },
}

# Only vision and projector compression (no language decoder)
COMPONENT_COMBOS = {
    "V":   [COMP_V],
    "V+P": [COMP_V, COMP_P],
    "P":   [COMP_P]
}

METHODS = ["wanda", "awq"]

METHOD_CONFIGS = {
    "wanda": {"sparsity_ratio": 0.5, "sparsity_type": "unstructured"},
    "awq":   {"w_bit": 4, "q_group_size": 128},
}

MODEL_CONFIGS = {
    "blip2": {
        "model_id": BLIP_VQA_MODEL_ID,
        "model_class": "BlipForQuestionAnswering",
        "processor_class": "BlipProcessor",
    },
    "qwen3vl": {
        "model_id": QWEN3VL_2B_MODEL_ID,
        "model_class": "Qwen3VLForConditionalGeneration",
        "processor_class": "AutoProcessor",
    },
    "llava15": {
        "model_id": LLAVA15_7B_MODEL_ID,
        "model_class": "LlavaForConditionalGeneration",
        "processor_class": "AutoProcessor",
    },
}

OUTPUT_DIR  = "./compressed_models"
RESULTS_DIR = "./eval_results"
LOG_FILE    = "./pipeline_log.json"


# =====================================================================
# UTILITIES
# =====================================================================

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
    return [MODULE_MAP[model_name][c] for c in components]


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


# =====================================================================
# CHECKPOINT / RESUME
# =====================================================================

def load_log() -> dict:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return {"done_compress": [], "done_eval": [], "timings": {}}


def save_log(log: dict):
    with open(LOG_FILE, "w") as f:
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


# =====================================================================
# MODEL LOADING (memory-efficient)
# =====================================================================

def load_model(model_name: str):
    """Load a VLM in FP16 on the default GPU (e.g. single A6000)."""
    cfg = MODEL_CONFIGS[model_name]
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


# =====================================================================
# COMPRESSION: WANDA (magnitude pruning, component-targeted)
# =====================================================================

def apply_wanda(model, model_name: str, components: List[str], config: dict):
    """
    Wanda-style pruning: zero out smallest |W| entries per layer.
    Applied only to Linear layers within targeted components.

    For full activation-aware Wanda, you'd collect input activation norms
    via forward hooks with calibration data. This implementation uses
    magnitude as proxy, which is standard for initial experiments.
    """
    sparsity = config["sparsity_ratio"]
    paths = get_module_paths(model_name, components)

    for path in paths:
        submodule = get_submodule(model, path)
        n_layers = 0

        for name, child in submodule.named_modules():
            if not isinstance(child, torch.nn.Linear):
                continue

            W = child.weight.data
            metric = W.abs().flatten()
            n_prune = int(metric.numel() * sparsity)

            if n_prune == 0 or n_prune >= metric.numel():
                n_layers += 1
                del metric
                continue

            # kthvalue handles arbitrarily large tensors (quantile does not)
            threshold = metric.float().kthvalue(n_prune).values
            mask = W.abs() >= threshold
            child.weight.data.mul_(mask)

            n_layers += 1
            del metric, mask, threshold

        print(f"    [Wanda] {path}: pruned {n_layers} Linear layers @ {sparsity:.0%}")

    return model


# =====================================================================
# COMPRESSION: AWQ (weight-only INT4, component-targeted)
# =====================================================================
#
# We follow standard AWQ practice: store packed INT4 weights with
# scale and zero_point (per-group), and quantization_config in config.json.
# AutoAWQ does not support BLIP-2 (OPT) or all VLM layers, so we use
# the same quantization algorithm but save in standard AWQ format and
# dequantize at load time for evaluation.
#
# =====================================================================

# Pack 8 x 4-bit values into one int32 (LSB to MSB: first value in low 4 bits).
def _pack_int4(w_q: torch.Tensor) -> torch.Tensor:
    """Pack INT4 tensor [..., N] to int32 [..., N//8]. w_q must have last dim divisible by 8."""
    *rest, n = w_q.shape
    w_q = w_q.reshape(-1, 8).to(torch.int32)
    packed = (w_q[:, 0] | (w_q[:, 1] << 4) | (w_q[:, 2] << 8) | (w_q[:, 3] << 12))
    packed = packed | (w_q[:, 4] << 16) | (w_q[:, 5] << 20) | (w_q[:, 6] << 24) | (w_q[:, 7] << 28)
    return packed.reshape(*rest, n // 8)


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 [..., N] to INT4 [..., N*8] (values 0..15)."""
    out_f, in_packed = packed.shape
    w_q = torch.zeros((out_f, in_packed * 8), dtype=torch.int32, device=packed.device)
    for k in range(8):
        w_q[:, k::8] = (packed >> (k * 4)) & 0xF
    return w_q


def build_awq_state_dict(
    model, model_name: str, components: List[str], config: dict
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Build a state dict with AWQ-packed INT4 weights and scale/zero_point for
    targeted Linear layers. Other weights are copied unchanged. Returns
    (state_dict, quantized_layer_names) for saving and for load-time conversion.
    """
    bits = config["w_bit"]
    group_size = config["q_group_size"]
    paths = get_module_paths(model_name, components)
    state_dict = dict(model.state_dict())
    quantized_layers: List[str] = []

    for path in paths:
        submodule = get_submodule(model, path)
        prefix = path + "."

        for name, child in submodule.named_modules():
            if not isinstance(child, torch.nn.Linear):
                continue
            full_key = (prefix + name).rstrip(".")
            weight_key = f"{full_key}.weight"
            if weight_key not in state_dict:
                continue

            W = state_dict[weight_key].float()
            out_f, in_f = W.shape
            if in_f % 8 != 0:
                continue  # skip if not packable

            gs = group_size if (group_size > 0 and in_f % group_size == 0) else in_f
            W_g = W.reshape(out_f, -1, gs)

            w_min = W_g.min(dim=-1, keepdim=True).values
            w_max = W_g.max(dim=-1, keepdim=True).values
            qmax = (1 << bits) - 1
            scale = (w_max - w_min).clamp(min=1e-8) / qmax
            zp = torch.round(-w_min / scale).clamp(0, qmax).to(torch.int32)

            W_q = torch.round(W_g / scale + zp).clamp(0, qmax).to(torch.int32)
            scale = scale.squeeze(-1)  # [out_f, n_groups]
            # Pack weight: [out_f, in_f] -> [out_f, in_f//8]
            packed = _pack_int4(W_q.reshape(out_f, in_f))

            state_dict.pop(weight_key, None)
            state_dict[f"{full_key}.qweight"] = packed.to(torch.int32)
            state_dict[f"{full_key}.scales"] = scale.float()
            state_dict[f"{full_key}.zeros"] = zp
            quantized_layers.append(full_key)

    return state_dict, quantized_layers


def _awq_state_dict_to_fp16(
    state_dict: Dict[str, torch.Tensor],
    quantized_layers: List[str],
    group_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert state dict from AWQ format (qweight, scales, zeros) to FP16 .weight
    for loading into a standard model. Expands scale/zero per group to full dims.
    """
    out = {k: v.clone() for k, v in state_dict.items() if not k.endswith(".qweight") and not k.endswith(".scales") and not k.endswith(".zeros")}
    for full_key in quantized_layers:
        qkey = f"{full_key}.qweight"
        skey = f"{full_key}.scales"
        zkey = f"{full_key}.zeros"
        if qkey not in state_dict:
            continue
        packed = state_dict[qkey]
        scales = state_dict[skey]
        zeros = state_dict[zkey]
        out_f, in_packed = packed.shape
        in_f = in_packed * 8
        # Zeros may be stored as [out_f, n_groups, 1]; squeeze to [out_f, n_groups]
        if zeros.ndim == 3:
            zeros = zeros.squeeze(-1)
        n_groups = scales.shape[1]
        # Unpack INT4
        w_q = _unpack_int4(packed)  # [out_f, in_f]
        # Expand scales and zeros from [out_f, n_groups] to [out_f, in_f]
        scales_exp = scales.repeat_interleave(group_size, dim=1)
        zeros_exp = zeros.repeat_interleave(group_size, dim=1)
        w_fp = (w_q.float() - zeros_exp.float()) * scales_exp
        out[f"{full_key}.weight"] = w_fp.to(torch.float16)
    return out


def save_awq_checkpoint(
    opath: str,
    model,
    model_name: str,
    state_dict: Dict[str, torch.Tensor],
    quantized_layers: List[str],
    processor,
    components: List[str],
    comp_label: str,
    paths: List[str],
    method_config: dict,
) -> None:
    """Save AWQ checkpoint: packed INT4 state dict, config with quantization_config, processor, meta."""
    from safetensors.torch import save_file
    os.makedirs(opath, exist_ok=True)
    # Clone tensors so tied weights (shared memory) are stored separately; safetensors rejects shared memory.
    state_dict_to_save = {k: v.clone() for k, v in state_dict.items()}
    save_file(state_dict_to_save, os.path.join(opath, "model.safetensors"))
    config_dict = model.config.to_dict()
    config_dict["quantization_config"] = {
        "quant_method": "awq",
        "bits": method_config["w_bit"],
        "group_size": method_config["q_group_size"],
        "zero_point": True,
    }
    config_dict["quantized_layers"] = quantized_layers
    config_dict["base_model_id"] = MODEL_CONFIGS[model_name]["model_id"]
    with open(os.path.join(opath, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    processor.save_pretrained(opath)
    meta = {
        "model": model_name,
        "method": "awq",
        "components": components,
        "comp_label": comp_label,
        "config": method_config,
        "module_paths": paths,
    }
    with open(os.path.join(opath, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# =====================================================================
# ORCHESTRATOR
# =====================================================================

def out_path(model_name: str, method: str, comp_label: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{model_name}__{method}__{_sanitize_comp_label(comp_label)}")


def run_compression(quick: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log = load_log()

    models = ["llava15"] if quick else list(MODEL_CONFIGS.keys())
    methods = ["wanda"] if quick else METHODS
    combos = {"V": COMPONENT_COMBOS["V"]} if quick else COMPONENT_COMBOS

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
                                       METHOD_CONFIGS[method])
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
                        "config": METHOD_CONFIGS[method],
                        "module_paths": paths,
                    }
                    with open(os.path.join(opath, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=2)
                elif method == "awq":
                    print(f"  Building AWQ state dict (packed INT4 + scale/zero_point)...")
                    state_dict, quantized_layers = build_awq_state_dict(
                        model, model_name, components, METHOD_CONFIGS[method]
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
                        METHOD_CONFIGS[method],
                    )

                del model, processor
                flush_gpu()

                elapsed = time.time() - t0
                mark_done(log, "compress", job, elapsed)
                print(f"  Done in {elapsed:.0f}s")


# =====================================================================
# EVALUATION — Direct HuggingFace (no lmms-eval dependency)
# =====================================================================
#
# Benchmarks loaded directly from HuggingFace datasets:
#   VQA:  GQA (lmms-lab/GQA: testdev_balanced_instructions + testdev_balanced_images)
#         TextVQA (lmms-lab/textvqa, validation)
#         ScienceQA (derek-thomas/ScienceQA, test, filter has image)
#         COCO Captions 2017 (HF: lmms-lab/COCO-Caption-2017, split val)
#
# =====================================================================

from PIL import Image
import io
import re

# Dataset configs: HF repo, split, and how to extract question/answer/image.
# ScienceQA: derek-thomas/ScienceQA (test), filter rows with image.
# TextVQA: lmms-lab/textvqa (facebook/textvqa uses deprecated loading script).
# GQA: lmms-lab/GQA uses separate configs for instructions (QA) and images; we merge in eval.
EVAL_DATASETS = {
    "scienceqa_img": {
        "hf_path": "derek-thomas/ScienceQA",
        "split": "test",
        "filter_fn": lambda ex: ex.get("image") is not None,
        "build_prompt_fn": "scienceqa",
        "answer_key": "answer",
        "choices_key": "choices",
        "metric": "accuracy",
    },
    "textvqa_val": {
        "hf_path": "lmms-lab/textvqa",
        "split": "validation",
        "build_prompt_fn": "textvqa",
        "answer_key": "answers",
        "metric": "vqa_accuracy",
    },
    "gqa": {
        "hf_path": "lmms-lab/GQA",
        "gqa_instructions_config": "testdev_balanced_instructions",
        "gqa_images_config": "testdev_balanced_images",
        "gqa_split": "testdev",
        "build_prompt_fn": "gqa",
        "answer_key": "answer",
        "metric": "accuracy",
    },
}

# Lite subset for --quick mode
EVAL_DATASETS_LITE = {
    "scienceqa_img": EVAL_DATASETS["scienceqa_img"],
}


def build_prompt_blip2(dataset_name: str, example: dict) -> Tuple[str, Image.Image]:
    """Build prompt + image for BLIP-VQA (question as text; BlipProcessor expects image + text)."""
    img = example.get("image")
    if img is None:
        return None, None
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img)).convert("RGB")
    else:
        img = img.convert("RGB")

    if dataset_name == "scienceqa":
        q = example.get("question", "")
        choices = example.get("choices", [])
        choices_str = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
        prompt = f"Question: {q} Choices: {choices_str} Answer:"
    elif dataset_name == "textvqa":
        q = example.get("question", "")
        # Official BLIP VQA inference uses the raw question as text.
        prompt = q
    elif dataset_name == "gqa":
        q = example.get("question", "")
        prompt = q
    else:
        prompt = "Describe this image."

    return prompt, img


def build_prompt_qwen3vl(dataset_name: str, example: dict) -> Tuple[str, Image.Image]:
    """Build prompt + image for Qwen3-VL-2B (USER/ASSISTANT style)"""
    img = example.get("image")
    if img is None:
        return None, None
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img)).convert("RGB")
    else:
        img = img.convert("RGB")

    if dataset_name == "scienceqa":
        q = example.get("question", "")
        choices = example.get("choices", [])
        choices_str = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
        prompt = (
            "USER: <image>\n"
            f"Question: {q}\n"
            f"Choices: {choices_str}\n"
            "Answer with the option letter.\n"
            "ASSISTANT:"
        )
    elif dataset_name == "textvqa":
        q = example.get("question", "")
        prompt = f"USER: <image>\n{q}\nASSISTANT:"
    elif dataset_name == "gqa":
        q = example.get("question", "")
        prompt = f"USER: <image>\n{q} Answer with a single word.\nASSISTANT:"
    else:
        prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"

    return prompt, img


def extract_answer(dataset_name: str, example: dict) -> str:
    """Extract ground truth answer from dataset example."""
    cfg = None
    for k, v in EVAL_DATASETS.items():
        if v["build_prompt_fn"] == dataset_name or k == dataset_name:
            cfg = v
            break

    if dataset_name == "scienceqa":
        choices = example.get("choices", [])
        ans_idx = example.get("answer", 0)
        if isinstance(ans_idx, int) and ans_idx < len(choices):
            return choices[ans_idx]
        return str(ans_idx)
    elif dataset_name == "textvqa":
        answers = example.get("answers", [])
        if isinstance(answers, list) and len(answers) > 0:
            return answers  # return list for VQA accuracy
        return str(answers)
    elif dataset_name == "gqa":
        return str(example.get("answer", ""))

    return ""


class EvalAIAnswerProcessor:
    """
    Official TextVQA / VQA-style answer normalization (EvalAI).
    Adapted from https://textvqa.org/ evaluation script.
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "id've": "i'd've",
        "i'dve": "i'd've",
        "im": "i'm",
        "ive": "i've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "shes": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }
    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def word_tokenize(self, word: str) -> str:
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text: str) -> str:
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text: str) -> str:
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item: str) -> str:
        item = self.word_tokenize(str(item))
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


_ANSWER_PROCESSOR = EvalAIAnswerProcessor()


def _normalize_answer(ans: str) -> str:
    return _ANSWER_PROCESSOR(ans)


def _scienceqa_pred_to_index(prediction: str, choices: List[str]) -> int:
    pred_raw = str(prediction).strip()
    if not pred_raw:
        return -1

    # First, look for explicit letter choices (A, B, C, D) in the raw string.
    letter_match = re.findall(r"\b([A-Z])\b", pred_raw.upper())
    if letter_match:
        # Use the first letter in case model outputs multiple tokens.
        idx = ord(letter_match[0]) - ord("A")
        if 0 <= idx < len(choices):
            return idx

    # Try numeric index in the raw string.
    digit_match = re.findall(r"\b(\d+)\b", pred_raw)
    if digit_match:
        idx = int(digit_match[0])
        if 0 <= idx < len(choices):
            return idx

    # Fallback: match normalized text to choice strings.
    pred_norm = _normalize_answer(pred_raw)
    for i, choice in enumerate(choices):
        choice_norm = _normalize_answer(choice)
        if pred_norm == choice_norm:
            return i
        if choice_norm and choice_norm in pred_norm:
            return i

    return -1


def compute_score(dataset_name: str, prediction: str, example: dict) -> float:
    """
    Dataset-aware scoring with official TextVQA/VQA normalization.
    """
    if dataset_name == "scienceqa":
        choices = example.get("choices", [])
        ans_idx = example.get("answer", -1)
        if isinstance(ans_idx, str) and ans_idx.isdigit():
            ans_idx = int(ans_idx)
        pred_idx = _scienceqa_pred_to_index(prediction, choices)
        return 1.0 if (isinstance(ans_idx, int) and pred_idx == ans_idx) else 0.0

    if dataset_name == "textvqa":
        answers = example.get("answers", [])
        pred_norm = _normalize_answer(prediction)
        if not pred_norm:
            return 0.0
        if not isinstance(answers, list):
            answers = [answers]
        match_count = 0
        for gt in answers:
            gt_norm = _normalize_answer(gt)
            if gt_norm and gt_norm == pred_norm:
                match_count += 1
        return min(1.0, match_count / 3.0)

    if dataset_name == "gqa":
        gt = example.get("answer", "")
        pred_norm = _normalize_answer(prediction)
        gt_norm = _normalize_answer(gt)
        if not pred_norm or not gt_norm:
            return 0.0
        return 1.0 if pred_norm == gt_norm else 0.0

    return 0.0


def _load_eval_data(ds_cfg: dict, limit: int):
    """
    Load evaluation data. Returns (sequence, n_total) where sequence supports
    __len__ and __getitem__(i) and each item has keys needed by build_prompt/extract_answer.
    For GQA we merge instructions + images from lmms-lab/GQA.
    """
    from datasets import load_dataset

    if "gqa_instructions_config" in ds_cfg:
        # GQA: load instructions and images, merge by imageId
        inst = load_dataset(
            ds_cfg["hf_path"],
            ds_cfg["gqa_instructions_config"],
            split=ds_cfg["gqa_split"],
        )
        imgs = load_dataset(
            ds_cfg["hf_path"],
            ds_cfg["gqa_images_config"],
            split=ds_cfg["gqa_split"],
        )
        image_by_id = {}
        for idx in range(len(imgs)):
            row = imgs[idx]
            image_by_id[row["id"]] = row["image"]
        merged = []
        for idx in range(len(inst)):
            row = inst[idx]
            image_id = row["imageId"]
            if image_id not in image_by_id:
                continue
            merged.append({
                "question": row["question"],
                "answer": row["answer"],
                "image": image_by_id[image_id],
            })
        n_total = len(merged)
        n_eval = min(limit, n_total) if limit > 0 else n_total
        return merged, n_total, n_eval

    ds = load_dataset(ds_cfg["hf_path"], split=ds_cfg["split"])
    if "filter_fn" in ds_cfg:
        ds = ds.filter(ds_cfg["filter_fn"])
    n_total = len(ds)
    n_eval = min(limit, n_total) if limit > 0 else n_total
    return ds, n_total, n_eval


def _is_awq_checkpoint(model_path: str) -> bool:
    """Return True if checkpoint has quantization_config.quant_method == 'awq'."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path) as f:
        config = json.load(f)
    qc = config.get("quantization_config") or {}
    return qc.get("quant_method") == "awq"


def load_model_and_processor_for_eval(
    model_name: str, model_path: str, device_map: str
) -> tuple:
    """
    Load model and processor for evaluation. For AWQ checkpoints, loads base model
    and converts packed INT4 + scale/zero_point to FP16 in memory.
    Returns (model, processor, build_prompt_fn).
    """
    if _is_awq_checkpoint(model_path):
        from safetensors.torch import load_file
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
        base_model_id = config.get("base_model_id") or MODEL_CONFIGS[model_name]["model_id"]
        quantized_layers = config.get("quantized_layers", [])
        group_size = (config.get("quantization_config") or {}).get("group_size", 128)
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        converted = _awq_state_dict_to_fp16(state_dict, quantized_layers, group_size)
        if model_name == "blip2":
            from transformers import BlipForQuestionAnswering, BlipProcessor
            model = BlipForQuestionAnswering.from_pretrained(
                base_model_id, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, device_map=device_map,
            )
            model.load_state_dict(converted, strict=True)
            processor = BlipProcessor.from_pretrained(model_path)
            build_prompt = build_prompt_blip2
        elif model_name == "qwen3vl":
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_id, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, device_map=device_map,
            )
            model.load_state_dict(converted, strict=True)
            processor = AutoProcessor.from_pretrained(model_path)
            build_prompt = build_prompt_qwen3vl
        else:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(
                base_model_id, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, device_map=device_map,
            )
            model.load_state_dict(converted, strict=True)
            processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
            build_prompt = build_prompt_qwen3vl
        if model_name in {"qwen3vl", "llava15"}:
            processor.tokenizer.padding_side = "left"
        return model, processor, build_prompt

    if model_name == "blip2":
        from transformers import BlipForQuestionAnswering, BlipProcessor
        model = BlipForQuestionAnswering.from_pretrained(
            model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map=device_map,
        )
        processor = BlipProcessor.from_pretrained(
            MODEL_CONFIGS["blip2"]["model_id"]
        )
        build_prompt = build_prompt_blip2
    elif model_name == "qwen3vl":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_CONFIGS["qwen3vl"]["model_id"],
        )
        build_prompt = build_prompt_qwen3vl
    else:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_CONFIGS["llava15"]["model_id"],
            use_fast=False,
        )
        build_prompt = build_prompt_qwen3vl
    if model_name in {"qwen3vl", "llava15"}:
        processor.tokenizer.padding_side = "left"  # decoder-only: correct batched generation
    return model, processor, build_prompt


def _run_batch_inference(model, processor, device, model_name: str,
                         batch_items: List[Tuple]) -> List[str]:
    """
    Run batched inference on a list of (example, prompt, img, _) items.
    Returns list of prediction strings, one per item.
    Mirrors preprocessing/run_inference.py: BLIP uses padding + output_scores and decodes
    per sequence; Qwen3-VL uses input lengths from attention_mask to slice generated tokens.
    """
    if not batch_items:
        return []
    images = [item[2] for item in batch_items]
    prompts = [item[1] for item in batch_items]

    inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
    ).to(device, torch.float16)
    if "pixel_values" not in inputs:
        raise RuntimeError(
            f"{model_name} processor did not return pixel_values; "
            "image is not being encoded."
        )

    if model_name == "blip2":
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        num_steps = len(out.scores) if out.scores else 0
        preds = []
        for i in range(out.sequences.shape[0]):
            if num_steps == 0:
                preds.append("")
            else:
                gen_ids = out.sequences[i, -num_steps:]
                preds.append(processor.decode(gen_ids, skip_special_tokens=True).strip())
        return preds
    else:
        input_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
        )
        preds = []
        for i in range(out.sequences.shape[0]):
            gen_ids = out.sequences[i, input_len:]
            preds.append(processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return preds


@torch.no_grad()
def evaluate_single_model(model_name: str, model_path: str, datasets_to_eval: dict, limit: int = 0, batch_size: int = 64):
    """
    Evaluate a single model on all specified datasets with batched inference.
    Returns {dataset_name: {"accuracy": float, "n_samples": int}}, detail_rows.
    """
    flush_gpu()
    print(f"  Loading model from {model_path}...")

    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    model, processor, build_prompt = load_model_and_processor_for_eval(model_name, model_path, device_map)

    model.eval()
    device = next(model.parameters()).device
    print(f"  Model loaded on {device} ({gpu_mem()}), batch_size={batch_size}")

    results = {}
    detail_rows: List[dict] = []

    for ds_name, ds_cfg in datasets_to_eval.items():
        print(f"  Evaluating {ds_name}...")
        prompt_style = ds_cfg["build_prompt_fn"]

        data, n_total, n_eval = _load_eval_data(ds_cfg, limit)
        print(f"    Samples: {n_eval} / {n_total}")

        # Collect valid (example, prompt, img, orig_idx) for batching
        valid_items: List[Tuple] = []
        for i in range(n_eval):
            example = data[i]
            prompt, img = build_prompt(prompt_style, example)
            if prompt is None or img is None:
                continue
            valid_items.append((example, prompt, img, i))

        correct = 0.0
        evaluated = 0
        n_valid = len(valid_items)

        for start in range(0, n_valid, batch_size):
            batch_items = valid_items[start : start + batch_size]
            predictions = _run_batch_inference(model, processor, device, model_name, batch_items)
            for (example, prompt, _img, orig_i), prediction in zip(batch_items, predictions):
                gt = extract_answer(prompt_style, example)
                score = compute_score(prompt_style, prediction, example)
                correct += score
                evaluated += 1

                sample_id = example.get("id", example.get("question_id", f"{ds_name}_{orig_i}"))
                gt_str = " | ".join(str(a) for a in gt) if isinstance(gt, list) else str(gt)
                detail_rows.append({
                    "id": sample_id,
                    "dataset": ds_name,
                    "question": example.get("question", prompt),
                    "ground_truth": gt_str,
                    "predicted": prediction,
                    "correct": 1 if score > 0 else 0,
                })

                if evaluated <= DEBUG_EVAL_SAMPLES:
                    if isinstance(gt, list):
                        gt_dbg = "[" + ", ".join(str(a)[:30] for a in gt[:3]) + ("]" if len(gt) <= 3 else ", ...]")
                    else:
                        gt_dbg = str(gt)[:80] + ("..." if len(str(gt)) > 80 else "")
                    prompt_preview = prompt[:120] + ("..." if len(prompt) > 120 else "")
                    pred_preview = prediction[:120] + ("..." if len(prediction) > 120 else "")
                    print(f"    [sample {evaluated}] in:  {repr(prompt_preview)}")
                    print(f"         out: {repr(pred_preview)}")
                    print(f"         gt:  {gt_dbg}  -> score={score:.0f}")

            if (evaluated % 100 == 0) and evaluated > 0:
                print(f"    Progress: {evaluated}/{n_valid}, acc={correct/evaluated:.3f}")

        acc = correct / evaluated if evaluated > 0 else 0
        results[ds_name] = {
            "accuracy": round(acc * 100, 2),
            "n_samples": evaluated,
            "correct": correct,
        }
        print(f"    {ds_name}: {acc*100:.2f}% ({correct:.0f}/{evaluated})")

    # Cleanup
    del model, processor
    flush_gpu()

    return results, detail_rows


def run_evaluation(quick: bool = False, batch_size: int = 64):
    """Evaluate all baseline + compressed models with batched inference."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = load_log()

    datasets_to_eval = EVAL_DATASETS_LITE if quick else EVAL_DATASETS
    limit = 50 if quick else 0  # 0 = full dataset

    models = ["llava15", "blip2"] if quick else list(MODEL_CONFIGS.keys())
    methods_list = ["wanda"] if quick else METHODS
    combos = {"V": COMPONENT_COMBOS["V"]} if quick else COMPONENT_COMBOS

    # Build job list
    jobs = []
    for model_name in models:
        cfg = MODEL_CONFIGS[model_name]
        # Baseline
        jobs.append({
            "job": jid(model_name, "baseline", "FP16"),
            "model_name": model_name,
            "model_path": cfg["model_id"],
        })
        # Compressed
        for method in methods_list:
            for comp_label in combos:
                cpath = out_path(model_name, method, comp_label)
                jobs.append({
                    "job": jid(model_name, method, comp_label),
                    "model_name": model_name,
                    "model_path": cpath,
                    "requires": cpath,
                })

    total = len(jobs)
    for i, j in enumerate(jobs, 1):
        job_name = j["job"]

        if is_done(log, "eval", job_name):
            print(f"[{i}/{total}] SKIP (done): {job_name}")
            continue

        req = j.get("requires")
        if req and not os.path.exists(req):
            print(f"[{i}/{total}] SKIP (no model): {job_name}")
            continue

        print(f"\n[{i}/{total}] EVAL: {job_name}")
        t0 = time.time()

        results, detail_rows = evaluate_single_model(
            model_name=j["model_name"],
            model_path=j["model_path"],
            datasets_to_eval=datasets_to_eval,
            limit=limit,
            batch_size=batch_size,
        )

        out_dir = os.path.join(RESULTS_DIR, job_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump({"job": job_name, "results": results}, f, indent=2)

        # Per-sample CSV for manual similarity / correctness checks
        details_path = os.path.join(out_dir, "eval_details.csv")
        if detail_rows:
            with open(details_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["id", "dataset", "question", "ground_truth", "predicted", "correct"],
                    quoting=csv.QUOTE_MINIMAL,
                )
                w.writeheader()
                w.writerows(detail_rows)
            print(f"  Saved: {details_path}")

        elapsed = time.time() - t0
        mark_done(log, "eval", job_name, elapsed)
        print(f"  Done in {elapsed:.0f}s")

        flush_gpu()


# =====================================================================
# RESULTS TABLE
# =====================================================================

def collect_results() -> pd.DataFrame:
    rows = []
    rdir = Path(RESULTS_DIR)
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
        pivot = mdf.pivot_table(
            index=["method", "components"], columns="task",
            values="accuracy", aggfunc="first"
        )
        if not pivot.empty:
            print(tabulate(pivot, headers="keys", tablefmt="github", floatfmt=".2f"))

    csv = os.path.join(RESULTS_DIR, "results.csv")
    df.to_csv(csv, index=False)
    print(f"\nSaved: {csv}")


def _template():
    header = ["Model", "Method", "Components"]
    header += list(EVAL_DATASETS.keys())
    rows = []
    for model in ["blip2", "qwen3vl", "llava15"]:
        rows.append([model, "FP16", "—"] + ["—"] * len(EVAL_DATASETS))
        for method in ["wanda", "awq"]:
            for comp in ["V", "V_P"]:
                rows.append([model, method, comp] + ["—"] * len(EVAL_DATASETS))
        rows.append([""] * (3 + len(EVAL_DATASETS)))

    print(tabulate(rows, headers=header, tablefmt="github"))
    print("\nV=Vision, P=Projector/Q-Former (no LLM compression)")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="VLM Compression Pipeline v5")
    parser.add_argument("--stage",
                        choices=["compress", "eval", "table", "all"],
                        required=True)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 model, 1 method, 50 samples, lite benchmarks")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for eval inference (default: 64)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem()})")
    else:
        print("WARNING: No GPU detected.")

    if args.stage in ("compress", "all"):
        run_compression(quick=args.quick)
    if args.stage in ("eval", "all"):
        run_evaluation(quick=args.quick, batch_size=args.batch_size)
    if args.stage in ("table", "all"):
        generate_table()


if __name__ == "__main__":
    main()