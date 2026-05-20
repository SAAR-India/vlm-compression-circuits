#!/usr/bin/env python
"""
This script runs vision + projector compression (Wanda, AWQ/INT4) for BLIP-VQA, Qwen3-VL-2B, 
and LLaVA-1.5-7B. Q-VLM can be found in the src/qvlm_compression/ directory.

Usage:
    python src/run_compression_eval.py --stage compress
    python src/run_compression_eval.py --stage eval [--batch_size 64]
    python src/run_compression_eval.py --stage table
    python src/run_compression_eval.py --stage all [--quick]
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

import compression_utils
import compression_configs

# Model IDs from preprocessing/config.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Compression: WANDA
def apply_wanda(model, model_name: str, components: List[str], config: dict):
    """
    Wanda-style pruning: zero out smallest |W| entries per layer.
    Applied only to Linear layers within targeted components.

    For full activation-aware Wanda, you'd collect input activation norms
    via forward hooks with calibration data. This implementation uses
    magnitude as proxy, which is standard for initial experiments.
    """
    sparsity = config["sparsity_ratio"]
    paths = compression_utils.get_module_paths(model_name, components)

    for path in paths:
        submodule = compression_utils.get_submodule(model, path)
        n_layers = 0

        for _, child in submodule.named_modules():
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


def build_awq_state_dict(model, model_name: str, components: List[str], 
                         config: dict) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Build a state dict with AWQ-packed INT4 weights and scale/zero_point for
    targeted Linear layers. Other weights are copied unchanged. Returns
    (state_dict, quantized_layer_names) for saving and for load-time conversion.
    """
    bits = config["w_bit"]
    group_size = config["q_group_size"]
    paths = compression_utils.get_module_paths(model_name, components)
    state_dict = dict(model.state_dict())
    quantized_layers: List[str] = []

    for path in paths:
        submodule = compression_utils.get_submodule(model, path)
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


def _awq_state_dict_to_fp16(state_dict: Dict[str, torch.Tensor],
                            quantized_layers: List[str], group_size: int) -> Dict[str, torch.Tensor]:
    """
    Convert state dict from AWQ format (qweight, scales, zeros) to FP16 .weight
    for loading into a standard model. Expands scale/zero per group to full dims.
    """
    out = {k: v.clone() for k, v in state_dict.items() if not k.endswith(".qweight") 
           and not k.endswith(".scales") 
           and not k.endswith(".zeros")}
    
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


def save_awq_checkpoint(opath: str, model, model_name: str, state_dict: Dict[str, torch.Tensor],
                        quantized_layers: List[str], processor, components: List[str], comp_label: str,
                        paths: List[str], method_config: dict) -> None:
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
    config_dict["base_model_id"] = compression_configs.MODEL_CONFIGS[model_name]["model_id"]
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

def main():
    parser = argparse.ArgumentParser(description="VLM Compression Pipeline v5")
    parser.add_argument("--stage", choices=["compress", "eval", "table", "all"], required=True)
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 model, 1 method, 50 samples, lite benchmarks")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for eval inference (default: 64)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({compression_utils.gpu_mem()})")
    else:
        print("WARNING: No GPU detected.")

    if args.stage in ("compress", "all"):
        compression_utils.run_compression(quick=args.quick)
    if args.stage in ("eval", "all"):
        compression_utils.run_evaluation(quick=args.quick, batch_size=args.batch_size)
    if args.stage in ("table", "all"):
        compression_utils.generate_table()


if __name__ == "__main__":
    main()