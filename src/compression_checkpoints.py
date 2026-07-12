"""Shared checkpoint conversion and runtime restoration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import torch
from safetensors.torch import load_file, save_file

try:
    from compression_methods import install_smoothquant_wrappers
except ModuleNotFoundError:  # Package execution: python -m src.crosscoder.main
    from .compression_methods import install_smoothquant_wrappers


def load_checkpoint_metadata(checkpoint_path: str | Path) -> tuple[dict, dict]:
    path = Path(checkpoint_path)
    config = {}
    meta = {}
    if (path / "config.json").exists():
        with open(path / "config.json", encoding="utf-8") as handle:
            config = json.load(handle)
    if (path / "meta.json").exists():
        with open(path / "meta.json", encoding="utf-8") as handle:
            meta = json.load(handle)
    return config, meta


def checkpoint_method(checkpoint_path: str | Path) -> str | None:
    config, meta = load_checkpoint_metadata(checkpoint_path)
    if meta.get("method"):
        return meta["method"]
    quantization = config.get("quantization_config") or {}
    if quantization.get("quant_method"):
        return quantization["quant_method"]
    compression = config.get("compression_config") or {}
    return compression.get("method")


def load_state_dict_from_checkpoint(
    checkpoint_path: str | Path,
) -> Dict[str, torch.Tensor]:
    path = Path(checkpoint_path)
    single_file = path / "model.safetensors"
    index_file = path / "model.safetensors.index.json"
    if single_file.exists():
        return load_file(str(single_file))
    if index_file.exists():
        with open(index_file, encoding="utf-8") as handle:
            index = json.load(handle)
        state_dict: Dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(index.get("weight_map", {}).values())):
            state_dict.update(load_file(str(path / shard_name)))
        return state_dict
    raise FileNotFoundError(f"No safetensors checkpoint found under {path}")


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    out_features, packed_features = packed.shape
    unpacked = torch.zeros(
        (out_features, packed_features * 8),
        dtype=torch.int32,
        device=packed.device,
    )
    for offset in range(8):
        unpacked[:, offset::8] = (packed >> (offset * 4)) & 0xF
    return unpacked


def awq_state_dict_to_fp16(
    state_dict: Mapping[str, torch.Tensor],
    quantized_layers: list[str],
    group_size: int,
) -> Dict[str, torch.Tensor]:
    """Convert this repository's packed AWQ checkpoint to normal FP16 weights."""
    output = {
        key: value.clone()
        for key, value in state_dict.items()
        if not key.endswith((".qweight", ".scales", ".zeros"))
    }
    for layer_name in quantized_layers:
        qweight = state_dict.get(f"{layer_name}.qweight")
        if qweight is None:
            continue
        scales = state_dict[f"{layer_name}.scales"]
        zeros = state_dict[f"{layer_name}.zeros"]
        if zeros.ndim == 3:
            zeros = zeros.squeeze(-1)
        unpacked = _unpack_int4(qweight)
        actual_group_size = unpacked.shape[1] // scales.shape[1]
        layer_group_size = group_size
        if layer_group_size > 0 and actual_group_size != layer_group_size:
            # Layers that were not divisible by the configured group size used
            # their full input width as one group when they were saved.
            layer_group_size = actual_group_size
        scales_expanded = scales.repeat_interleave(layer_group_size, dim=1)
        zeros_expanded = zeros.repeat_interleave(layer_group_size, dim=1)
        output[f"{layer_name}.weight"] = (
            (unpacked.float() - zeros_expanded.float()) * scales_expanded
        ).to(torch.float16)
    return output


def converted_state_dict(
    checkpoint_path: str | Path,
    config: dict | None = None,
) -> Dict[str, torch.Tensor]:
    config = config or load_checkpoint_metadata(checkpoint_path)[0]
    state_dict = load_state_dict_from_checkpoint(checkpoint_path)
    quantization = config.get("quantization_config") or {}
    if quantization.get("quant_method") == "awq":
        return awq_state_dict_to_fp16(
            state_dict,
            config.get("quantized_layers", []),
            quantization.get("group_size", 128),
        )
    return state_dict


def save_smoothquant_scales(
    checkpoint_path: str | Path,
    scales: Mapping[str, torch.Tensor],
) -> None:
    path = Path(checkpoint_path) / "smoothquant_scales.safetensors"
    tensors = {
        name: scale.detach().float().cpu().contiguous()
        for name, scale in scales.items()
    }
    save_file(tensors, str(path))


def load_smoothquant_scales(
    checkpoint_path: str | Path,
) -> Dict[str, torch.Tensor]:
    path = Path(checkpoint_path) / "smoothquant_scales.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Missing SmoothQuant scales: {path}")
    # Clone so Windows can release safetensors' memory map immediately.
    return {name: value.clone() for name, value in load_file(str(path)).items()}


def restore_compression_runtime(
    model,
    checkpoint_path: str | Path,
    method: str | None = None,
    config: dict | None = None,
):
    """Install runtime modules required by a loaded compressed checkpoint."""
    config = config or load_checkpoint_metadata(checkpoint_path)[0]
    method = method or checkpoint_method(checkpoint_path)
    if method == "smoothquant":
        compression = config.get("compression_config") or {}
        scales = load_smoothquant_scales(checkpoint_path)
        install_smoothquant_wrappers(
            model,
            scales,
            activation_bits=compression.get("activation_bits", 8),
        )
    return model
