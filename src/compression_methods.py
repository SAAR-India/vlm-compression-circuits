"""Core compression algorithms used by the component-wise VLM pipeline.

The SparseGPT and GPTQ implementations are adapted from the official
IST-DASLab repositories:
  https://github.com/IST-DASLab/sparsegpt (Apache-2.0)
  https://github.com/IST-DASLab/gptq (Apache-2.0)

The SmoothQuant equations and fake-quantization behavior are adapted from:
  https://github.com/mit-han-lab/smoothquant (MIT)

The adaptations make the algorithms operate on arbitrary ``nn.Linear``
submodules. They intentionally simulate quantized arithmetic in PyTorch;
they do not provide accelerated sparse/INT kernels.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def quantize_dequantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
) -> torch.Tensor:
    """Quantize and immediately dequantize a tensor."""
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class WeightQuantizer:
    """Small per-channel quantizer matching the original GPTQ helper."""

    def __init__(self, bits: int = 4, symmetric: bool = False) -> None:
        self.bits = bits
        self.symmetric = symmetric
        self.maxq = 2**bits - 1
        self.scale: torch.Tensor | None = None
        self.zero: torch.Tensor | None = None

    def find_params(self, weight: torch.Tensor) -> None:
        rows = weight.flatten(1)
        zeros = torch.zeros(rows.shape[0], device=rows.device)
        xmin = torch.minimum(rows.min(dim=1).values, zeros)
        xmax = torch.maximum(rows.max(dim=1).values, zeros)

        if self.symmetric:
            xmax = torch.maximum(xmin.abs(), xmax)
            xmin = -xmax

        dead = (xmin == 0) & (xmax == 0)
        xmin[dead] = -1
        xmax[dead] = 1

        scale = ((xmax - xmin) / self.maxq).clamp_min(1e-8)
        if self.symmetric:
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)

        self.scale = scale.unsqueeze(1)
        self.zero = zero.unsqueeze(1)

    def apply(self, weight: torch.Tensor) -> torch.Tensor:
        if self.scale is None or self.zero is None:
            self.find_params(weight)
        return quantize_dequantize(weight, self.scale, self.zero, self.maxq)


class SecondOrderCompressor:
    """Accumulate an input Hessian approximation for one linear layer."""

    def __init__(self, layer: nn.Linear) -> None:
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(layer)!r}")
        self.layer = layer
        self.device = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.H = torch.zeros(
            (self.columns, self.columns),
            dtype=torch.float32,
            device=self.device,
        )
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        batch_samples = inp.shape[0]
        if inp.ndim == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        else:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t().float()

        total = self.nsamples + batch_samples
        if total == 0:
            return
        self.H.mul_(self.nsamples / total)
        self.nsamples = total
        inp.mul_(math.sqrt(2.0 / self.nsamples))
        self.H.addmm_(inp, inp.t())

    def _prepare(
        self, percdamp: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.layer.weight.detach().float().clone()
        H = self.H
        self.H = torch.empty(0, device=self.device)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        weight[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp.clamp_min(1e-8)
        try:
            chol = torch.linalg.cholesky(H)
        except torch.linalg.LinAlgError:
            # Calibration can be rank-deficient for tiny smoke datasets.
            H[diag, diag] += 1e-4
            chol = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(chol)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
        return weight, Hinv

    @torch.no_grad()
    def prune_sparsegpt(
        self,
        sparsity: float,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        """Apply the official SparseGPT blockwise reconstruction update."""
        if not 0 <= sparsity < 1:
            raise ValueError("sparsity must be in [0, 1)")

        weight, Hinv = self._prepare(percdamp)
        mask: torch.Tensor | None = None

        for start in range(0, self.columns, blocksize):
            stop = min(start + blocksize, self.columns)
            W1 = weight[:, start:stop].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[start:stop, start:stop]

            if mask is None:
                importance = W1.square() / torch.diag(Hinv1).reshape(1, -1).square()
                prune_count = int(importance.numel() * sparsity)
                if prune_count == 0:
                    mask1 = torch.zeros_like(importance, dtype=torch.bool)
                else:
                    threshold = torch.kthvalue(
                        importance.flatten(), min(prune_count, importance.numel())
                    ).values
                    mask1 = importance <= threshold
            else:
                mask1 = mask[:, start:stop]

            for column in range(stop - start):
                w = W1[:, column]
                diagonal = Hinv1[column, column]
                q = w.clone()
                q[mask1[:, column]] = 0
                Q1[:, column] = q
                err = (w - q) / diagonal
                W1[:, column:] -= err.unsqueeze(1) @ Hinv1[column, column:].unsqueeze(0)
                Err1[:, column] = err

            weight[:, start:stop] = Q1
            weight[:, stop:] -= Err1 @ Hinv[start:stop, stop:]

        self.layer.weight.copy_(weight.to(self.layer.weight.dtype))
        _synchronize(self.device)

    @torch.no_grad()
    def quantize_gptq(
        self,
        bits: int = 4,
        group_size: int = 128,
        blocksize: int = 128,
        percdamp: float = 0.01,
        symmetric: bool = False,
    ) -> None:
        """Apply GPTQ and retain fake-quantized weights in the original dtype."""
        weight, Hinv = self._prepare(percdamp)
        quantizer = WeightQuantizer(bits=bits, symmetric=symmetric)
        quantized = torch.zeros_like(weight)

        for start in range(0, self.columns, blocksize):
            stop = min(start + blocksize, self.columns)
            W1 = weight[:, start:stop].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[start:stop, start:stop]

            for column in range(stop - start):
                global_column = start + column
                if group_size <= 0 or global_column % group_size == 0:
                    group_stop = (
                        self.columns
                        if group_size <= 0
                        else min(global_column + group_size, self.columns)
                    )
                    quantizer.find_params(weight[:, global_column:group_stop])

                w = W1[:, column]
                diagonal = Hinv1[column, column]
                q = quantizer.apply(w.unsqueeze(1)).flatten()
                Q1[:, column] = q
                err = (w - q) / diagonal
                W1[:, column:] -= err.unsqueeze(1) @ Hinv1[column, column:].unsqueeze(0)
                Err1[:, column] = err

            quantized[:, start:stop] = Q1
            weight[:, stop:] -= Err1 @ Hinv[start:stop, stop:]

        self.layer.weight.copy_(quantized.to(self.layer.weight.dtype))
        _synchronize(self.device)

    def free(self) -> None:
        self.H = torch.empty(0, device=self.device)


class ActivationScaleCollector:
    """Collect per-input-channel absolute maxima for SmoothQuant."""

    def __init__(self, in_features: int) -> None:
        self.amax = torch.zeros(in_features, dtype=torch.float32)

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        values = inp.detach().reshape(-1, inp.shape[-1]).abs().amax(dim=0).cpu()
        self.amax = torch.maximum(self.amax, values)


@torch.no_grad()
def smoothquant_weight(
    layer: nn.Linear,
    activation_amax: torch.Tensor,
    alpha: float = 0.5,
    weight_bits: int = 8,
) -> torch.Tensor:
    """Smooth and fake-quantize one weight; return the runtime input scale."""
    weight = layer.weight.detach()
    act_scales = activation_amax.to(device=weight.device, dtype=torch.float32)
    weight_scales = weight.float().abs().amax(dim=0).clamp_min(1e-5)
    scales = (
        act_scales.clamp_min(1e-5).pow(alpha)
        / weight_scales.pow(1.0 - alpha)
    ).clamp_min(1e-5)

    smoothed = weight.float() * scales.unsqueeze(0)
    qmax = 2 ** (weight_bits - 1) - 1
    row_scale = smoothed.abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / qmax
    smoothed = torch.clamp(torch.round(smoothed / row_scale), -qmax, qmax) * row_scale
    layer.weight.copy_(smoothed.to(layer.weight.dtype))
    return scales.cpu()


class SmoothQuantLinear(nn.Module):
    """W8A8 fake-quantized linear used for SmoothQuant accuracy experiments."""

    def __init__(
        self,
        linear: nn.Linear,
        input_scale: torch.Tensor,
        activation_bits: int = 8,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.activation_bits = activation_bits
        self.register_buffer(
            "smoothquant_input_scale",
            input_scale.to(device=linear.weight.device, dtype=linear.weight.dtype),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        scaled = inp / self.smoothquant_input_scale
        qmax = 2 ** (self.activation_bits - 1) - 1
        act_scale = scaled.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / qmax
        quantized = (
            torch.clamp(torch.round(scaled / act_scale), -qmax, qmax) * act_scale
        )
        return F.linear(quantized, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"W8A{self.activation_bits}, bias={self.bias is not None}"
        )


def _parent_and_attr(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def install_smoothquant_wrappers(
    model: nn.Module,
    scales: Mapping[str, torch.Tensor],
    activation_bits: int = 8,
) -> nn.Module:
    """Replace named linear layers with runtime W8A8 fake-quant wrappers."""
    for name, scale in scales.items():
        parent, attr = _parent_and_attr(model, name)
        layer = getattr(parent, attr)
        if isinstance(layer, SmoothQuantLinear):
            continue
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"SmoothQuant target {name!r} is {type(layer)!r}, not Linear")
        setattr(
            parent,
            attr,
            SmoothQuantLinear(layer, scale, activation_bits=activation_bits),
        )
    return model


def collect_named_linears(
    model: nn.Module, module_paths: Iterable[str]
) -> Dict[str, nn.Linear]:
    """Return unique fully-qualified Linear layers below selected subtrees."""
    found: Dict[str, nn.Linear] = {}
    for path in module_paths:
        submodule = model
        for part in path.split("."):
            submodule = getattr(submodule, part)
        for child_name, child in submodule.named_modules():
            if isinstance(child, nn.Linear):
                full_name = f"{path}.{child_name}".rstrip(".")
                found.setdefault(full_name, child)
    return found
