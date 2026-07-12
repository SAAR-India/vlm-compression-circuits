"""Multimodal calibration helpers for component-targeted compression."""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Tuple

import torch
from PIL import Image
from tqdm import tqdm

try:
    from compression_methods import (
        ActivationScaleCollector,
        SecondOrderCompressor,
        collect_named_linears,
    )
except ModuleNotFoundError:
    from .compression_methods import (
        ActivationScaleCollector,
        SecondOrderCompressor,
        collect_named_linears,
    )


CalibrationMode = Literal["second_order", "activation_scale"]
CalibrationStats = Dict[str, SecondOrderCompressor | ActivationScaleCollector]


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _move_inputs(inputs, device: torch.device) -> dict:
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=torch.float16)
        else:
            moved[key] = value.to(device=device)
    moved.pop("token_type_ids", None)
    return moved


def _qwen_inputs(processor, image: Image.Image, question: str, device) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return _move_inputs(inputs, device)


def _build_items(limit: int) -> list[Tuple[Image.Image, str]]:
    # Imported lazily so basic compression-method tests do not require datasets.
    try:
        from crosscoder.dataset import VisualCounterfactDataset
    except ModuleNotFoundError:
        from .crosscoder.dataset import VisualCounterfactDataset

    dataset = VisualCounterfactDataset(split="all")
    items: list[Tuple[Image.Image, str]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        for image_key in ("image_original", "image_counterfact"):
            image = sample[image_key]
            if isinstance(image, Image.Image):
                items.append((image.convert("RGB"), sample["question"]))
                if len(items) >= limit:
                    return items
    return items


def _run_calibration_forwards(
    model,
    processor,
    model_name: str,
    items: list[Tuple[Image.Image, str]],
    batch_size: int,
) -> None:
    device = _model_device(model)
    model.eval()

    with torch.no_grad():
        if model_name == "qwen3vl":
            for image, question in tqdm(items, desc="Calibration"):
                inputs = _qwen_inputs(processor, image, question, device)
                model.generate(**inputs, max_new_tokens=1, do_sample=False)
            return

        for start in tqdm(range(0, len(items), batch_size), desc="Calibration"):
            batch = items[start : start + batch_size]
            images = [image for image, _ in batch]
            questions = [question for _, question in batch]
            if model_name == "blip2":
                texts = questions
            elif model_name == "llava15":
                texts = [
                    f"USER: <image>\n{question}\nASSISTANT:"
                    for question in questions
                ]
            else:
                raise ValueError(f"Unknown model for calibration: {model_name}")

            inputs = processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
            )
            inputs = _move_inputs(inputs, device)
            model.generate(**inputs, max_new_tokens=1, do_sample=False)


def collect_calibration_stats(
    model,
    processor,
    model_name: str,
    module_paths: Iterable[str],
    mode: CalibrationMode,
    num_samples: int = 128,
    batch_size: int = 4,
) -> CalibrationStats:
    """Collect Hessians or activation maxima for every targeted linear layer."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    layers = collect_named_linears(model, module_paths)
    if not layers:
        raise ValueError(f"No Linear layers found below {list(module_paths)!r}")

    if mode == "second_order":
        stats: CalibrationStats = {
            name: SecondOrderCompressor(layer) for name, layer in layers.items()
        }
    elif mode == "activation_scale":
        stats = {
            name: ActivationScaleCollector(layer.in_features)
            for name, layer in layers.items()
        }
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")

    handles = []
    for name, layer in layers.items():
        collector = stats[name]

        def capture(_module, inputs, _output, collector=collector):
            if inputs and torch.is_tensor(inputs[0]):
                collector.add_batch(inputs[0])

        handles.append(layer.register_forward_hook(capture))

    try:
        items = _build_items(num_samples)
        if not items:
            raise RuntimeError(
                "Visual-Counterfact calibration data contains no usable images"
            )
        _run_calibration_forwards(
            model,
            processor,
            model_name,
            items,
            batch_size=batch_size,
        )
    finally:
        for handle in handles:
            handle.remove()

    return stats
