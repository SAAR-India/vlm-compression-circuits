import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BlipForQuestionAnswering,
    BlipProcessor,
    LlavaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from . import config
from .dataset import VisualCounterfactDataset
from .utils import flush_gpu, get_device, get_compressed_model_path, set_seed

# Qwen3-VL: use apply_chat_template(..., tokenize=True) so image tokens are correctly inserted (aligned with preprocessing/blip_visual_counterfact.py).
_DTYPE = torch.float16


def _load_state_dict_from_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load model state dict from checkpoint. Supports:
    - Single file: checkpoint_path / "model.safetensors"
    - Sharded: checkpoint_path / "model.safetensors.index.json" with weight_map pointing to
      shard files (e.g. model-00001-of-00002.safetensors). All shards are loaded and merged.
    """
    single_file = checkpoint_path / "model.safetensors"
    index_file = checkpoint_path / "model.safetensors.index.json"
    if single_file.exists():
        return load_file(single_file)
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        state_dict = {}
        for shard_name in shard_files:
            shard_path = checkpoint_path / shard_name
            state_dict.update(load_file(shard_path))
        return state_dict
    raise FileNotFoundError(
        f"No safetensors found under {checkpoint_path}: "
        "expected either 'model.safetensors' or 'model.safetensors.index.json' with shard files."
    )


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    out_f, in_packed = packed.shape
    w_q = torch.zeros((out_f, in_packed * 8), dtype=torch.int32, device=packed.device)
    for k in range(8):
        w_q[:, k::8] = (packed >> (k * 4)) & 0xF
    return w_q


def _awq_state_dict_to_fp16(
    state_dict: Dict[str, torch.Tensor],
    quantized_layers: List[str],
    group_size: int,
) -> Dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in state_dict.items() 
           if not k.endswith(".qweight") and not k.endswith(".scales") and not k.endswith(".zeros")}
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
        if zeros.ndim == 3:
            zeros = zeros.squeeze(-1)
        n_groups = scales.shape[1]
        w_q = _unpack_int4(packed)
        scales_exp = scales.repeat_interleave(group_size, dim=1)
        zeros_exp = zeros.repeat_interleave(group_size, dim=1)
        w_fp = (w_q.float() - zeros_exp.float()) * scales_exp
        out[f"{full_key}.weight"] = w_fp.to(torch.float16)
    return out


def _qwen_inputs_for_sample(processor, image: Image.Image, question: str, device):
    """Build model inputs for one (image, question) using apply_chat_template(tokenize=True)."""
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    for k, v in inputs.items():
        if hasattr(v, "to"):
            if v.is_floating_point():
                inputs[k] = v.to(device=device, dtype=_DTYPE)
            else:
                inputs[k] = v.to(device)
    return inputs


def load_uncompressed_model(model_name: str) -> Tuple:
    device = get_device()
    if model_name == "blip2":
        model = BlipForQuestionAnswering.from_pretrained(
            config.BLIP_VQA_MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = BlipProcessor.from_pretrained(config.BLIP_VQA_MODEL_ID)
    elif model_name == "qwen3vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.QWEN3VL_2B_MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(config.QWEN3VL_2B_MODEL_ID)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
    elif model_name == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(
            config.LLAVA_V1_5_7B_MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(config.LLAVA_V1_5_7B_MODEL_ID)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.eval()
    return model, processor


def load_compressed_model(model_name: str, method: str, component: str) -> Tuple:
    device = get_device()
    checkpoint_path = get_compressed_model_path(model_name, method, component)
    
    with open(checkpoint_path / "meta.json") as f:
        meta = json.load(f)
    with open(checkpoint_path / "config.json") as f:
        model_config = json.load(f)

    state_dict = _load_state_dict_from_checkpoint(checkpoint_path)
    
    if method == "awq":
        quantized_layers = model_config.get("quantized_layers", [])
        group_size = model_config.get("quantization_config", {}).get("group_size", 128)
        state_dict = _awq_state_dict_to_fp16(state_dict, quantized_layers, group_size)
    
    if model_name == "blip2":
        model = BlipForQuestionAnswering.from_pretrained(
            config.BLIP_VQA_MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        processor = BlipProcessor.from_pretrained(config.BLIP_VQA_MODEL_ID)
    elif model_name == "qwen3vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.QWEN3VL_2B_MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(config.QWEN3VL_2B_MODEL_ID)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
    elif model_name == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(
            config.LLAVA_V1_5_7B_MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(config.LLAVA_V1_5_7B_MODEL_ID)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model, processor


def get_submodule(model, dotted_path: str):
    m = model
    for attr in dotted_path.split("."):
        m = getattr(m, attr)
    return m


class ActivationExtractor:
    def __init__(self, model, model_name: str, component: str):
        self.model = model
        self.model_name = model_name
        self.component = component
        self.activations = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        if self.component in ["V", "V_P"]:
            self._register_vision_hook()
        if self.component in ["P", "V_P"]:
            self._register_projector_hook()
    
    def _register_vision_hook(self):
        hook_path = config.VISION_HOOK_PATHS[self.model_name]
        layers = get_submodule(self.model, hook_path)
        last_layer = layers[-1]
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations["vision"] = output[0].detach()
            else:
                self.activations["vision"] = output.detach()
        
        handle = last_layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
    
    def _register_projector_hook(self):
        if self.model_name == "blip2":
            hook_path = config.PROJECTOR_HOOK_PATHS[self.model_name]
            layers = get_submodule(self.model, hook_path)
            target_layers = [layers[i] for i in config.BLIP_CROSS_ATTENTION_LAYERS if i < len(layers)]
            
            self.activations["projector_layers"] = []
            
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        act = output[0].detach()
                    else:
                        act = output.detach()
                    if len(self.activations["projector_layers"]) <= layer_idx:
                        self.activations["projector_layers"].append(act)
                    else:
                        self.activations["projector_layers"][layer_idx] = act
                return hook_fn
            
            for i, layer in enumerate(target_layers):
                handle = layer.register_forward_hook(make_hook(i))
                self.hooks.append(handle)
        else:
            hook_path = config.PROJECTOR_HOOK_PATHS[self.model_name]
            projector = get_submodule(self.model, hook_path)
            
            def hook_fn(module, input, output):
                self.activations["projector"] = output.detach()
            
            handle = projector.register_forward_hook(hook_fn)
            self.hooks.append(handle)
    
    def extract(self, token_type: str = "cls") -> Dict[str, torch.Tensor]:
        result = {}
        
        if "vision" in self.activations:
            vision_act = self.activations["vision"]
            if vision_act.dim() == 3:
                if token_type == "cls":
                    result["vision"] = vision_act[:, 0, :]
                else:
                    result["vision"] = vision_act[:, 1:, :].mean(dim=1)
            else:
                result["vision"] = vision_act
        
        if "projector" in self.activations:
            proj_act = self.activations["projector"]
            if proj_act.dim() == 3:
                result["projector"] = proj_act.mean(dim=1)
            else:
                result["projector"] = proj_act
        
        if "projector_layers" in self.activations and len(self.activations["projector_layers"]) > 0:
            stacked = torch.stack(self.activations["projector_layers"], dim=0)
            mean_act = stacked.mean(dim=0)
            if mean_act.dim() == 3:
                result["projector"] = mean_act.mean(dim=1)
            else:
                result["projector"] = mean_act
        
        return result
    
    def clear(self):
        self.activations = {}
        if self.model_name == "blip2" and self.component in ["P", "V_P"]:
            self.activations["projector_layers"] = []
    
    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def compute_awq_normalization_stats(
    activations_list: List[torch.Tensor],
    n_samples: int = config.AWQ_CALIBRATION_SAMPLES
) -> Tuple[torch.Tensor, torch.Tensor]:
    subset = activations_list[:n_samples]
    stacked = torch.stack(subset, dim=0)
    mu = stacked.mean(dim=0)
    sigma = stacked.std(dim=0)
    return mu, sigma


def normalize_activations(
    activations: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    return (activations - mu) / (sigma + 1e-8)


def extract_activations_for_config(
    model_name: str,
    method: str,
    component: str,
    token_type: str,
) -> Dict:
    set_seed()
    device = get_device()
    
    dataset = VisualCounterfactDataset(split="all")
    
    print(f"Loading uncompressed {model_name} model...")
    model_u, processor_u = load_uncompressed_model(model_name)
    extractor_u = ActivationExtractor(model_u, model_name, component)
    
    print(f"Loading compressed {model_name} ({method}, {component}) model...")
    model_c, processor_c = load_compressed_model(model_name, method, component)
    extractor_c = ActivationExtractor(model_c, model_name, component)
    
    activations_u_list = []
    activations_c_list = []
    sample_ids = []
    image_types = []
    splits = []
    
    comp_key = "projector" if component == "P" else "vision"
    batch_size = config.EXTRACT_BATCH_SIZE
    
    items = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        for img_type in ["original", "counterfact"]:
            img_key = f"image_{img_type}"
            image = sample[img_key]
            if not isinstance(image, Image.Image):
                continue
            image = image.convert("RGB")
            items.append({
                "image": image,
                "question": sample["question"],
                "sample_id": sample["sample_id"],
                "image_type": img_type,
                "split": sample["split"],
            })
    
    print("Extracting activations (batched)...")
    for i in tqdm(range(0, len(items), batch_size), desc="Processing batches"):
        batch_items = items[i:i + batch_size]
        if model_name == "blip2":
            images = [it["image"] for it in batch_items]
            questions = [it["question"] for it in batch_items]
            inputs_u = processor_u(images=images, text=questions, return_tensors="pt", padding=True)
            inputs_c = processor_c(images=images, text=questions, return_tensors="pt", padding=True)
            inputs_u = {k: v.to(device) for k, v in inputs_u.items()}
            inputs_c = {k: v.to(device) for k, v in inputs_c.items()}
            extractor_u.clear()
            extractor_c.clear()
            with torch.no_grad():
                _ = model_u.generate(**inputs_u, max_new_tokens=1)
                _ = model_c.generate(**inputs_c, max_new_tokens=1)
            acts_u = extractor_u.extract(token_type)
            acts_c = extractor_c.extract(token_type)
            if comp_key in acts_u and comp_key in acts_c:
                act_u = acts_u[comp_key].cpu()
                act_c = acts_c[comp_key].cpu()
                for j, it in enumerate(batch_items):
                    if j < act_u.shape[0]:
                        activations_u_list.append(act_u[j])
                        activations_c_list.append(act_c[j])
                        sample_ids.append(it["sample_id"])
                        image_types.append(it["image_type"])
                        splits.append(it["split"])
        else:
            # Qwen3-VL: apply_chat_template(..., tokenize=True) per sample (aligned with blip_visual_counterfact.py)
            for it in batch_items:
                inputs_u = _qwen_inputs_for_sample(processor_u, it["image"], it["question"], device)
                inputs_c = _qwen_inputs_for_sample(processor_c, it["image"], it["question"], device)
                extractor_u.clear()
                extractor_c.clear()
                with torch.no_grad():
                    _ = model_u.generate(**inputs_u, max_new_tokens=1)
                    _ = model_c.generate(**inputs_c, max_new_tokens=1)
                acts_u = extractor_u.extract(token_type)
                acts_c = extractor_c.extract(token_type)
                if comp_key in acts_u and comp_key in acts_c:
                    act_u = acts_u[comp_key].cpu().squeeze(0)
                    act_c = acts_c[comp_key].cpu().squeeze(0)
                    # Qwen3-VL vision has variable seq length (image-dependent); pool to (hidden_size,) for stacking
                    if act_u.dim() == 2:
                        act_u = act_u.mean(dim=0)
                        act_c = act_c.mean(dim=0)
                    activations_u_list.append(act_u)
                    activations_c_list.append(act_c)
                    sample_ids.append(it["sample_id"])
                    image_types.append(it["image_type"])
                    splits.append(it["split"])
    
    extractor_u.remove_hooks()
    extractor_c.remove_hooks()
    
    activations_u = torch.stack(activations_u_list, dim=0)
    activations_c = torch.stack(activations_c_list, dim=0)
    
    if method == "awq":
        print("Computing AWQ normalization statistics...")
        mu_c, sigma_c = compute_awq_normalization_stats(activations_c_list)
        activations_c = normalize_activations(activations_c, mu_c, sigma_c)
        awq_stats = {"mu": mu_c, "sigma": sigma_c}
    else:
        awq_stats = None
    
    del model_u, model_c, processor_u, processor_c
    flush_gpu()
    
    result = {
        "activations_u": activations_u,
        "activations_c": activations_c,
        "sample_ids": sample_ids,
        "image_types": image_types,
        "splits": splits,
        "model_name": model_name,
        "method": method,
        "component": component,
        "token_type": token_type,
    }
    
    if awq_stats is not None:
        result["awq_stats"] = awq_stats
    
    return result


def extract_activations_for_vp_config(
    model_name: str,
    method: str,
) -> Tuple[Dict, Dict]:
    set_seed()
    device = get_device()
    
    dataset = VisualCounterfactDataset(split="all")
    
    print(f"Loading uncompressed {model_name} model...")
    model_u, processor_u = load_uncompressed_model(model_name)
    extractor_u = ActivationExtractor(model_u, model_name, "V_P")
    
    print(f"Loading compressed {model_name} ({method}, V_P) model...")
    model_c, processor_c = load_compressed_model(model_name, method, "V_P")
    extractor_c = ActivationExtractor(model_c, model_name, "V_P")
    
    v_activations_u_list = []
    v_activations_c_list = []
    p_activations_u_list = []
    p_activations_c_list = []
    sample_ids = []
    image_types = []
    splits = []
    
    batch_size = config.EXTRACT_BATCH_SIZE
    items = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        for img_type in ["original", "counterfact"]:
            img_key = f"image_{img_type}"
            image = sample[img_key]
            if not isinstance(image, Image.Image):
                continue
            image = image.convert("RGB")
            items.append({
                "image": image,
                "question": sample["question"],
                "sample_id": sample["sample_id"],
                "image_type": img_type,
                "split": sample["split"],
            })
    
    print("Extracting activations for V+P (batched)...")
    for i in tqdm(range(0, len(items), batch_size), desc="Processing batches"):
        batch_items = items[i:i + batch_size]
        if model_name == "blip2":
            images = [it["image"] for it in batch_items]
            questions = [it["question"] for it in batch_items]
            inputs_u = processor_u(images=images, text=questions, return_tensors="pt", padding=True)
            inputs_c = processor_c(images=images, text=questions, return_tensors="pt", padding=True)
            inputs_u = {k: v.to(device) for k, v in inputs_u.items()}
            inputs_c = {k: v.to(device) for k, v in inputs_c.items()}
            extractor_u.clear()
            extractor_c.clear()
            with torch.no_grad():
                _ = model_u.generate(**inputs_u, max_new_tokens=1)
                _ = model_c.generate(**inputs_c, max_new_tokens=1)
            acts_u = extractor_u.extract("cls")
            acts_c = extractor_c.extract("cls")
            if "vision" in acts_u and "vision" in acts_c and "projector" in acts_u and "projector" in acts_c:
                v_u, v_c = acts_u["vision"].cpu(), acts_c["vision"].cpu()
                p_u, p_c = acts_u["projector"].cpu(), acts_c["projector"].cpu()
                for j, it in enumerate(batch_items):
                    if j < v_u.shape[0]:
                        v_activations_u_list.append(v_u[j])
                        v_activations_c_list.append(v_c[j])
                        p_activations_u_list.append(p_u[j])
                        p_activations_c_list.append(p_c[j])
                        sample_ids.append(it["sample_id"])
                        image_types.append(it["image_type"])
                        splits.append(it["split"])
        else:
            # Qwen3-VL: apply_chat_template(..., tokenize=True) per sample (aligned with blip_visual_counterfact.py)
            for it in batch_items:
                inputs_u = _qwen_inputs_for_sample(processor_u, it["image"], it["question"], device)
                inputs_c = _qwen_inputs_for_sample(processor_c, it["image"], it["question"], device)
                extractor_u.clear()
                extractor_c.clear()
                with torch.no_grad():
                    _ = model_u.generate(**inputs_u, max_new_tokens=1)
                    _ = model_c.generate(**inputs_c, max_new_tokens=1)
                acts_u = extractor_u.extract("cls")
                acts_c = extractor_c.extract("cls")
                if "vision" in acts_u and "vision" in acts_c and "projector" in acts_u and "projector" in acts_c:
                    # Vision activations: pool variable-length sequences to a single vector
                    v_u = acts_u["vision"].cpu().squeeze(0)
                    v_c = acts_c["vision"].cpu().squeeze(0)
                    if v_u.dim() == 2:
                        v_u = v_u.mean(dim=0)
                        v_c = v_c.mean(dim=0)
                    v_activations_u_list.append(v_u)
                    v_activations_c_list.append(v_c)

                    # Projector activations (Qwen3-VL merger) can also have variable sequence length.
                    # Pool across the sequence dimension so all samples have shape [hidden_size].
                    p_u = acts_u["projector"].cpu().squeeze(0)
                    p_c = acts_c["projector"].cpu().squeeze(0)
                    if p_u.dim() == 2:
                        p_u = p_u.mean(dim=0)
                        p_c = p_c.mean(dim=0)
                    p_activations_u_list.append(p_u)
                    p_activations_c_list.append(p_c)
                    sample_ids.append(it["sample_id"])
                    image_types.append(it["image_type"])
                    splits.append(it["split"])
    
    extractor_u.remove_hooks()
    extractor_c.remove_hooks()
    
    v_activations_u = torch.stack(v_activations_u_list, dim=0)
    v_activations_c = torch.stack(v_activations_c_list, dim=0)
    p_activations_u = torch.stack(p_activations_u_list, dim=0)
    p_activations_c = torch.stack(p_activations_c_list, dim=0)
    
    if method == "awq":
        print("Computing AWQ normalization statistics...")
        v_mu_c, v_sigma_c = compute_awq_normalization_stats(v_activations_c_list)
        v_activations_c = normalize_activations(v_activations_c, v_mu_c, v_sigma_c)
        p_mu_c, p_sigma_c = compute_awq_normalization_stats(p_activations_c_list)
        p_activations_c = normalize_activations(p_activations_c, p_mu_c, p_sigma_c)
        v_awq_stats = {"mu": v_mu_c, "sigma": v_sigma_c}
        p_awq_stats = {"mu": p_mu_c, "sigma": p_sigma_c}
    else:
        v_awq_stats = None
        p_awq_stats = None
    
    del model_u, model_c, processor_u, processor_c
    flush_gpu()
    
    v_result = {
        "activations_u": v_activations_u,
        "activations_c": v_activations_c,
        "sample_ids": sample_ids,
        "image_types": image_types,
        "splits": splits,
        "model_name": model_name,
        "method": method,
        "component": "V_P",
        "token_type": "cls",
        "extraction_component": "V",
    }
    if v_awq_stats is not None:
        v_result["awq_stats"] = v_awq_stats
    
    p_result = {
        "activations_u": p_activations_u,
        "activations_c": p_activations_c,
        "sample_ids": sample_ids,
        "image_types": image_types,
        "splits": splits,
        "model_name": model_name,
        "method": method,
        "component": "V_P",
        "token_type": "cls",
        "extraction_component": "P",
    }
    if p_awq_stats is not None:
        p_result["awq_stats"] = p_awq_stats
    
    return v_result, p_result
