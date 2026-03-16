"""
LLaVA activation patching (NOTICE / Edge Activation Patching style).

LLaVA-1.5-7B uses vision encoder (CLIP) + projector + Vicuna/LLaMA LLM.
Patching follows NOTICE methodology on LLM layers (self-attention, MLP):
- Clean run: image_original + question -> logit_diff = logit(correct) - logit(incorrect)
- Corrupt run: image_counterfact + question -> cf_logit_diff
- For each (layer, token, component): patch clean activation into corrupt run
- Score = patched_logit_diff - cf_logit_diff (how much patching restores toward clean)
"""

import gc
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from . import config as patching_config
from .dataset import load_patching_dataset
from .metrics import compute_comparison_metrics
from .plotting import (
    plot_layer_importance_bars,
    plot_notice_heatmaps,
    plot_token_importance,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _normalize_answer(answer) -> str:
    if isinstance(answer, list):
        return str(answer[0]).strip() if answer else ""
    s = str(answer).strip()
    if s.startswith("[") and s.endswith("]"):
        import ast
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    return s


def _build_prompt(question: str, correct_answer, incorrect_answer) -> str:
    """Build VQA prompt: question + "Answer with one word: X or Y?"""
    correct = _normalize_answer(correct_answer)
    incorrect = _normalize_answer(incorrect_answer)
    return f"{question} Answer with one word: {correct} or {incorrect}?"


def _get_device(model) -> torch.device:
    return getattr(model, "device", next(model.parameters()).device)


def _get_answer_tokens(processor, correct_answer, incorrect_answer) -> Tuple[int, int]:
    correct = _normalize_answer(correct_answer)
    incorrect = _normalize_answer(incorrect_answer)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ct = tokenizer(correct, add_special_tokens=False, return_tensors="pt").input_ids
    it = tokenizer(incorrect, add_special_tokens=False, return_tensors="pt").input_ids
    c_tok = ct[0, 0].item() if ct.shape[1] > 0 else -1
    i_tok = it[0, 0].item() if it.shape[1] > 0 else -1
    return c_tok, i_tok


def _llava_inputs_for_forward(processor, image, prompt_text: str, device, batch_size: int = 1):
    """Build model inputs using apply_chat_template with image + text."""
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if batch_size == 1:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        batch_inputs = []
        for _ in range(batch_size):
            inp = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            batch_inputs.append(inp)
        inputs = {}
        for k in batch_inputs[0].keys():
            vals = [b[k] for b in batch_inputs]
            if hasattr(vals[0], "shape"):
                inputs[k] = torch.cat(vals, dim=0)
            else:
                inputs[k] = vals[0]
    inputs.pop("token_type_ids", None)
    for k, v in inputs.items():
        if hasattr(v, "to"):
            if v.is_floating_point():
                inputs[k] = v.to(device=device, dtype=torch.float16)
            else:
                inputs[k] = v.to(device)
    return inputs


def _forward_logits(model, processor, image, prompt_text: str, use_amp: bool = False):
    """Single-sample forward. Returns logits [1, seq, vocab] at last position."""
    device = _get_device(model)
    inputs = _llava_inputs_for_forward(processor, image, prompt_text, device, batch_size=1)
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )
    with torch.inference_mode(), amp_ctx:
        outputs = model(**inputs)
    logits = outputs.logits.float()
    return logits


def _forward_logits_batched(
    model, processor, image, prompt_text: str, batch_size: int, use_amp: bool = False
):
    """Batched forward with B identical (image, prompt). Returns logits [B, seq, vocab]."""
    if batch_size <= 1:
        logits = _forward_logits(model, processor, image, prompt_text, use_amp)
        return logits.unsqueeze(0) if logits.dim() == 2 else logits
    device = _get_device(model)
    inputs = _llava_inputs_for_forward(processor, image, prompt_text, device, batch_size=batch_size)
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )
    with torch.inference_mode(), amp_ctx:
        outputs = model(**inputs)
    logits = outputs.logits.float()
    return logits


def _map_tokens_to_labels_llava(
    processor, prompt_text: str, correct_answer, incorrect_answer, num_tokens: int
) -> List[str]:
    """Map token positions to semantic labels. Pad with 'other' for image tokens at start."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    correct = _normalize_answer(correct_answer)
    incorrect = _normalize_answer(incorrect_answer)
    enc = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")
    input_ids_list = enc.input_ids[0].tolist()
    correct_ids = tokenizer(correct, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    incorrect_ids = tokenizer(incorrect, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    or_ids = tokenizer("or", add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    q_ids = tokenizer("?", add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()

    text_len = len(input_ids_list)
    labels = ["other"] * text_len
    labels[0] = "[CLS]"

    def find_subseq(seq, sub):
        results = []
        for i in range(len(seq) - len(sub) + 1):
            if seq[i : i + len(sub)] == sub:
                results.append(i)
        return results

    best = None
    for c_s in find_subseq(input_ids_list, correct_ids):
        c_e = c_s + len(correct_ids)
        for o_s in find_subseq(input_ids_list, or_ids):
            if o_s < c_e:
                continue
            o_e = o_s + len(or_ids)
            for i_s in find_subseq(input_ids_list, incorrect_ids):
                if i_s < o_e:
                    continue
                i_e = i_s + len(incorrect_ids)
                for q_s in find_subseq(input_ids_list, q_ids):
                    if q_s < i_e:
                        continue
                    best = (c_s, c_e, o_s, o_e, i_s, i_e, q_s, q_s + len(q_ids))
                    break
                if best:
                    break
            if best:
                break
        if best:
            break

    if best:
        c_s, c_e, o_s, o_e, i_s, i_e, q_s, q_e = best
        for t in range(c_s, c_e):
            labels[t] = "correct_object"
        for t in range(o_s, o_e):
            labels[t] = "or"
        for t in range(i_s, i_e):
            labels[t] = "incorrect_object"
        for t in range(q_s, min(q_e, len(labels))):
            labels[t] = "?"
    else:
        pos = 1
        for t in range(pos, min(pos + len(correct_ids), len(input_ids_list))):
            labels[t] = "correct_object"
        pos += len(correct_ids)
        for t in range(pos, min(pos + len(or_ids), len(input_ids_list))):
            labels[t] = "or"
        pos += len(or_ids)
        for t in range(pos, min(pos + len(incorrect_ids), len(input_ids_list))):
            labels[t] = "incorrect_object"
        pos += len(incorrect_ids)
        for t in range(pos, min(pos + len(q_ids), len(input_ids_list))):
            labels[t] = "?"

    text_len = len(labels)
    if text_len < num_tokens:
        labels = ["other"] * (num_tokens - text_len) + labels
    else:
        labels = labels[-num_tokens:]
    return labels


TOKEN_LABELS = ["[CLS]", "correct_object", "or", "incorrect_object", "?"]


def _aggregate_by_semantic_group(
    all_sample_results: List[Dict], num_layers: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Aggregate per-sample scores by token label. LLaVA has no cross-attn in LLM."""
    group_mlp = {lab: [] for lab in TOKEN_LABELS}
    group_attn = {lab: [] for lab in TOKEN_LABELS}
    for result in all_sample_results:
        mlp, attn = result["mlp_scores"], result["attn_scores"]
        labels = result["token_labels"]
        for lab in TOKEN_LABELS:
            idx = [i for i, l in enumerate(labels) if l == lab]
            if not idx:
                continue
            group_mlp[lab].append(mlp[idx, :].mean(axis=0))
            group_attn[lab].append(attn[idx, :].mean(axis=0))
    avg_mlp = np.zeros((len(TOKEN_LABELS), num_layers))
    avg_attn = np.zeros((len(TOKEN_LABELS), num_layers))
    for i, lab in enumerate(TOKEN_LABELS):
        if group_mlp[lab]:
            avg_mlp[i] = np.mean(group_mlp[lab], axis=0)
        if group_attn[lab]:
            avg_attn[i] = np.mean(group_attn[lab], axis=0)
    return avg_mlp, avg_attn, TOKEN_LABELS


def _get_llm_layers(model):
    """Get LLM decoder layers. LLaVA uses model.model.language_model with .model.layers or .layers."""
    lm = model.model.language_model
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return list(lm.model.layers)
    if hasattr(lm, "layers"):
        return list(lm.layers)
    raise AttributeError("Could not find language model layers in LLaVA")


def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_llava_patching_scores(
    model,
    processor,
    dataset: List[Dict],
    num_samples: int,
    model_label: str = "LLaVA",
    patch_batch_size: int = None,
    use_amp: bool = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str], List[Dict]]:
    """
    Compute per-layer, per-token, per-component patching scores for LLaVA-1.5.
    Returns (avg_mlp, avg_attn, xattn=None, token_labels, all_sample_results).
    LLaVA has no cross-attention in LLM; vision is merged into the sequence.
    """
    patch_batch_size = patch_batch_size or getattr(patching_config, "PATCH_BATCH_SIZE", 16)
    use_amp = use_amp if use_amp is not None else getattr(patching_config, "USE_AMP", True)
    device = next(model.parameters()).device
    use_amp = use_amp and device.type == "cuda"
    print(f"    GPU optimizations: patch_batch_size={patch_batch_size}, use_amp={use_amp}")

    llm_layers = _get_llm_layers(model)
    if not llm_layers:
        return None, None, None, [], []
    num_layers = len(llm_layers)
    print(f"    LLM has {num_layers} layers")

    all_sample_results = []
    n = min(num_samples, len(dataset))
    skipped = 0

    for sample_idx in tqdm(range(n), desc="Patching samples", unit="sample"):
        sample = dataset[sample_idx]
        image_clean = sample["image_original"]
        image_cf = sample["image_counterfact"]
        correct_ans = sample["correct_answer"]
        incorrect_ans = sample["incorrect_answer"]
        question = sample["question"]
        correct_str = _normalize_answer(correct_ans)
        incorrect_str = _normalize_answer(incorrect_ans)
        if not correct_str or not incorrect_str:
            skipped += 1
            continue
        c_tok, i_tok = _get_answer_tokens(processor, correct_ans, incorrect_ans)
        if c_tok < 0 or i_tok < 0:
            skipped += 1
            continue

        prompt_text = _build_prompt(question, correct_ans, incorrect_ans)

        def make_capture_hook(store_dict, layer_id):
            def hook(mod, inp, out, _lid=layer_id):
                t = out[0] if isinstance(out, tuple) else out
                store_dict[_lid] = t.detach().cpu()
            return hook

        clean_mlp, clean_attn = {}, {}
        handles = []
        for lid, layer in enumerate(llm_layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                handles.append(
                    layer.self_attn.o_proj.register_forward_hook(make_capture_hook(clean_attn, lid))
                )
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                handles.append(
                    layer.mlp.down_proj.register_forward_hook(make_capture_hook(clean_mlp, lid))
                )

        cl = _forward_logits(model, processor, image_clean, prompt_text, use_amp)
        clean_diff = (cl[0, -1, c_tok] - cl[0, -1, i_tok]).item()
        del cl
        for h in handles:
            h.remove()

        cfl = _forward_logits(model, processor, image_cf, prompt_text, use_amp)
        cf_diff = (cfl[0, -1, c_tok] - cfl[0, -1, i_tok]).item()
        del cfl

        sample_act = next(iter(clean_mlp.values()), None)
        if sample_act is None:
            skipped += 1
            continue
        num_tokens = sample_act.shape[1] if sample_act.dim() == 3 else sample_act.shape[0]

        token_labels = _map_tokens_to_labels_llava(
            processor, prompt_text, correct_ans, incorrect_ans, num_tokens
        )
        if not {"[CLS]", "correct_object", "or", "incorrect_object"}.issubset(set(token_labels)):
            skipped += 1
            continue

        mlp_scores = np.zeros((num_tokens, num_layers))
        attn_scores = np.zeros((num_tokens, num_layers))

        def get_attn_target(l):
            return llm_layers[l].self_attn.o_proj

        def get_mlp_target(l):
            return llm_layers[l].mlp.down_proj

        def make_llava_patch_hook(contribution_clean, token_indices):
            def hook(mod, inp, out, _contrib=contribution_clean, _toks=token_indices):
                t = out[0] if isinstance(out, tuple) else out
                tc = t.clone()
                for i, _tok in enumerate(_toks):
                    tc[i, _tok, :] = _contrib[0, _tok, :].to(t.device)
                return (tc,) + out[1:] if isinstance(out, tuple) else tc
            return hook

        num_token_batches = (num_tokens + patch_batch_size - 1) // patch_batch_size
        total_patch_batches = num_layers * 2 * num_token_batches
        with tqdm(total=total_patch_batches, desc="Patch forwards", unit="batch", leave=False) as pbar:
            for lid in range(num_layers):
                for comp, scores, get_target in [
                    (clean_mlp, mlp_scores, get_mlp_target),
                    (clean_attn, attn_scores, get_attn_target),
                ]:
                    if lid not in comp:
                        continue
                    target = get_target(lid)
                    if target is None:
                        continue
                    src = comp[lid]
                    for tok_start in range(0, num_tokens, patch_batch_size):
                        pbar.update(1)
                        tok_batch = list(range(tok_start, min(tok_start + patch_batch_size, num_tokens)))
                        B = len(tok_batch)

                        h = target.register_forward_hook(make_llava_patch_hook(src, tok_batch))
                        pl = _forward_logits_batched(
                            model, processor, image_cf, prompt_text, batch_size=B, use_amp=use_amp
                        )
                        for i, tok in enumerate(tok_batch):
                            pd = (pl[i, -1, c_tok] - pl[i, -1, i_tok]).item()
                            scores[tok, lid] = pd - cf_diff
                        h.remove()

        _flush()
        all_sample_results.append({
            "sample_idx": sample_idx,
            "question": str(question),
            "prompt_text": prompt_text,
            "correct_answer": correct_str,
            "incorrect_answer": incorrect_str,
            "clean_logit_diff": clean_diff,
            "cf_logit_diff": cf_diff,
            "num_tokens": num_tokens,
            "token_labels": token_labels,
            "mlp_scores": mlp_scores,
            "attn_scores": attn_scores,
        })

    print(f"    Completed: {len(all_sample_results)} successful, {skipped} skipped")
    if not all_sample_results:
        return None, None, None, [], []
    avg_mlp, avg_attn, group_labels = _aggregate_by_semantic_group(all_sample_results, num_layers)
    return avg_mlp, avg_attn, None, group_labels, all_sample_results


def run_llava_patching(
    model_variant: str,
    num_samples: int = 100,
    output_dir: Optional[Any] = None,
    patch_batch_size: Optional[int] = None,
    use_amp: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run activation patching for LLaVA-1.5-7B (uncompressed or compressed).
    model_variant: "baseline" (uncompressed), "wanda", or "awq"
    """
    from src.crosscoder.activations import load_compressed_model, load_uncompressed_model

    output_dir = output_dir or patching_config.PATCHING_RESULTS_DIR
    out_path = output_dir / f"llava__{model_variant}"
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_path / "metrics"
    plots_dir = out_path / "plots"
    metrics_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_patching_dataset(split="all")
    print(f"Loaded {len(dataset)} samples with incorrect_answer")

    if model_variant == "baseline":
        model, processor = load_uncompressed_model("llava")
        model_label = "LLaVA-1.5-7B Uncompressed"
    else:
        try:
            model, processor = load_compressed_model("llava", model_variant, "V_P")
            model_label = f"LLaVA-1.5-7B {model_variant.upper()}"
        except (FileNotFoundError, OSError) as e:
            print(f"Compressed checkpoint for llava {model_variant} not found: {e}")
            print("Run with --variant baseline for uncompressed patching.")
            return {"model": "llava", "variant": model_variant, "error": "Compressed checkpoint not found"}

    model = model.to(device)
    model.eval()

    mlp, attn, xattn, tok_labels, all_results = compute_llava_patching_scores(
        model, processor, dataset, num_samples, model_label,
        patch_batch_size=patch_batch_size,
        use_amp=use_amp,
    )

    del model, processor
    _flush()

    if mlp is None:
        return {"model": "llava", "variant": model_variant, "error": "No successful samples"}

    per_layer = {
        "mlp": mlp.sum(axis=0).tolist(),
        "self_attention": attn.sum(axis=0).tolist(),
        "cross_attention": [],
    }
    per_component = {
        "mlp": float(mlp.sum()),
        "self_attention": float(attn.sum()),
        "cross_attention": 0.0,
    }

    metrics = {
        "model": "llava",
        "variant": model_variant,
        "model_label": model_label,
        "num_samples": len(all_results),
        "num_layers": mlp.shape[1],
        "per_layer": per_layer,
        "per_component": per_component,
        "token_labels": tok_labels,
    }

    import json
    with open(metrics_dir / "patching_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_notice_heatmaps(mlp, attn, xattn, tok_labels, model_label, plots_dir / "notice_heatmap.png")
    plot_layer_importance_bars(mlp, attn, xattn, model_label, plots_dir / "layer_importance.png")
    plot_token_importance(mlp, attn, xattn, tok_labels, model_label, plots_dir / "token_importance.png")

    np.savez(out_path / "patching_data.npz", mlp=mlp, attn=attn, xattn=np.zeros_like(mlp), token_labels=tok_labels)

    return metrics
