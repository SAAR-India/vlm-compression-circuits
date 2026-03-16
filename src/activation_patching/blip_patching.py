"""
BLIP activation patching (Edge Activation Patching / NOTICE-style).

Patching methodology (ACDC, Edge Pruning, NOTICE):
- Clean run: image_original + question -> logit_diff = logit(correct) - logit(incorrect)
- Corrupt run: image_counterfact + question -> cf_logit_diff
- For each (layer, token, component): patch clean activation into corrupt run
- Score = patched_logit_diff - cf_logit_diff (how much patching restores toward clean)
"""

import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import BlipForQuestionAnswering, BlipProcessor

from . import config as patching_config
from .dataset import load_patching_dataset
from .metrics import compute_comparison_metrics
from .plotting import plot_notice_heatmaps, plot_layer_importance_bars, plot_token_importance


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


def _build_prompt(correct_answer, incorrect_answer):
    correct = _normalize_answer(correct_answer)
    incorrect = _normalize_answer(incorrect_answer)
    return f"{correct} or {incorrect} ?"


def _get_device(model) -> torch.device:
    """Get model device (PyTorch nn.Module has no .device by default)."""
    return getattr(model, "device", next(model.parameters()).device)


def _get_answer_tokens(processor, correct_answer, incorrect_answer):
    correct = _normalize_answer(correct_answer)
    incorrect = _normalize_answer(incorrect_answer)
    ct = processor.tokenizer(correct, add_special_tokens=False, return_tensors="pt").input_ids
    it = processor.tokenizer(incorrect, add_special_tokens=False, return_tensors="pt").input_ids
    c_tok = ct[0, 0].item() if ct.shape[1] > 0 else -1
    i_tok = it[0, 0].item() if it.shape[1] > 0 else -1
    return c_tok, i_tok


def _forward_logits(model, processor, image, prompt_text, use_amp: bool = False):
    """Single-sample forward. Returns logits [1, seq, vocab]."""
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    device = _get_device(model)
    inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    bos_id = processor.tokenizer.bos_token_id or 30522
    decoder_input_ids = torch.tensor([[bos_id]], device=device)
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if (use_amp and getattr(model, "device", torch.device("cpu")).type == "cuda")
        else nullcontext()
    )
    with torch.inference_mode(), amp_ctx:
        vision_outputs = model.vision_model(pixel_values=inputs.get("pixel_values"))
        image_embeds = vision_outputs[0]
        text_encoder_output = model.text_encoder(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask", None),
            encoder_hidden_states=image_embeds,
            return_dict=True,
        )
        decoder_outputs = model.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=text_encoder_output.last_hidden_state,
            return_dict=True,
        )
        if hasattr(decoder_outputs, "logits"):
            logits = decoder_outputs.logits.float()
        elif hasattr(decoder_outputs, "prediction_logits"):
            logits = decoder_outputs.prediction_logits.float()
        else:
            hidden = decoder_outputs.last_hidden_state
            logits = model.text_decoder.cls.predictions(hidden).float()
    return logits


def _forward_logits_batched(
    model, processor, image, prompt_text, batch_size: int, use_amp: bool = False
):
    """Batched forward with B identical (image, prompt). Returns logits [B, seq, vocab]."""
    if batch_size <= 1:
        logits = _forward_logits(model, processor, image, prompt_text, use_amp)
        return logits.unsqueeze(0) if logits.dim() == 2 else logits
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    device = _get_device(model)
    images = [image] * batch_size
    prompts = [prompt_text] * batch_size
    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    bos_id = processor.tokenizer.bos_token_id or 30522
    decoder_input_ids = torch.full(
        (batch_size, 1), bos_id, dtype=torch.long, device=device
    )
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )
    with torch.inference_mode(), amp_ctx:
        vision_outputs = model.vision_model(pixel_values=inputs.get("pixel_values"))
        image_embeds = vision_outputs[0]
        text_encoder_output = model.text_encoder(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask", None),
            encoder_hidden_states=image_embeds,
            return_dict=True,
        )
        decoder_outputs = model.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=text_encoder_output.last_hidden_state,
            return_dict=True,
        )
        if hasattr(decoder_outputs, "logits"):
            logits = decoder_outputs.logits.float()
        elif hasattr(decoder_outputs, "prediction_logits"):
            logits = decoder_outputs.prediction_logits.float()
        else:
            hidden = decoder_outputs.last_hidden_state
            logits = model.text_decoder.cls.predictions(hidden).float()
    return logits


def _map_tokens_to_labels(processor, prompt_text, correct_answer, incorrect_answer):
    tokenizer = processor.tokenizer
    correct = _normalize_answer(correct_answer)
    incorrect = _normalize_answer(incorrect_answer)
    full_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = full_enc.input_ids[0].tolist()
    num_tokens = len(input_ids)
    correct_ids = tokenizer(correct, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    incorrect_ids = tokenizer(incorrect, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    or_ids = tokenizer("or", add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    q_ids = tokenizer("?", add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()

    def find_subseq(seq, sub):
        results = []
        for i in range(len(seq) - len(sub) + 1):
            if seq[i:i+len(sub)] == sub:
                results.append(i)
        return results

    labels = ["other"] * num_tokens
    labels[0] = "[CLS]"
    sep_id = tokenizer.sep_token_id
    if sep_id is not None and num_tokens > 1 and input_ids[-1] == sep_id:
        labels[-1] = "[SEP]"

    best = None
    for c_s in find_subseq(input_ids, correct_ids):
        c_e = c_s + len(correct_ids)
        for o_s in find_subseq(input_ids, or_ids):
            if o_s < c_e:
                continue
            o_e = o_s + len(or_ids)
            for i_s in find_subseq(input_ids, incorrect_ids):
                if i_s < o_e:
                    continue
                i_e = i_s + len(incorrect_ids)
                for q_s in find_subseq(input_ids, q_ids):
                    if q_s < i_e:
                        continue
                    best = (c_s, c_e, o_s, o_e, i_s, i_e, q_s, q_s+len(q_ids))
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
        for t in range(q_s, q_e):
            labels[t] = "?"
    else:
        pos = 1
        end = num_tokens - 1 if (sep_id and input_ids[-1] == sep_id) else num_tokens
        for t in range(pos, min(pos+len(correct_ids), end)):
            labels[t] = "correct_object"
        pos += len(correct_ids)
        for t in range(pos, min(pos+len(or_ids), end)):
            labels[t] = "or"
        pos += len(or_ids)
        for t in range(pos, min(pos+len(incorrect_ids), end)):
            labels[t] = "incorrect_object"
        pos += len(incorrect_ids)
        for t in range(pos, min(pos+len(q_ids), end)):
            labels[t] = "?"
    return labels


# Semantic groups for aggregation
TOKEN_LABELS = ["[CLS]", "correct_object", "or", "incorrect_object", "?"]


def _aggregate_by_semantic_group(all_sample_results: List[Dict], num_layers: int):
    group_mlp = {lab: [] for lab in TOKEN_LABELS}
    group_attn = {lab: [] for lab in TOKEN_LABELS}
    group_xattn = {lab: [] for lab in TOKEN_LABELS}
    for result in all_sample_results:
        mlp, attn, xattn = result["mlp_scores"], result["attn_scores"], result["xattn_scores"]
        labels = result["token_labels"]
        for lab in TOKEN_LABELS:
            idx = [i for i, l in enumerate(labels) if l == lab]
            if not idx:
                continue
            group_mlp[lab].append(mlp[idx, :].mean(axis=0))
            group_attn[lab].append(attn[idx, :].mean(axis=0))
            group_xattn[lab].append(xattn[idx, :].mean(axis=0))
    avg_mlp = np.zeros((len(TOKEN_LABELS), num_layers))
    avg_attn = np.zeros((len(TOKEN_LABELS), num_layers))
    avg_xattn = np.zeros((len(TOKEN_LABELS), num_layers))
    for i, lab in enumerate(TOKEN_LABELS):
        if group_mlp[lab]:
            avg_mlp[i] = np.mean(group_mlp[lab], axis=0)
        if group_attn[lab]:
            avg_attn[i] = np.mean(group_attn[lab], axis=0)
        if group_xattn[lab]:
            avg_xattn[i] = np.mean(group_xattn[lab], axis=0)
    return avg_mlp, avg_attn, avg_xattn, TOKEN_LABELS


def _get_qformer_layers(model):
    try:
        return list(model.text_encoder.encoder.layer)
    except AttributeError:
        try:
            return list(model.text_encoder.bert.encoder.layer)
        except AttributeError:
            return []


def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_blip_patching_scores(
    model: torch.nn.Module,
    processor: BlipProcessor,
    dataset: List[Dict],
    num_samples: int,
    model_label: str = "BLIP",
    patch_batch_size: int = None,
    use_amp: bool = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str], List[Dict]]:
    """
    Compute per-layer, per-token, per-component patching scores for BLIP.
    Returns (avg_mlp, avg_attn, avg_xattn, token_labels, all_sample_results).
    patch_batch_size: tokens to patch per forward (reduces L*3*T forwards to L*3*ceil(T/B)).
    use_amp: mixed precision for faster GPU forwards.
    """
    patch_batch_size = patch_batch_size if patch_batch_size is not None else getattr(patching_config, "PATCH_BATCH_SIZE", 16)
    use_amp = use_amp if use_amp is not None else getattr(patching_config, "USE_AMP", True)
    device = next(model.parameters()).device
    use_amp = use_amp and device.type == "cuda"
    print(f"    GPU optimizations: patch_batch_size={patch_batch_size}, use_amp={use_amp}")
    qf_layers = _get_qformer_layers(model)
    if not qf_layers:
        return None, None, None, [], []
    num_layers = len(qf_layers)
    print(f"    Text encoder has {num_layers} layers")
    all_sample_results = []
    n = min(num_samples, len(dataset))
    skipped = 0

    for sample_idx in tqdm(range(n), desc="Patching samples", unit="sample"):
        sample = dataset[sample_idx]
        image_clean = sample["image_original"]
        image_cf = sample["image_counterfact"]
        correct_ans = sample["correct_answer"]
        incorrect_ans = sample["incorrect_answer"]
        correct_str = _normalize_answer(correct_ans)
        incorrect_str = _normalize_answer(incorrect_ans)
        if not correct_str or not incorrect_str:
            skipped += 1
            continue
        c_tok, i_tok = _get_answer_tokens(processor, correct_ans, incorrect_ans)
        if c_tok < 0 or i_tok < 0:
            skipped += 1
            continue

        prompt_text = _build_prompt(correct_ans, incorrect_ans)
        token_labels = _map_tokens_to_labels(processor, prompt_text, correct_ans, incorrect_ans)
        if not {"[CLS]", "correct_object", "or", "incorrect_object", "?"}.issubset(set(token_labels)):
            skipped += 1
            continue

        def make_capture_hook(store_dict, layer_id):
            """Factory with default-arg binding to avoid Python closure capture bug (BLIP_PATCHING_ANALYSIS.md)."""
            def hook(mod, inp, out, _lid=layer_id):
                t = out[0] if isinstance(out, tuple) else out
                store_dict[_lid] = t.detach().cpu()
            return hook

        clean_mlp, clean_attn, clean_xattn = {}, {}, {}
        handles = []
        for lid, layer in enumerate(qf_layers):
            if hasattr(layer, "output") and hasattr(layer.output, "dense"):
                handles.append(layer.output.dense.register_forward_hook(make_capture_hook(clean_mlp, lid)))
            if hasattr(layer, "attention") and hasattr(layer.attention, "output") and hasattr(layer.attention.output, "dense"):
                handles.append(layer.attention.output.dense.register_forward_hook(make_capture_hook(clean_attn, lid)))
            if hasattr(layer, "crossattention") and hasattr(layer.crossattention, "output") and hasattr(layer.crossattention.output, "dense"):
                handles.append(layer.crossattention.output.dense.register_forward_hook(make_capture_hook(clean_xattn, lid)))
        cl = _forward_logits(model, processor, image_clean, prompt_text, use_amp)
        clean_diff = (cl[0, -1, c_tok] - cl[0, -1, i_tok]).item()
        del cl
        for h in handles:
            h.remove()

        cfl = _forward_logits(model, processor, image_cf, prompt_text, use_amp)
        cf_diff = (cfl[0, -1, c_tok] - cfl[0, -1, i_tok]).item()
        del cfl

        if sample_idx == 0:
            for check_lid in list(clean_mlp.keys())[:3]:
                mlp_val = clean_mlp[check_lid]
                attn_val = clean_attn.get(check_lid)
                xattn_val = clean_xattn.get(check_lid)
                if attn_val is not None and mlp_val.shape == attn_val.shape:
                    if torch.allclose(mlp_val.float(), attn_val.float()):
                        print(f"    [VERIFY] Layer {check_lid}: MLP and Self-Attn captures identical!")
                if xattn_val is not None and mlp_val.shape == xattn_val.shape:
                    if torch.allclose(mlp_val.float(), xattn_val.float()):
                        print(f"    [VERIFY] Layer {check_lid}: MLP and Cross-Attn captures identical!")
                mlp_tgt = qf_layers[check_lid].output
                attn_tgt = qf_layers[check_lid].attention.output
                xattn_tgt = qf_layers[check_lid].crossattention.output if hasattr(qf_layers[check_lid], "crossattention") else None
                if mlp_tgt is attn_tgt:
                    print(f"    [VERIFY] Layer {check_lid}: MLP and Attn target are same module!")
                if xattn_tgt is not None and mlp_tgt is xattn_tgt:
                    print(f"    [VERIFY] Layer {check_lid}: MLP and Cross-Attn target are same module!")

        sample_act = next(iter(clean_mlp.values()), None)
        if sample_act is None:
            skipped += 1
            continue
        num_tokens = sample_act.shape[1] if sample_act.dim() == 3 else sample_act.shape[0]

        if len(token_labels) != num_tokens:
            if len(token_labels) < num_tokens:
                token_labels += ["other"] * (num_tokens - len(token_labels))
            else:
                token_labels = token_labels[:num_tokens]

        mlp_scores = np.zeros((num_tokens, num_layers))
        attn_scores = np.zeros((num_tokens, num_layers))
        xattn_scores = np.zeros((num_tokens, num_layers))

        def get_mlp_target(l):
            return qf_layers[l].output

        def get_attn_target(l):
            return qf_layers[l].attention.output

        def get_xattn_target(l):
            layer = qf_layers[l]
            if hasattr(layer, "crossattention") and hasattr(layer.crossattention, "output"):
                return layer.crossattention.output
            return None

        # Batched token patching: process tokens in batches to reduce forward passes
        num_token_batches = (num_tokens + patch_batch_size - 1) // patch_batch_size
        total_patch_batches = num_layers * 3 * num_token_batches
        with tqdm(total=total_patch_batches, desc="Patch forwards", unit="batch", leave=False) as pbar:
            for lid in range(num_layers):
                for comp, scores, get_target in [
                    (clean_mlp, mlp_scores, get_mlp_target),
                    (clean_attn, attn_scores, get_attn_target),
                    (clean_xattn, xattn_scores, get_xattn_target),
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

                        def make_batched_patch_hook(contribution_clean, token_indices):
                            def hook(mod, inp, out, _contrib=contribution_clean, _toks=token_indices):
                                t = out[0] if isinstance(out, tuple) else out
                                tc = t.clone()
                                residual = inp[1] if isinstance(inp, (tuple, list)) and len(inp) > 1 else inp[0]
                                for i, _tok in enumerate(_toks):
                                    contrib_i = _contrib[:, _tok, :].to(residual.device) + residual[i, _tok, :]
                                    ln_out = mod.LayerNorm(contrib_i.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                                    tc[i, _tok, :] = ln_out
                                return (tc,) + out[1:] if isinstance(out, tuple) else tc
                            return hook

                        h = target.register_forward_hook(make_batched_patch_hook(src, tok_batch))
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
            "question": str(sample["question"]),
            "prompt_text": prompt_text,
            "correct_answer": correct_str,
            "incorrect_answer": incorrect_str,
            "clean_logit_diff": clean_diff,
            "cf_logit_diff": cf_diff,
            "num_tokens": num_tokens,
            "token_labels": token_labels,
            "mlp_scores": mlp_scores,
            "attn_scores": attn_scores,
            "xattn_scores": xattn_scores,
        })

    print(f"    Completed: {len(all_sample_results)} successful, {skipped} skipped")
    if not all_sample_results:
        return None, None, None, [], []
    avg_mlp, avg_attn, avg_xattn, group_labels = _aggregate_by_semantic_group(all_sample_results, num_layers)
    return avg_mlp, avg_attn, avg_xattn, group_labels, all_sample_results


def run_blip_patching(
    model_variant: str,
    num_samples: int = 100,
    output_dir: Optional[Path] = None,
    patch_batch_size: Optional[int] = None,
    use_amp: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run activation patching for BLIP (uncompressed or compressed).
    model_variant: "baseline" (uncompressed), "wanda", or "awq"
    """
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from src.crosscoder.activations import load_uncompressed_model, load_compressed_model

    output_dir = output_dir or patching_config.PATCHING_RESULTS_DIR
    out_path = output_dir / f"blip2__{model_variant}"
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_path / "metrics"
    plots_dir = out_path / "plots"
    metrics_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_patching_dataset(split="all")
    print(f"Loaded {len(dataset)} samples with incorrect_answer")

    if model_variant == "baseline":
        model, processor = load_uncompressed_model("blip2")
        model_label = "BLIP Uncompressed"
    else:
        model, processor = load_compressed_model("blip2", model_variant, "V_P")
        model_label = f"BLIP {model_variant.upper()}"

    model = model.to(device)
    model.eval()

    mlp, attn, xattn, tok_labels, all_results = compute_blip_patching_scores(
        model, processor, dataset, num_samples, model_label,
        patch_batch_size=patch_batch_size,
        use_amp=use_amp,
    )

    del model, processor
    _flush()

    if mlp is None:
        return {"model": "blip2", "variant": model_variant, "error": "No successful samples"}

    per_layer = {
        "mlp": mlp.sum(axis=0).tolist(),
        "self_attention": attn.sum(axis=0).tolist(),
        "cross_attention": xattn.sum(axis=0).tolist() if xattn is not None else [],
    }
    per_component = {
        "mlp": float(mlp.sum()),
        "self_attention": float(attn.sum()),
        "cross_attention": float(xattn.sum()) if xattn is not None else 0.0,
    }

    metrics = {
        "model": "blip2",
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

    np.savez(out_path / "patching_data.npz", mlp=mlp, attn=attn, xattn=xattn, token_labels=tok_labels)

    return metrics
