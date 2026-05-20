"""
This script is utility code for the model evals
"""
import os
import io
import re
import json
import csv
import time
from PIL import Image
from typing import Tuple, List

import torch

import compression_utils
import compression_configs
from eval_ai_processor import EvalAIAnswerProcessor
from run_compression_eval import _awq_state_dict_to_fp16


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
_ANSWER_PROCESSOR = EvalAIAnswerProcessor()


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
    """Dataset-aware scoring with official TextVQA/VQA normalization."""
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


def load_model_and_processor_for_eval(model_name: str, model_path: str, device_map: str) -> tuple:
    """
    Load model and processor for evaluation. For AWQ checkpoints, loads base model
    and converts packed INT4 + scale/zero_point to FP16 in memory.
    Returns (model, processor, build_prompt_fn).
    """
    if _is_awq_checkpoint(model_path):
        from safetensors.torch import load_file
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
        base_model_id = config.get("base_model_id") or compression_configs.MODEL_CONFIGS[model_name]["model_id"]
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
        processor = BlipProcessor.from_pretrained(compression_configs.MODEL_CONFIGS["blip2"]["model_id"])
        build_prompt = build_prompt_blip2
    elif model_name == "qwen3vl":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(compression_configs.MODEL_CONFIGS["qwen3vl"]["model_id"])
        build_prompt = build_prompt_qwen3vl
    else:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(compression_configs.MODEL_CONFIGS["llava15"]["model_id"],
                                                use_fast=False)
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
    """Evaluate a single model on all specified datasets with batched inference."""
    compression_utils.flush_gpu()
    print(f"  Loading model from {model_path}...")

    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    model, processor, build_prompt = load_model_and_processor_for_eval(model_name, model_path, device_map)

    model.eval()
    device = next(model.parameters()).device
    print(f"  Model loaded on {device} ({compression_utils.gpu_mem()}), batch_size={batch_size}")

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
            for (example, prompt, _, orig_i), prediction in zip(batch_items, predictions):
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

                if evaluated <= compression_configs.DEBUG_EVAL_SAMPLES:
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
    compression_utils.flush_gpu()

    return results, detail_rows


def run_evaluation(quick: bool = False, batch_size: int = 64):
    """Evaluate all baseline + compressed models with batched inference."""
    os.makedirs(compression_configs.RESULTS_DIR, exist_ok=True)
    log = compression_utils.load_log()

    datasets_to_eval = EVAL_DATASETS_LITE if quick else EVAL_DATASETS
    limit = 50 if quick else 0  # 0 = full dataset

    models = ["llava15", "blip2"] if quick else list(compression_configs.MODEL_CONFIGS.keys())
    methods_list = ["wanda"] if quick else compression_configs.METHODS
    combos = {"V": compression_configs.COMPONENT_COMBOS["V"]} if quick else compression_configs.COMPONENT_COMBOS

    # Build job list
    jobs = []
    for model_name in models:
        cfg = compression_configs.MODEL_CONFIGS[model_name]
        # Baseline
        jobs.append({
            "job": compression_utils.jid(model_name, "baseline", "FP16"),
            "model_name": model_name,
            "model_path": cfg["model_id"],
        })
        # Compressed
        for method in methods_list:
            for comp_label in combos:
                cpath = compression_utils.out_path(model_name, method, comp_label)
                jobs.append({
                    "job": compression_utils.jid(model_name, method, comp_label),
                    "model_name": model_name,
                    "model_path": cpath,
                    "requires": cpath,
                })

    total = len(jobs)
    for i, j in enumerate(jobs, 1):
        job_name = j["job"]

        if compression_utils.is_done(log, "eval", job_name):
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

        out_dir = os.path.join(compression_configs.RESULTS_DIR, job_name)
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
        compression_utils.mark_done(log, "eval", job_name, elapsed)
        print(f"  Done in {elapsed:.0f}s")

        compression_utils.flush_gpu()
