#!/usr/bin/env python
"""
This script runs OCRBench_V2 on the VLM baseline and compressed checkpoints.

Usage: python run_ocrbench.py <args>. Look at ocr_bench_globals.py to find args list. 
"""
from __future__ import annotations

import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from PIL import Image

import utils
import ocr_bench_globals

def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]

def move_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return inputs.to(device=device, dtype=dtype)

def move_mapping_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if torch.is_tensor(value) and value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved

def blip_question_seq_max_len(processor: Any, user_cap: int) -> int:
    """Clamp question seq length so it fits the BLIP-VQA text encoder (512 positional embeddings)."""
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return min(user_cap, 512)
    tok_max = getattr(tok, "model_max_length", 512)
    if not isinstance(tok_max, int) or tok_max <= 0:
        tok_max = 512
    if tok_max > 512:
        tok_max = 512
    return min(tok_max, user_cap)

def flush_gpu() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def is_awq_checkpoint(model_path: Path) -> bool:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    quantization_config = config.get("quantization_config") or {}
    return quantization_config.get("quant_method") == "awq"

def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    out_features, in_features_packed = packed.shape
    unpacked = torch.zeros(
        (out_features, in_features_packed * 8),
        dtype=torch.int32,
        device=packed.device,
    )
    for offset in range(8):
        unpacked[:, offset::8] = (packed >> (offset * 4)) & 0xF
    return unpacked

def awq_state_dict_to_fp16(state_dict: Dict[str, torch.Tensor], quantized_layers: List[str], 
                           group_size: int) -> Dict[str, torch.Tensor]:
    converted = {
        key: value.clone()
        for key, value in state_dict.items()
        if not key.endswith(".qweight")
        and not key.endswith(".scales")
        and not key.endswith(".zeros")
    }
    for full_key in quantized_layers:
        qkey = f"{full_key}.qweight"
        skey = f"{full_key}.scales"
        zkey = f"{full_key}.zeros"
        if qkey not in state_dict:
            continue

        packed = state_dict[qkey]
        scales = state_dict[skey]
        zeros = state_dict[zkey]
        if zeros.ndim == 3:
            zeros = zeros.squeeze(-1)

        weight_q = unpack_int4(packed)
        scales_expanded = scales.repeat_interleave(group_size, dim=1)
        zeros_expanded = zeros.repeat_interleave(group_size, dim=1)
        weight_fp = (weight_q.float() - zeros_expanded.float()) * scales_expanded
        converted[f"{full_key}.weight"] = weight_fp.to(torch.float16)
    return converted

def model_load_kwargs() -> Dict[str, Any]:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "cuda:0"
    return kwargs

def load_model_for_eval(model_family: str, model_path: str) -> Tuple[Any, Any]:
    model_kwargs = model_load_kwargs()
    path = Path(model_path)

    if model_family == "blip":
        from transformers import BlipForQuestionAnswering, BlipProcessor

        model_cls = BlipForQuestionAnswering
        processor_cls = BlipProcessor
        default_model_id = ocr_bench_globals.BLIP_VQA_MODEL_ID
        processor_kwargs: Dict[str, Any] = {}
    elif model_family == "llava":
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model_cls = LlavaForConditionalGeneration
        processor_cls = AutoProcessor
        default_model_id = ocr_bench_globals.LLAVA15_MODEL_ID
        processor_kwargs = {"use_fast": False}
    elif model_family == "qwen":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
        processor_cls = AutoProcessor
        default_model_id = ocr_bench_globals.QWEN3VL_MODEL_ID
        processor_kwargs = {}
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    if path.is_dir() and is_awq_checkpoint(path):
        from safetensors.torch import load_file

        with (path / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        base_model_id = config.get("base_model_id") or default_model_id
        quantized_layers = config.get("quantized_layers", [])
        group_size = (config.get("quantization_config") or {}).get("group_size", 128)

        state_dict = load_file(str(path / "model.safetensors"))
        converted = awq_state_dict_to_fp16(state_dict, quantized_layers, group_size)
        model = model_cls.from_pretrained(base_model_id, **model_kwargs)
        model.load_state_dict(converted, strict=True)
        processor = processor_cls.from_pretrained(path, **processor_kwargs)
    else:
        model = model_cls.from_pretrained(model_path, **model_kwargs)
        processor = processor_cls.from_pretrained(default_model_id, **processor_kwargs)

    tokenizer = getattr(processor, "tokenizer", None)
    if model_family in {"llava", "qwen"} and tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model, processor

def generate_chat_prediction(model: Any, processor: Any, image: Image.Image, question: str,
                            device: torch.device, model_family: str, max_new_tokens: int) -> str:
    if model_family == "qwen":
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
    elif model_family == "llava":
        inputs = processor(images=image, text=ocr_bench_globals.build_llava_prompt(question), return_tensors="pt", padding=True)
    else:
        raise ValueError(f"Chat prediction is not used for model family: {model_family}")

    inputs = move_mapping_to_device(dict(inputs), device)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True)
    gen_ids = out.sequences[0, prompt_len:]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

@torch.no_grad()
def generate_predictions(model_family: str, model_path: str, rows: List[Dict[str, Any]], ocrbench_dir: Path,
                        output_path: Path, batch_size: int, max_new_tokens: int, max_question_tokens: int) -> None:
    flush_gpu()
    display_name = ocr_bench_globals.MODEL_CONFIGS[model_family]["display_name"]
    print(f"Loading {display_name} model from: {model_path}")
    model, processor = load_model_for_eval(model_family, model_path)
    device = next(model.parameters()).device
    q_cap = blip_question_seq_max_len(processor, max_question_tokens) if model_family == "blip" else max_question_tokens
    effective_batch_size = batch_size if model_family == "blip" else 1
    print(f"Model device: {device}; samples: {len(rows)}; batch_size: {effective_batch_size}. question truncation max_length={q_cap}")

    predictions: List[Dict[str, Any]] = []
    progress = tqdm(total=len(rows), desc=output_path.stem, unit="sample")
    try:
        for batch in batched(rows, effective_batch_size):
            images = [ocr_bench_globals.load_image(ocrbench_dir / str(row["image_path"])) for row in batch]
            questions = [str(row.get("question", "")) for row in batch]
            if model_family == "blip":
                inputs = processor(
                    images=images,
                    text=questions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=q_cap,
                )
                inputs = move_inputs_to_device(inputs, device)
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True,
                                    output_scores=True)
                num_steps = len(out.scores) if out.scores else 0
                if num_steps == 0:
                    batch_preds = [""] * len(batch)
                else:
                    batch_preds = [processor.decode(out.sequences[i, -num_steps:], skip_special_tokens=True).strip() for i in range(out.sequences.shape[0])]
            else:
                batch_preds = [
                    generate_chat_prediction(
                        model=model,
                        processor=processor,
                        image=image,
                        question=question[:q_cap],
                        device=device,
                        model_family=model_family,
                        max_new_tokens=max_new_tokens,
                    )
                    for image, question in zip(images, questions)
                ]

            for row, pred in zip(batch, batch_preds):
                next_row = dict(row)
                next_row["predict"] = str(pred)
                predictions.append(next_row)
            progress.update(len(batch))
    finally:
        progress.close()
        del model, processor
        flush_gpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Wrote predictions: {output_path}")

def run_official_eval(eval_scripts_dir: Path, ocrbench_dir: Path, preds_path: Path, scored_path: Path,
                    score_stdout_path: Path, skip_get_score: bool) -> Optional[str]:
    eval_py = eval_scripts_dir / "eval.py"
    get_score_py = eval_scripts_dir / "get_score.py"
    if not eval_py.is_file() or not get_score_py.is_file():
        raise FileNotFoundError(f"Expected eval.py/get_score.py under {eval_scripts_dir}")

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scoring predictions with official eval.py: {preds_path}")
    print(f"(subprocess scorer python={sys.executable})", flush=True)
    subprocess.run([sys.executable, str(eval_py), "--input_path",
                    str(preds_path), "--output_path", str(scored_path)],
                    cwd=str(ocrbench_dir),
                    check=True)
    print(f"Wrote scored samples: {scored_path}")

    if skip_get_score:
        return None

    print(f"Running official get_score.py: {scored_path}")
    proc = subprocess.run([sys.executable, str(get_score_py), "--json_file", str(scored_path)],
                        cwd=str(ocrbench_dir), text=True, capture_output=True)
    score_stdout_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print("Official get_score.py failed. This often happens for subset runs that do have both EN/CN in them.")
        return None

    print(proc.stdout)
    print(f"Wrote official score output: {score_stdout_path}")
    return proc.stdout

def average(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None

def summarize_scores(scored_path: Path, summary_path: Path) -> Dict[str, Any]:
    with scored_path.open("r", encoding="utf-8") as f:
        scored_rows = json.load(f)

    def collect(category_types: Dict[str, set]) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for category, task_types in category_types.items():
            scores = [
                float(row["score"])
                for row in scored_rows
                if row.get("ignore") != "True"
                and row.get("type") in task_types
                and isinstance(row.get("score"), (int, float))
            ]
            summary[category] = {"score": average(scores), "count": len(scores)}
        return summary

    english = collect(ocr_bench_globals.EN_CATEGORY_TYPES)
    chinese = collect(ocr_bench_globals.CN_CATEGORY_TYPES)
    english_present = [item["score"] for item in english.values() if item["score"] is not None]
    chinese_present = [item["score"] for item in chinese.values() if item["score"] is not None]
    language_scores = [score for score in [average(english_present), average(chinese_present)] if score is not None]

    summary = {
        "n_samples": len(scored_rows),
        "english": english,
        "chinese": chinese,
        "english_overall": average(english_present),
        "chinese_overall": average(chinese_present),
        "overall_present_languages": average(language_scores),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    return summary

def validate_selected_models(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    jobs: List[Tuple[str, str, str]] = []
    for model_family in args.models:
        if model_family == "blip":
            baseline_path = args.baseline_model_path
            compressed_path = args.compressed_model_path
        elif model_family == "llava":
            baseline_path = args.llava_baseline_model_path
            compressed_path = args.llava_compressed_model_path
        elif model_family == "qwen":
            baseline_path = args.qwen_baseline_model_path
            compressed_path = args.qwen_compressed_model_path
        else:
            raise ValueError(f"Unsupported model family: {model_family}")

        if compressed_path is None:
            raise ValueError(
                f"--{ocr_bench_globals.MODEL_CONFIGS[model_family]['default_compressed_arg'].replace('_', '-')} "
                f"is required when '{model_family}' is included in --models."
            )
        resolved_compressed_path = ocr_bench_globals.resolve_path(compressed_path)
        if not resolved_compressed_path.is_dir():
            raise FileNotFoundError(f"Compressed {model_family} model directory not found: {resolved_compressed_path}")

        jobs.append((model_family, "baseline", str(baseline_path)))
        jobs.append((model_family, "compressed", str(resolved_compressed_path)))
    return jobs

def main() -> None:
    args = ocr_bench_globals.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.max_question_tokens < 16:
        raise ValueError("--max-question-tokens must be >= 16")

    ocrbench_dir = utils.resolve_path(args.ocrbench_dir)
    json_file = utils.resolve_path(args.json_file) if args.json_file else ocrbench_dir / "OCRBench_v2.json"
    eval_scripts_dir = utils.resolve_path(args.eval_scripts_dir) if args.eval_scripts_dir else ocrbench_dir / "eval_scripts"
    output_dir = utils.resolve_path(args.output_dir)
    jobs = validate_selected_models(args)

    if not json_file.is_file():
        raise FileNotFoundError(f"OCRBench JSON not found: {json_file}")

    eval_py_chk = eval_scripts_dir / "eval.py"
    if not eval_py_chk.is_file():
        raise FileNotFoundError(f"Missing OCRBench scorer: {eval_py_chk}")

    req_txt = ocrbench_dir / "requirements.txt"
    if not req_txt.is_file():
        req_txt = ocr_bench_globals.REPO_ROOT / "src" / "ocr_bench" / "requirements.txt"

    rows = utils.load_ocrbench_rows(json_file, ocrbench_dir, args.subset, args.limit)
    if not rows:
        raise RuntimeError("No OCRBench rows selected. Check --ocrbench-dir, --subset, and --limit.")
    print(f"Selected {len(rows)} OCRBench row(s) from {json_file}")

    summaries: Dict[str, Dict[str, Any]] = {}

    for model_family, label, model_path in jobs:
        run_name = utils.output_run_name(model_family, label)
        preds_path = output_dir / f"{run_name}_preds.json"
        scored_path = output_dir / f"{run_name}_scored.json"
        stdout_path = output_dir / f"{run_name}_get_score.txt"
        summary_path = output_dir / f"{run_name}_summary.json"

        if args.skip_existing_preds and preds_path.is_file():
            print(f"Reusing existing predictions: {preds_path}")
        else:
            generate_predictions(
                model_family=model_family,
                model_path=model_path,
                rows=rows,
                ocrbench_dir=ocrbench_dir,
                output_path=preds_path,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_question_tokens=args.max_question_tokens,
            )

        run_official_eval(
            eval_scripts_dir=eval_scripts_dir,
            ocrbench_dir=ocrbench_dir,
            preds_path=preds_path,
            scored_path=scored_path,
            score_stdout_path=stdout_path,
            skip_get_score=args.skip_official_get_score,
        )
        summaries[run_name] = summarize_scores(scored_path, summary_path)

    comparison_path = output_dir / "vlm_ocrbench_comparison_summary.json"
    comparison_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    utils.print_comparison(summaries)
    print(f"Wrote comparison summary: {comparison_path}")


if __name__ == "__main__":
    main()
