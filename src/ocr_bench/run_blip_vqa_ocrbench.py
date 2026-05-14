"""
Run BLIP-VQA on OCRBench v2 with baseline and compressed checkpoints.

This script writes OCRBench-compatible prediction JSON files, runs the official
per-sample scorer, and writes a compact summary for comparing uncompressed vs.
compressed BLIP-VQA.

Example:
    python src/ocr_bench/run_blip_vqa_ocrbench.py \
        --compressed-model-path compressed_models/blip2__wanda__V_P \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BLIP_VQA_MODEL_ID = "Salesforce/blip-vqa-base"
DEFAULT_OCRBENCH_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "ocrbench_blip_vqa"


EN_CATEGORY_TYPES = {
    "text_recognition": {
        "text recognition en",
        "fine-grained text recognition en",
        "full-page OCR en",
    },
    "text_detection": {
        "text grounding en",
        "VQA with position en",
    },
    "text_spotting": {
        "text spotting en",
    },
    "relationship_extraction": {
        "key information extraction en",
        "key information mapping en",
    },
    "element_parsing": {
        "document parsing en",
        "chart parsing en",
        "table parsing en",
        "formula recognition en",
    },
    "mathematical_calculation": {
        "math QA en",
        "text counting en",
    },
    "visual_text_understanding": {
        "document classification en",
        "cognition VQA en",
        "diagram QA en",
    },
    "knowledge_reasoning": {
        "reasoning VQA en",
        "science QA en",
        "APP agent en",
        "ASCII art classification en",
    },
}

CN_CATEGORY_TYPES = {
    "text_recognition": {
        "full-page OCR cn",
    },
    "relationship_extraction": {
        "key information extraction cn",
        "handwritten answer extraction cn",
    },
    "element_parsing": {
        "document parsing cn",
        "table parsing cn",
        "formula recognition cn",
    },
    "visual_text_understanding": {
        "cognition VQA cn",
    },
    "knowledge_reasoning": {
        "reasoning VQA cn",
        "text translation cn",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and score OCRBench v2 predictions for baseline and compressed BLIP-VQA."
    )
    parser.add_argument(
        "--ocrbench-dir",
        type=Path,
        default=DEFAULT_OCRBENCH_DIR,
        help="Directory containing OCRBench_v2.json, EN_part/CN_part, and eval_scripts.",
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        default=None,
        help="Path to OCRBench_v2.json. Defaults to <ocrbench-dir>/OCRBench_v2.json.",
    )
    parser.add_argument(
        "--eval-scripts-dir",
        type=Path,
        default=None,
        help="Path to official eval_scripts. Defaults to <ocrbench-dir>/eval_scripts.",
    )
    parser.add_argument(
        "--compressed-model-path",
        type=Path,
        required=True,
        help="Compressed BLIP-VQA checkpoint directory, e.g. compressed_models/blip2__wanda__V_P.",
    )
    parser.add_argument(
        "--baseline-model-path",
        default=BLIP_VQA_MODEL_ID,
        help="Baseline BLIP-VQA model id/path. Defaults to Salesforce/blip-vqa-base.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for prediction, scored, stdout, and summary files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated answer tokens per OCRBench sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional sample limit for smoke tests. 0 means all selected samples.",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "en", "cn", "available"],
        default="all",
        help=(
            "Rows to run. Defaults to the full OCRBench v2 JSON. "
            "'available' keeps only rows whose image files exist under ocrbench-dir."
        ),
    )
    parser.add_argument(
        "--skip-existing-preds",
        action="store_true",
        help="Reuse prediction JSONs if they already exist.",
    )
    parser.add_argument(
        "--skip-official-get-score",
        action="store_true",
        help="Skip eval_scripts/get_score.py. The script still writes its own summary JSON.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_ocrbench_rows(
    json_file: Path,
    ocrbench_dir: Path,
    subset: str,
    limit: int,
) -> List[Dict[str, Any]]:
    with json_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    selected: List[Dict[str, Any]] = []
    missing_paths: List[str] = []
    for row in rows:
        image_path = str(row.get("image_path", ""))
        if subset == "en" and not image_path.startswith("EN_part/"):
            continue
        if subset == "cn" and not image_path.startswith("CN_part/"):
            continue

        full_image_path = ocrbench_dir / image_path
        if not full_image_path.is_file():
            if subset == "available":
                missing_paths.append(image_path)
                continue
            missing_paths.append(image_path)
        selected.append(row)

        if limit and len(selected) >= limit:
            break

    if subset == "available" and missing_paths:
        print(f"Filtered out {len(missing_paths)} row(s) with missing images.")
    elif missing_paths:
        examples = "\n  ".join(missing_paths[:10])
        raise FileNotFoundError(
            f"{len(missing_paths)} selected OCRBench image(s) are missing under {ocrbench_dir}. "
            "Use --subset available to skip missing files, or restore the full EN_part/CN_part tree. "
            f"Examples:\n  {examples}"
        )
    return selected


def load_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Could not load OCRBench image: {path}") from exc


def move_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return inputs.to(device=device, dtype=dtype)


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


def awq_state_dict_to_fp16(
    state_dict: Dict[str, torch.Tensor],
    quantized_layers: List[str],
    group_size: int,
) -> Dict[str, torch.Tensor]:
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


def load_blip_vqa_for_eval(model_path: str) -> Tuple[Any, Any]:
    from transformers import BlipForQuestionAnswering, BlipProcessor

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "cuda:0"

    path = Path(model_path)
    if path.is_dir() and is_awq_checkpoint(path):
        from safetensors.torch import load_file

        with (path / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        base_model_id = config.get("base_model_id") or BLIP_VQA_MODEL_ID
        quantized_layers = config.get("quantized_layers", [])
        group_size = (config.get("quantization_config") or {}).get("group_size", 128)

        state_dict = load_file(str(path / "model.safetensors"))
        converted = awq_state_dict_to_fp16(state_dict, quantized_layers, group_size)
        model = BlipForQuestionAnswering.from_pretrained(base_model_id, **model_kwargs)
        model.load_state_dict(converted, strict=True)
        processor = BlipProcessor.from_pretrained(path)
    else:
        model = BlipForQuestionAnswering.from_pretrained(model_path, **model_kwargs)
        processor = BlipProcessor.from_pretrained(BLIP_VQA_MODEL_ID)

    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model, processor


@torch.no_grad()
def generate_predictions(
    model_path: str,
    rows: List[Dict[str, Any]],
    ocrbench_dir: Path,
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    flush_gpu()
    print(f"Loading BLIP-VQA model from: {model_path}")
    model, processor = load_blip_vqa_for_eval(model_path)
    device = next(model.parameters()).device
    print(f"Model device: {device}; samples: {len(rows)}; batch_size: {batch_size}")

    predictions: List[Dict[str, Any]] = []
    progress = tqdm(total=len(rows), desc=output_path.stem, unit="sample")
    try:
        for batch in batched(rows, batch_size):
            images = [load_image(ocrbench_dir / str(row["image_path"])) for row in batch]
            questions = [str(row.get("question", "")) for row in batch]
            inputs = processor(
                images=images,
                text=questions,
                return_tensors="pt",
                padding=True,
            )
            inputs = move_inputs_to_device(inputs, device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            num_steps = len(out.scores) if out.scores else 0
            if num_steps == 0:
                batch_preds = [""] * len(batch)
            else:
                batch_preds = [
                    processor.decode(out.sequences[i, -num_steps:], skip_special_tokens=True).strip()
                    for i in range(out.sequences.shape[0])
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


def run_official_eval(
    eval_scripts_dir: Path,
    ocrbench_dir: Path,
    preds_path: Path,
    scored_path: Path,
    score_stdout_path: Path,
    skip_get_score: bool,
) -> Optional[str]:
    eval_py = eval_scripts_dir / "eval.py"
    get_score_py = eval_scripts_dir / "get_score.py"
    if not eval_py.is_file() or not get_score_py.is_file():
        raise FileNotFoundError(f"Expected eval.py/get_score.py under {eval_scripts_dir}")

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scoring predictions with official eval.py: {preds_path}")
    subprocess.run(
        [
            sys.executable,
            str(eval_py),
            "--input_path",
            str(preds_path),
            "--output_path",
            str(scored_path),
        ],
        cwd=str(ocrbench_dir),
        check=True,
    )
    print(f"Wrote scored samples: {scored_path}")

    if skip_get_score:
        return None

    print(f"Running official get_score.py: {scored_path}")
    proc = subprocess.run(
        [sys.executable, str(get_score_py), "--json_file", str(scored_path)],
        cwd=str(ocrbench_dir),
        text=True,
        capture_output=True,
    )
    score_stdout_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(
            "Official get_score.py failed. This often happens for subset runs that do "
            f"not contain both EN and CN categories. See {score_stdout_path}"
        )
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

    english = collect(EN_CATEGORY_TYPES)
    chinese = collect(CN_CATEGORY_TYPES)
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


def print_comparison(summaries: Dict[str, Dict[str, Any]]) -> None:
    print("\nOCRBench BLIP-VQA comparison")
    print("=" * 34)
    for name, summary in summaries.items():
        en = summary.get("english_overall")
        cn = summary.get("chinese_overall")
        overall = summary.get("overall_present_languages")
        print(
            f"{name}: n={summary['n_samples']} | "
            f"EN={en:.4f}" if en is not None else f"{name}: n={summary['n_samples']} | EN=n/a",
            end="",
        )
        print(f" | CN={cn:.4f}" if cn is not None else " | CN=n/a", end="")
        print(f" | overall={overall:.4f}" if overall is not None else " | overall=n/a")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")

    ocrbench_dir = resolve_path(args.ocrbench_dir)
    json_file = resolve_path(args.json_file) if args.json_file else ocrbench_dir / "OCRBench_v2.json"
    eval_scripts_dir = (
        resolve_path(args.eval_scripts_dir) if args.eval_scripts_dir else ocrbench_dir / "eval_scripts"
    )
    compressed_model_path = resolve_path(args.compressed_model_path)
    output_dir = resolve_path(args.output_dir)

    if not json_file.is_file():
        raise FileNotFoundError(f"OCRBench JSON not found: {json_file}")
    if not compressed_model_path.is_dir():
        raise FileNotFoundError(f"Compressed model directory not found: {compressed_model_path}")

    rows = load_ocrbench_rows(json_file, ocrbench_dir, args.subset, args.limit)
    if not rows:
        raise RuntimeError("No OCRBench rows selected. Check --ocrbench-dir, --subset, and --limit.")
    print(f"Selected {len(rows)} OCRBench row(s) from {json_file}")

    jobs: List[Tuple[str, str]] = [
        ("baseline", str(args.baseline_model_path)),
        ("compressed", str(compressed_model_path)),
    ]
    summaries: Dict[str, Dict[str, Any]] = {}

    for label, model_path in jobs:
        preds_path = output_dir / f"blip_vqa_{label}_preds.json"
        scored_path = output_dir / f"blip_vqa_{label}_scored.json"
        stdout_path = output_dir / f"blip_vqa_{label}_get_score.txt"
        summary_path = output_dir / f"blip_vqa_{label}_summary.json"

        if args.skip_existing_preds and preds_path.is_file():
            print(f"Reusing existing predictions: {preds_path}")
        else:
            generate_predictions(
                model_path=model_path,
                rows=rows,
                ocrbench_dir=ocrbench_dir,
                output_path=preds_path,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )

        run_official_eval(
            eval_scripts_dir=eval_scripts_dir,
            ocrbench_dir=ocrbench_dir,
            preds_path=preds_path,
            scored_path=scored_path,
            score_stdout_path=stdout_path,
            skip_get_score=args.skip_official_get_score,
        )
        summaries[label] = summarize_scores(scored_path, summary_path)

    comparison_path = output_dir / "blip_vqa_comparison_summary.json"
    comparison_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print_comparison(summaries)
    print(f"\nWrote comparison summary: {comparison_path}")


if __name__ == "__main__":
    main()
