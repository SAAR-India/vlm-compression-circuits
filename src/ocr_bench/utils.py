import json
import Path
import argparse
from typing import Dict, Any, List

from PIL import Image, UnidentifiedImageError

import ocr_bench_globals

def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ocr_bench_globals.REPO_ROOT / path

def build_llava_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate + score OCRBench_v2 on VLMs")
    parser.add_argument("--models", nargs="+", choices=sorted(ocr_bench_globals.MODEL_CONFIGS),
                        default=["blip"], help="Model families to evaluate")
    parser.add_argument("--ocrbench-dir", type=Path,
                        default=ocr_bench_globals.DEFAULT_OCRBENCH_DIR, 
                        help="Directory containing all of the scripts and evals")
    parser.add_argument("--json-file", type=Path, help="Path to OCRBench_v2.json")
    parser.add_argument("--eval-scripts-dir", type=Path, help="Path to official eval_scripts")

    # BLIP-VQA defaults to  Salesforce/blip-vqa-base
    parser.add_argument("--compressed-model-path", type=Path, help="Compressed BLIP-VQA checkpoint directory")
    parser.add_argument("--baseline-model-path", default=ocr_bench_globals.BLIP_VQA_MODEL_ID, help="Baseline BLIP-VQA model id/path")
    # LLaVA defaults to llava-hf/llava-1.5-7b-hf
    parser.add_argument("--llava-baseline-model-path", default=ocr_bench_globals.LLAVA15_MODEL_ID,help="Baseline LLaVA model id/path")
    parser.add_argument("--llava-compressed-model-path", type=Path, help="Compressed LLaVA checkpoint directory")
    # QWEN defaults to Qwen/Qwen3-VL-2B-Instruct
    parser.add_argument("--qwen-baseline-model-path", default=ocr_bench_globals.QWEN3VL_MODEL_ID, help="Baseline Qwen model id/path")
    parser.add_argument("--qwen-compressed-model-path", type=Path, help="Compressed Qwen checkpoint directory")

    parser.add_argument("--output-dir", type=Path, default=ocr_bench_globals.DEFAULT_OUTPUT_DIR, help="Directory for prediction, scored, stdout, and summary files")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max generated answer tokens per OCRBench sample")
    # This value is set to 512 to prevent the BLIP-VQA encoder from breaking.
    parser.add_argument("--max-question-tokens", type=int, default=512, help="Truncate tokenized OCRBench questions to this length")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for smoke tests")
    parser.add_argument("--subset", choices=["all", "en", "cn", "available"], default="all",
                        help="Rows to run. Defaults to the full OCRBench v2 JSON")
    parser.add_argument("--skip-existing-preds", action="store_true", help="Reuse prediction JSONs if they already exist")
    parser.add_argument("--skip-official-get-score", action="store_true", help="Skip eval_scripts/get_score.py")

    return parser.parse_args()

def load_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Could not load OCRBench image: {path}") from exc

def print_comparison(summaries: Dict[str, Dict[str, Any]]) -> None:
    print("OCRBench VLM comparison")
    print("=" * 24)
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

def output_run_name(model_family: str, label: str) -> str:
    if model_family == "blip":
        return f"blip_vqa_{label}"
    return f"{model_family}_{label}"

def load_ocrbench_rows(json_file: Path, ocrbench_dir: Path, subset: str, limit: int) -> List[Dict[str, Any]]:
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