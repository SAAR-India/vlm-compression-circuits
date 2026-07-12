"""
Load datasets from data/ into a unified list of records for inference and filtering.

For Visual-Counterfact (VQA):
- `correct_answer` is stored on disk as a string; e.g. "['green']".
- We parse it into a list of lowercase gold answers in `correct_answer_normalized`
  for robust correctness checks.
"""
import ast
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from config import (
    CIRCUIT_TYPE_ATTRIBUTE_BINDING,
    CIRCUIT_TYPE_OBJECT_RECOGNITION,
    CIRCUIT_TYPE_ANSWER_GENERATION,
    COCO_COUNTERFACTUALS_DIR,
    VISUAL_COUNTERFACT_DIR,
    XAITK_DIR,
)
import prepross_utils


def _normalize_correct_answer(answer):
    """Parse and normalize Visual-Counterfact `correct_answer` into a list of strings.

    On disk this dataset stores color answers as a string like \"['green']\" (not a real list).
    Size answers are strings like \"tree\".
    """

    if isinstance(answer, list):
        return [str(a).lower().strip() for a in answer]
    s = str(answer).strip()
    if s.startswith("["):
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x).lower().strip() for x in parsed]
    return [s.lower()]


def _load_visual_counterfact():
    """Load Visual-Counterfact color and size splits. Returns list of dicts."""
    records = []
    ds_dict = load_from_disk(str(VISUAL_COUNTERFACT_DIR))
    for split_name in ("color", "size"):
        if split_name not in ds_dict:
            continue
        ds = ds_dict[split_name]
        for i, row in enumerate(tqdm(ds, desc=f"Visual-Counterfact ({split_name})", unit="sample")):
            question = prepross_utils.question_for_split(split_name, row)
            correct_answer = row["correct_answer"]
            correct_answer_normalized = _normalize_correct_answer(correct_answer)
            
            records.append({
                "source": "Visual-Counterfact",
                "source_split": split_name,
                "circuit_type": CIRCUIT_TYPE_ATTRIBUTE_BINDING,
                "sample_id": f"visual_counterfact_{split_name}_{i}",
                "image_original": row["original_image"],
                "image_counterfact": row["counterfact_image"],
                "question": question,
                "caption_original": None,
                "caption_counterfact": None,
                "correct_answer": correct_answer,
                "correct_answer_normalized": correct_answer_normalized,
                "incorrect_answer": row["incorrect_answer"],
                "object": row["object"],
                "task": "vqa",
            })
    return records


def _load_coco_counterfactuals():
    """Load COCO-Counterfactuals train split.
    
    Per arxiv:2309.14356, creates two samples per original record:
    1. Text-counterfactual: image_0 with caption_0 (original) and caption_1 (counterfactual text)
       - Tests if model's caption matches original when text is modified
    2. Image-counterfactual: image_0 (original) vs image_1 (edited image) with caption_0
       - Tests if model's caption changes appropriately when image is modified
    
    Returns list of dicts.
    """
    records = []
    if not COCO_COUNTERFACTUALS_DIR.exists():
        return records
    ds_dict = load_from_disk(str(COCO_COUNTERFACTUALS_DIR))
    ds = ds_dict["train"]
    for i, row in enumerate(tqdm(ds, desc="COCO-Counterfactuals", unit="sample")):
        sample_id = row.get("id", f"coco_counterfact_{i}")
        
        # Sample 1: Text-counterfactual
        # Single image (image_0), two captions (caption_0=original, caption_1=counterfactual)
        # Task: Caption image_0 and compare to both captions
        records.append({
            "source": "COCO-Counterfactuals",
            "source_split": "text_counterfactual",
            "circuit_type": CIRCUIT_TYPE_ANSWER_GENERATION,
            "sample_id": f"{sample_id}_text_cf",
            "image_original": row["image_0"],
            "image_counterfact": None,  # No second image for text-cf
            "question": None,
            "caption_original": row["caption_0"],
            "caption_counterfact": row["caption_1"],
            "correct_answer": row["caption_0"],
            "incorrect_answer": row["caption_1"],
            "object": None,
            "task": "caption",
            "counterfactual_type": "text",
        })
        
        # Sample 2: Image-counterfactual  
        # Two images (image_0=original, image_1=edited), single reference caption (caption_0)
        # Task: Caption both images and see if edits change the caption appropriately
        records.append({
            "source": "COCO-Counterfactuals",
            "source_split": "image_counterfactual",
            "circuit_type": CIRCUIT_TYPE_ANSWER_GENERATION,
            "sample_id": f"{sample_id}_image_cf",
            "image_original": row["image_0"],
            "image_counterfact": row["image_1"],
            "question": None,
            "caption_original": row["caption_0"],
            "caption_counterfact": row["caption_1"],  # Expected caption for edited image
            "correct_answer": row["caption_0"],
            "incorrect_answer": None,
            "object": None,
            "task": "caption",
            "counterfactual_type": "image",
        })
    return records


def _load_xaitk():
    """Load XAITK from annotations.json and image dirs. Only ids with both images."""
    import json

    records = []
    ann_path = XAITK_DIR / "annotations.json"
    orig_dir = XAITK_DIR / "original"
    cf_dir = XAITK_DIR / "counterfactual"
    if not ann_path.exists() or not orig_dir.exists() or not cf_dir.exists():
        return records

    with open(ann_path) as f:
        data = json.load(f)

    keys = list(data.keys())
    n = len(data[keys[0]]) if keys else 0
    for i in tqdm(range(n), desc="XAITK", unit="sample"):
        row_id = data["id"][i]
        orig_path = orig_dir / f"{row_id}.png"
        cf_path = cf_dir / f"{row_id}.png"
        if not orig_path.exists() or not cf_path.exists():
            continue
        question = data["orig_question"][i]
        records.append({
            "source": "XAITK",
            "source_split": "default",
            "circuit_type": CIRCUIT_TYPE_OBJECT_RECOGNITION,
            "sample_id": f"xaitk_{row_id}",
            "image_original_path": str(orig_path),
            "image_counterfact_path": str(cf_path),
            "image_original": None,
            "image_counterfact": None,
            "question": question,
            "caption_original": None,
            "caption_counterfact": None,
            "correct_answer": None,
            "incorrect_answer": None,
            "object": None,
            "task": "vqa",
        })
    return records


def load_all_sources(include_visual_counterfact=True, include_coco=True, include_xaitk=False):
    """
    Load sources and return a single list of records.
    
    Args:
        include_visual_counterfact: Load Visual-Counterfact color/size splits (VQA task)
        include_coco: Load COCO-Counterfactuals (caption task, creates text-cf and image-cf samples)
        include_xaitk: Load XAITK dataset (disabled by default - dataset format not finalized)
    
    Note: XAITK records use image_original_path / image_counterfact_path; others use PIL Image objects.
    """
    all_records = []
    if include_visual_counterfact and VISUAL_COUNTERFACT_DIR.exists():
        all_records.extend(_load_visual_counterfact())
    if include_coco and COCO_COUNTERFACTUALS_DIR.exists():
        all_records.extend(_load_coco_counterfactuals())
    if include_xaitk and XAITK_DIR.exists():
        all_records.extend(_load_xaitk())
    return all_records
