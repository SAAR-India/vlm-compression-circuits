"""
Load Visual-Counterfact dataset with incorrect_answer for activation patching.

The filtered counterfactual_selected may not include incorrect_answer.
We merge it from the original Visual-Counterfact source (data/ or HF).
"""

from pathlib import Path
from typing import Dict, List, Optional

from datasets import concatenate_datasets, load_from_disk, load_dataset

from . import config


def _normalize_answer(answer) -> str:
    """Parse and return first/normalized answer string."""
    if isinstance(answer, list):
        return str(answer[0]).strip() if answer else ""
    s = str(answer).strip()
    if s.startswith("[") and s.endswith("]"):
        import ast
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    return s


def _load_incorrect_answer_mapping() -> Dict[str, str]:
    """
    Load sample_id -> incorrect_answer from original Visual-Counterfact.
    sample_id format: visual_counterfact_{split}_{i}
    """
    mapping = {}
    if config.VISUAL_COUNTERFACT_SOURCE_DIR.exists():
        ds_dict = load_from_disk(str(config.VISUAL_COUNTERFACT_SOURCE_DIR))
        for split_name in ("color", "size"):
            if split_name not in ds_dict:
                continue
            ds = ds_dict[split_name]
            for i in range(len(ds)):
                row = ds[i]
                sid = f"visual_counterfact_{split_name}_{i}"
                inc = row.get("incorrect_answer")
                if inc is not None:
                    mapping[sid] = _normalize_answer(inc)
    else:
        ds_dict = load_dataset(config.HF_VISUAL_COUNTERFACT_ID)
        for split_name in ("color", "size"):
            if split_name not in ds_dict:
                continue
            ds = ds_dict[split_name]
            for i in range(len(ds)):
                row = ds[i]
                sid = f"visual_counterfact_{split_name}_{i}"
                inc = row.get("incorrect_answer")
                if inc is not None:
                    mapping[sid] = _normalize_answer(inc)
    return mapping


def load_patching_dataset(split: str = "all") -> List[Dict]:
    """
    Load Visual-Counterfact dataset from counterfactual_selected with incorrect_answer.
    Returns list of dicts: sample_id, image_original, image_counterfact, question,
    correct_answer, incorrect_answer, split, source_split.
    """
    ds_dict = load_from_disk(str(config.VISUAL_COUNTERFACT_DIR))
    train_ds = ds_dict["attribute_binding_train"]
    val_ds = ds_dict["attribute_binding_val"]
    all_ds = concatenate_datasets([train_ds, val_ds])

    if split != "all":
        all_ds = all_ds.filter(lambda x: x.get("split") == split)

    inc_map = _load_incorrect_answer_mapping()
    has_incorrect_in_ds = "incorrect_answer" in all_ds.column_names
    records = []
    for i in range(len(all_ds)):
        row = all_ds[i]
        sid = row["sample_id"]
        inc = row.get("incorrect_answer") if has_incorrect_in_ds else None
        if inc is None or not str(inc).strip():
            inc = inc_map.get(sid)
        if inc is None or not str(inc).strip():
            corr = _normalize_answer(row.get("correct_answer", ""))
            if not corr:
                continue
            inc = _get_fallback_incorrect(sid, row, corr)
        if not str(inc).strip():
            continue
        records.append({
            "sample_id": sid,
            "image_original": row["image_original"],
            "image_counterfact": row["image_counterfact"],
            "question": row["question"],
            "correct_answer": row["correct_answer"],
            "incorrect_answer": inc,
            "split": row["split"],
            "source_split": row.get("source_split", "unknown"),
        })
    return records


def _get_fallback_incorrect(sample_id: str, row: Dict, correct: str) -> str:
    """
    Fallback when incorrect_answer not in source.
    For size split: question asks "Which object larger?" - we cannot infer.
    For color: we cannot infer. Return empty to skip or use a placeholder.
    """
    return ""
