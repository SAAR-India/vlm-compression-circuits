"""
Build HuggingFace DatasetDict from split_dict (circuit_type -> train/val records).
Used by build_hf_dataset_from_selected_jsonl.py.
"""
import csv
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value


def build_dataset_dict(split_dict: dict) -> DatasetDict:
    """
    Convert split_dict { circuit_type: { "train": [...], "val": [...] } } to DatasetDict.
    Uses attribute_binding for Visual-Counterfact; creates attribute_binding_train, attribute_binding_val.
    """
    train_records = []
    val_records = []
    for ct, d in split_dict.items():
        for r in d.get("train", []):
            rr = dict(r)
            rr["split"] = "train"
            rr["circuit_type"] = ct
            train_records.append(rr)
        for r in d.get("val", []):
            rr = dict(r)
            rr["split"] = "val"
            rr["circuit_type"] = ct
            val_records.append(rr)

    base_features = {
        "sample_id": Value("string"),
        "image_original": HFImage(),
        "image_counterfact": HFImage(),
        "question": Value("string"),
        "correct_answer": Value("string"),
        "split": Value("string"),
        "source_split": Value("string"),
    }
    if train_records and "incorrect_answer" in train_records[0]:
        base_features["incorrect_answer"] = Value("string")

    features = Features(base_features)
    train_ds = Dataset.from_list(train_records, features=features)
    val_ds = Dataset.from_list(val_records, features=features)

    return DatasetDict({
        "attribute_binding_train": train_ds,
        "attribute_binding_val": val_ds,
    })


def export_csv(flat_records: list, csv_path: Path) -> None:
    """Export flat records to CSV (subset of columns for metadata)."""
    if not flat_records:
        return
    keys = list(flat_records[0].keys())
    skip = {"image_original", "image_counterfact"}
    keys = [k for k in keys if k not in skip]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in flat_records:
            row = {k: r.get(k) for k in keys}
            for k, v in row.items():
                if hasattr(v, "tolist"):
                    row[k] = str(v)
            writer.writerow(row)
