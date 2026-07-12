"""
Self-contained script to load the Visual-Counterfact dataset and compressed VLM
models from Hugging Face, then save them where the crosscoder code expects.

Run from repo root:
  python preprocessing/setup_crosscoder_from_hf.py
  python preprocessing/setup_crosscoder_from_hf.py --dataset-only
  python preprocessing/setup_crosscoder_from_hf.py --models-only

Output:
  - Dataset: <repo_root>/output/counterfactual_selected/
  - Compressed models: <repo_root>/src/compressed_models/ (blip2__wanda__V, etc.)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, load_dataset
from huggingface_hub import snapshot_download

import prepross_utils
import crosscoder_configs

def _download_compressed_models(repo_id: str, local_dir: Path) -> None:
    """Download compressed model checkpoints from HF into src/compressed_models/."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading compressed models from {repo_id} to {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Done. Compressed models are at: {local_dir}")

def _records_from_hf_ds(hf_ds, split_name: str):
    """Convert one HF split (color or size) to list of records with crosscoder columns."""
    records = []
    for i, row in enumerate(hf_ds):
        # HF dataset uses original_image / counterfact_image (PIL when loaded)
        image_original = row.get("original_image")
        image_counterfact = row.get("counterfact_image")
        if image_original is None or image_counterfact is None:
            continue
        question = prepross_utils.question_for_split(split_name, row)
        records.append({
            "sample_id": f"visual_counterfact_{split_name}_{i}",
            "image_original": image_original,
            "image_counterfact": image_counterfact,
            "question": question,
            "correct_answer": row.get("correct_answer", ""),
            "source_split": split_name,
        })
    return records


def _build_dataset_dict(all_records: list[dict], val_fraction: float, seed: int) -> DatasetDict:
    """Split records into train/val and build DatasetDict with crosscoder split names."""
    rng = random.Random(seed)
    indices = list(range(len(all_records)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    n_train = len(indices) - n_val

    train_records = [all_records[i] for i in indices[:n_train]]
    val_records = [all_records[i] for i in indices[n_train:]]

    # Add "split" column
    for r in train_records:
        r["split"] = "train"
    for r in val_records:
        r["split"] = "val"

    features = Features({
        "sample_id": Value("string"),
        "image_original": HFImage(),
        "image_counterfact": HFImage(),
        "question": Value("string"),
        "correct_answer": Value("string"),
        "split": Value("string"),
        "source_split": Value("string"),
    })

    train_ds = Dataset.from_list(train_records, features=features)
    val_ds = Dataset.from_list(val_records, features=features)

    return DatasetDict({
        "attribute_binding_train": train_ds,
        "attribute_binding_val": val_ds,
    })


def _run_dataset_setup() -> None:
    """Download Visual-Counterfact from HF and save to output/counterfactual_selected."""
    print(f"Loading dataset: {crosscoder_configs.HF_DATASET_ID}")
    ds_dict = load_dataset(crosscoder_configs.HF_DATASET_ID)

    all_records = []
    for split_name in ("color", "size"):
        if split_name not in ds_dict:
            print(f"  Skipping missing split: {split_name}")
            continue
        part = _records_from_hf_ds(ds_dict[split_name], split_name)
        print(f"  {split_name}: {len(part)} samples")
        all_records.extend(part)

    if not all_records:
        raise RuntimeError("No records produced from HF dataset.")

    print(f"\nTotal samples: {len(all_records)}")
    print("Building train/val splits for crosscoder ...")
    out_ds = _build_dataset_dict(all_records, val_fraction=crosscoder_configs.VAL_FRACTION, seed=crosscoder_configs.RANDOM_SEED)

    crosscoder_configs.COUNTERFACT_SELECTED_DIR.parent.mkdir(parents=True, exist_ok=True)
    out_path = str(crosscoder_configs.COUNTERFACT_SELECTED_DIR)
    print(f"Saving to {out_path} ...")
    out_ds.save_to_disk(out_path)
    print("Done. Dataset is at:", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Visual-Counterfact dataset and compressed models from HF for crosscoder.")
    parser.add_argument("--dataset-only", action="store_true", help="Only download and prepare the dataset")
    parser.add_argument( "--models-only", action="store_true", help="Only download compressed models")
    parser.add_argument("--models-repo", type=str,
                        default=crosscoder_configs.HF_COMPRESSED_MODELS_REPO,
                        help=f"HuggingFace repo ID for compressed models (default: {crosscoder_configs.HF_COMPRESSED_MODELS_REPO}).")
    args = parser.parse_args()

    do_dataset = not args.models_only
    do_models = not args.dataset_only

    if do_dataset:
        _run_dataset_setup()
    if do_models:
        _download_compressed_models(args.models_repo, crosscoder_configs.COMPRESSED_MODELS_DIR)

    if do_dataset and do_models:
        print("\nCrosscoder expects: output/counterfactual_selected and src/compressed_models/")
    elif do_dataset:
        print("\nCrosscoder dataset path: output/counterfactual_selected (relative to repo root).")
    elif do_models:
        print("\nCompressed models path: src/compressed_models/ (relative to repo root).")
    print("Base models (BLIP-VQA, Qwen3-VL-2B) are loaded by crosscoder from HF by ID (cached by transformers).")


if __name__ == "__main__":
    main()
