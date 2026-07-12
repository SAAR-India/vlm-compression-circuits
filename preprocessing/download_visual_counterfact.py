"""
Download the Visual-Counterfact dataset from HuggingFace and save it to disk.
Dataset: https://huggingface.co/datasets/mgolov/Visual-Counterfact
Output: data/Visual-Counterfact/ (DatasetDict with "color" and "size" splits)

Usage:
    cd preprocessing && python download_visual_counterfact.py
"""
from pathlib import Path

from datasets import load_dataset

from config import DATA_DIR, VISUAL_COUNTERFACT_DIR

HF_DATASET_ID = "mgolov/Visual-Counterfact"


def main():
    print(f"Downloading {HF_DATASET_ID} from HuggingFace ...")
    ds_dict = load_dataset(HF_DATASET_ID)

    print(f"  Splits: {list(ds_dict.keys())}")
    for split_name, ds in ds_dict.items():
        print(f"  {split_name}: {len(ds)} samples, columns: {ds.column_names}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(VISUAL_COUNTERFACT_DIR)
    print(f"\nSaving to {out_path} ...")
    ds_dict.save_to_disk(out_path)
    print("Done. Visual-Counterfact is ready for load_sources.py.")


if __name__ == "__main__":
    main()
