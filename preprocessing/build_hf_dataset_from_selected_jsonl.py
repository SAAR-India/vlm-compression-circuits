"""
Build and optionally push a HuggingFace Dataset from a *selected subset* of inference outputs.

This is intended for the workflow:
1) Run inference to produce `output/inference_outputs.jsonl`
2) Filter to `output/inference_outputs_both_true.jsonl` (both models correct)
3) Build HF dataset using only those sample_ids, while re-loading images/metadata from sources.

We must re-load sources because inference JSONL intentionally omits `image_*` fields.
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

from tqdm import tqdm

from preprocess_configs import PRED_KEYS
from config import OUTPUT_DIR, HF_DATASET_ID  # noqa: E402
from build_hf_dataset import build_dataset_dict, export_csv  # noqa: E402
from filter_and_combine import train_val_split_by_circuit_type, flatten_splits  # noqa: E402
from load_sources import load_all_sources  # noqa: E402

_PREP_DIR = Path(__file__).resolve().parent
if str(_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(_PREP_DIR))

def _load_selected_predictions(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    by_id: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("sample_id")
            if not sid:
                continue
            by_id[sid] = {k: obj.get(k) for k in PRED_KEYS if k in obj}
    return by_id


def build_from_selected_jsonl(selected_jsonl: str | Path, push_to_hub: bool = False,
                            hub_id: str | None = None, out_dir: str | Path | None = None,
                            csv_path: str | Path | None = None) -> tuple[object, Path, int]:
    """Returns (DatasetDict, csv_path, n_selected)"""
    selected_jsonl = Path(selected_jsonl)
    preds_by_id = _load_selected_predictions(selected_jsonl)
    selected_ids = set(preds_by_id.keys())

    # Re-load full source records (includes image paths); keep only selected ids
    records = load_all_sources(
        include_visual_counterfact=True,
        include_coco=False,
        include_xaitk=False,
    )

    kept = []
    for r in tqdm(records, desc="Selecting records", unit="sample"):
        sid = r.get("sample_id")
        if sid in selected_ids:
            rr = dict(r)
            rr.update(preds_by_id[sid])
            kept.append(rr)

    if not kept:
        raise RuntimeError(f"No records matched selected sample_ids from {selected_jsonl}")

    split_dict = train_val_split_by_circuit_type(kept)
    ds_dict = build_dataset_dict(split_dict)

    flat = flatten_splits(split_dict)
    csv_path = Path(csv_path) if csv_path else (OUTPUT_DIR / "counterfactual_selected_metadata.csv")
    export_csv(flat, csv_path)

    out_dir = Path(out_dir) if out_dir else (OUTPUT_DIR / "counterfactual_selected")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(out_dir))
    print(f"Saved DatasetDict to {out_dir}")

    if push_to_hub:
        repo_id = hub_id or HF_DATASET_ID
        ds_dict.push_to_hub(repo_id, private=False)
        print(f"Pushed dataset to hub: {repo_id}")

    return ds_dict, csv_path, len(kept)


def main() -> None:
    p = argparse.ArgumentParser(description="Build HF dataset from selected inference JSONL")
    p.add_argument("--selected-jsonl", type=str,
                    default=str(OUTPUT_DIR / "inference_outputs_both_true.jsonl"),
                    help="Path to selected inference outputs JSONL")
    p.add_argument("--push-to-hub", action="store_true", help="Push dataset to HuggingFace Hub")
    p.add_argument("--hub-id", type=str, help="Override HuggingFace dataset repo id")
    p.add_argument("--out-dir", type=str, default=None, help="Where to save dataset on disk")
    p.add_argument("--csv", type=str, default=None, help="Path for metadata CSV")
    args = p.parse_args()

    build_from_selected_jsonl(
        selected_jsonl=args.selected_jsonl,
        push_to_hub=args.push_to_hub,
        hub_id=args.hub_id,
        out_dir=args.out_dir,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()

