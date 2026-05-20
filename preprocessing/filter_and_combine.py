"""
Filter records by base-model performance (correct on original with confidence > threshold),
then create train/val splits per circuit type.
"""
import random

from tqdm import tqdm

from config import CIRCUIT_TYPES, CONFIDENCE_THRESHOLD, RANDOM_SEED, VAL_FRACTION


def _passes_filter(record, require_both_models=True):
    """
    Keep sample if on original image:
    - Both models are correct.
    - Both confidences are >= CONFIDENCE_THRESHOLD.

    (Current pipeline scope is Visual-Counterfact only; no XAITK special cases.)
    """
    t_correct = record.get("qwen3vl_correct_original")
    t_conf = record.get("qwen3vl_confidence_original") or 0.0
    b_correct = record.get("blip_correct_original")
    b_conf = record.get("blip_confidence_original") or 0.0

    t_ok = (t_correct is True) and (t_conf >= CONFIDENCE_THRESHOLD)
    b_ok = (b_correct is True) and (b_conf >= CONFIDENCE_THRESHOLD)
    if require_both_models:
        return t_ok and b_ok
    return t_ok or b_ok


def filter_records(records, require_both_models=True):
    """Return only records that pass the quality filter."""
    return [
        r for r in tqdm(records, desc="Filtering", unit="sample")
        if _passes_filter(r, require_both_models=require_both_models)
    ]


def train_val_split_by_circuit_type(records, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    """
    Split records by circuit_type into train/val. Returns dict:
    { "attribute_binding": {"train": [...], "val": [...]}, ... }
    """
    rng = random.Random(seed)
    out = {}
    for ct in tqdm(CIRCUIT_TYPES, desc="Splitting by circuit type"):
        subset = [r for r in records if r.get("circuit_type") == ct]
        rng.shuffle(subset)
        n = len(subset)
        n_val = max(1, int(n * val_fraction)) if n > 0 else 0
        n_train = n - n_val
        out[ct] = {
            "train": subset[:n_train],
            "val": subset[n_train:],
        }
    return out


def flatten_splits(split_dict):
    """
    Flatten { circuit_type: { "train": [...], "val": [...] } } into a single list
    with added keys: split (train/val), circuit_type.
    """
    flat = []
    for ct, d in split_dict.items():
        for split_name, lst in d.items():
            for r in lst:
                r = dict(r)
                r["split"] = split_name
                r["circuit_type"] = ct
                flat.append(r)
    return flat
