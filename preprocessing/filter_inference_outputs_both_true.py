"""
Filter `output/inference_outputs.jsonl` to keep only samples where both models are correct.

Writes a new JSONL (default: `output/inference_outputs_both_true.jsonl`) containing only
the selected records (same JSON payloads as the input file).
"""

from __future__ import annotations

import json
from pathlib import Path


def _is_true(x) -> bool:
    if x is True:
        return True
    if isinstance(x, str):
        return x.strip().lower() == "true"
    return False


def filter_jsonl(in_path: str | Path, out_path: str | Path,
                require_fields: bool = True) -> tuple[int, int]:
    """Returns (kept, total). If require_fields is True, drop records missing correctness"""
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)

            t = obj.get("llava15_correct_original",
                        obj.get("qwen3vl_correct_original",
                                obj.get("tinyllava_correct_original", None)))
            b = obj.get("blip_correct_original", None)
            if require_fields and (t is None or b is None):
                continue

            if _is_true(t) and _is_true(b):
                fout.write(json.dumps(obj) + "\n")
                kept += 1

    return kept, total


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    in_path = root / "output" / "inference_outputs.jsonl"
    out_path = root / "output" / "inference_outputs_both_true.jsonl"

    kept, total = filter_jsonl(in_path, out_path)
    print(f"Filtered {kept}/{total} records -> {out_path}")


if __name__ == "__main__":
    main()

