"""Rename the compression-script outputs to the names the EAP loader expects.

run_compression_eval writes src/compressed_models/llava15__{method}__V_P/, but
crosscoder.utils.get_compressed_model_path keys on model="llava". This script
moves (not copies) each dir into place.
"""
from pathlib import Path
import shutil

base = Path("src/compressed_models")
for method in ["wanda", "awq"]:
    src = base / f"llava15__{method}__V_P"
    dst = base / f"llava__{method}__V_P"
    if not src.exists():
        print(f"SKIP {src}: source missing")
        continue
    if dst.exists():
        shutil.rmtree(dst)
    src.rename(dst)
    print("ready:", dst)
