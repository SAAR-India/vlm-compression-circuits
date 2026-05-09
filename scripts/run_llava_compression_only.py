"""LLaVA-1.5-7B compression (Wanda + AWQ INT4) on the V+P combo only.

Used to produce the compressed checkpoints the EAP loader expects under
src/compressed_models/. Run from src/:
    cd src && python run_llava_compression_only.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_compression_eval as r

r.MODEL_CONFIGS = {"llava15": r.MODEL_CONFIGS["llava15"]}
r.COMPONENT_COMBOS = {"V+P": r.COMPONENT_COMBOS["V+P"]}
r.run_compression(quick=False)
