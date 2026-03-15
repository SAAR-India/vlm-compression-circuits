"""
Q-VLM compression — separate script. Run from project root.

Usage:
    python src/run_qvlm_compression.py [--model blip2|qwen3vl|llava15] [--combo V|V+P]

Q-VLM must be run from its own repo: https://github.com/ChangyuanWang17/QVLM
This script generates shell scripts to run Q-VLM for vision (V) and projector (V+P)
compression only. Clone the QVLM repo and run the generated script from there.
"""

import argparse
import os
import sys
from pathlib import Path

# Align with preprocessing and main eval script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.config import BLIP2_MODEL_ID, LLAVA15_7B_MODEL_ID, QWEN3VL_2B_MODEL_ID

COMP_V = "vision"
COMP_P = "projector"

MODULE_MAP = {
    "blip2": {
        COMP_V: "vision_model",
        COMP_P: "qformer",
    },
    "qwen3vl": {
        COMP_V: "model.visual",
        COMP_P: "model.visual.merger",
    },
    "llava15": {
        COMP_V: "model.vision_tower",
        COMP_P: "model.multi_modal_projector",
    },
}

COMPONENT_COMBOS = {
    "V":   [COMP_V],
    "V+P": [COMP_V, COMP_P],
}

MODEL_CONFIGS = {
    "blip2": {"model_id": BLIP2_MODEL_ID},
    "qwen3vl": {"model_id": QWEN3VL_2B_MODEL_ID},
    "llava15": {"model_id": LLAVA15_7B_MODEL_ID},
}

QVLM_CONFIG = {"w_bit": 4, "a_bit": 4, "calib_samples": 64}
OUTPUT_BASE = "./compressed_models"


def get_module_paths(model_name: str, components: list) -> list:
    return [MODULE_MAP[model_name][c] for c in components]


def main():
    parser = argparse.ArgumentParser(description="Generate Q-VLM compression scripts (vision + projector only)")
    parser.add_argument("--model", choices=["blip2", "qwen3vl", "llava15"], default="llava15")
    parser.add_argument("--combo", choices=["V", "V+P"], default="V+P")
    parser.add_argument("--output-dir", default=OUTPUT_BASE, help="Base dir for compressed outputs")
    args = parser.parse_args()

    model_name = args.model
    comp_label = args.combo
    components = COMPONENT_COMBOS[comp_label]
    cfg = MODEL_CONFIGS[model_name]
    paths = get_module_paths(model_name, components)

    output_path = os.path.join(args.output_dir, f"{model_name}__qvlm__{comp_label}")
    os.makedirs(output_path, exist_ok=True)
    save_path_abs = str(Path(output_path).resolve())

    script = f"""#!/bin/bash
# Q-VLM for {model_name}, components: {comp_label} ({components})
# Run from the QVLM repo directory: git clone https://github.com/ChangyuanWang17/QVLM && cd QVLM && pip install -e .

cd "$(dirname "$0")/../QVLM" 2>/dev/null || cd QVLM 2>/dev/null || {{ echo "QVLM directory not found. Clone the repo first."; exit 1; }}

python quantize_vlm.py \\
    --model-path {cfg['model_id']} \\
    --w-bit {QVLM_CONFIG['w_bit']} \\
    --a-bit {QVLM_CONFIG['a_bit']} \\
    --calib-samples {QVLM_CONFIG['calib_samples']} \\
    --target-modules {','.join(paths)} \\
    --save-path {save_path_abs}
"""
    script_path = os.path.join(output_path, "run_qvlm.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    print(f"Model: {model_name}, combo: {comp_label}")
    print(f"Script: {script_path}")
    print(f"Run from project root or QVLM repo: bash {script_path}")
    if model_name == "blip2":
        print("NOTE: Q-VLM was designed for LLaVA-family; BLIP-2 may require adapting QVLM code.")


if __name__ == "__main__":
    main()
