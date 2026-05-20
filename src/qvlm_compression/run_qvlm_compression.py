#!/usr/bin/env python
"""
This script runs Q-VLM compression as a separate script.

Usage: python src/run_qvlm_compression.py [--model blip2|qwen3vl|llava15] [--combo V|V+P]
Q-VLM must be run from its own repo: https://github.com/ChangyuanWang17/QVLM
This script generates shell scripts to run Q-VLM for vision (V) and projector (V+P)
compression only. Clone the QVLM repo and run the generated script from there.
"""
import os
import sys
import argparse
from pathlib import Path

import qvlm_configs
import compression_configs

# Align with preprocessing and main eval script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def get_module_paths(model_name: str, components: list) -> list:
    return [compression_configs.QVLM_MODULE_MAP[model_name][c] for c in components]

def main():
    parser = argparse.ArgumentParser(description="Generate Q-VLM compression scripts (vision + projector only)")
    parser.add_argument("--model", choices=["blip2", "qwen3vl", "llava15"], default="llava15")
    parser.add_argument("--combo", choices=["V", "V+P"], default="V+P")
    parser.add_argument("--output-dir", default=qvlm_configs.OUTPUT_BASE, help="Base dir for compressed outputs")
    args = parser.parse_args()

    model_name = args.model
    comp_label = args.combo
    components = compression_configs.QVLM_COMPONENT_COMBOS[comp_label]
    cfg = compression_configs.MODEL_CONFIGS[model_name]
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
        --w-bit {qvlm_configs.QVLM_CONFIG['w_bit']} \\
        --a-bit {qvlm_configs.QVLM_CONFIG['a_bit']} \\
        --calib-samples {qvlm_configs.QVLM_CONFIG['calib_samples']} \\
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
