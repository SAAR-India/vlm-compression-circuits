"""Configuration for activation patching pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
SRC_DIR = PROJECT_ROOT / "src"
COMPRESSED_MODELS_DIR = SRC_DIR / "compressed_models"
DATA_DIR = PROJECT_ROOT / "data"

# Filtered Visual-Counterfact dataset (output/counterfactual_selected)
VISUAL_COUNTERFACT_DIR = OUTPUT_DIR / "counterfactual_selected"
METADATA_CSV = OUTPUT_DIR / "counterfactual_selected_metadata.csv"

# Original Visual-Counterfact for incorrect_answer lookup (data/ or HF)
VISUAL_COUNTERFACT_SOURCE_DIR = DATA_DIR / "Visual-Counterfact"
HF_VISUAL_COUNTERFACT_ID = "mgolov/Visual-Counterfact"

# Model IDs
BLIP_VQA_MODEL_ID = "Salesforce/blip-vqa-base"
QWEN3VL_2B_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
LLAVA_V1_5_7B_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# Patching results output
PATCHING_RESULTS_DIR = SRC_DIR / "activation_patching" / "results"

# Models and methods for patching sweep
PATCHING_MODELS = ["blip2", "qwen3vl", "llava"]
PATCHING_METHODS = ["baseline", "wanda", "awq"]
PATCHING_COMPONENTS = ["P", "V_P"]

# GPU optimization
PATCH_BATCH_SIZE = 16  # Token patches per forward (reduces from L*3*T to L*3*ceil(T/B) forwards)
USE_AMP = True  # Mixed precision (fp16/bf16) for faster GPU forward passes
