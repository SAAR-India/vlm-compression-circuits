from pathlib import Path

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"
COUNTERFACT_SELECTED_DIR = OUTPUT_DIR / "counterfactual_selected"
SRC_DIR = REPO_ROOT / "src"
COMPRESSED_MODELS_DIR = SRC_DIR / "compressed_models"

# HuggingFace dataset (Visual-Counterfact: color/size splits, VQA)
HF_DATASET_ID = "mgolov/Visual-Counterfact"
# HuggingFace repo for compressed models (subdirs: blip2__wanda__V, blip2__awq__P, ...)
HF_COMPRESSED_MODELS_REPO = "vlm_circuits/compressed_models"

# Train/val split for crosscoder
VAL_FRACTION = 0.2
RANDOM_SEED = 42