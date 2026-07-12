from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SRC_DIR = PROJECT_ROOT / "src"
COMPRESSED_MODELS_DIR = SRC_DIR / "compressed_models"
CROSSCODER_RESULTS_DIR = SRC_DIR / "crosscoder" / "results"

# Filtered Visual-Counterfact dataset (from preprocessing) - used for all activation processing
VISUAL_COUNTERFACT_DIR = OUTPUT_DIR / "counterfactual_selected"
METADATA_CSV = OUTPUT_DIR / "counterfactual_selected_metadata.csv"

SEED = 42
LEARNING_RATE = 3e-4
WARMUP_FRACTION = 0.05
BATCH_SIZE = 32
EXTRACT_BATCH_SIZE = 128
NUM_EPOCHS = 200
CHECKPOINT_EVERY = 100

LAMBDA_SPARSITY = 1e-3
LAMBDA_CROSS = 0.4
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-5
LAMBDA_SHARED_MULTIPLIER = 0.05
FORCED_SHARED_FRACTION = 0.06

TOPK_CLS = 400
TOPK_PATCH = 400
TOPK_PROJECTOR = 400

EXPANSION_FACTOR_VISION = 4
EXPANSION_FACTOR_PROJECTOR = 4

FVE_THRESHOLD = 0.5
DEAD_NEURON_THRESHOLD = 1

# GMM fallback when rho distribution is degenerate; theta thresholds (fixed)
RHO_UNCOMPRESSED_ONLY = 0.15
RHO_COMPRESSED_ONLY = 0.85
RHO_SHARED_LOW = 0.35
RHO_SHARED_HIGH = 0.65
THETA_ALIGNED = 0.80
THETA_REDIRECTED = 0.50

SUPERPOSITION_R2_THRESHOLD = 0.8
SUPERPOSITION_MAX_CONSTITUENTS = 50
SUPERPOSITION_TOP_SAMPLES = 100

AWQ_CALIBRATION_SAMPLES = 256

BLIP_VQA_MODEL_ID = "Salesforce/blip-vqa-base"
QWEN3VL_2B_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
LLAVA_V1_5_7B_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

MODULE_MAP = {
    "blip2": {
        "vision": "vision_model",
        "projector": "text_encoder",
    },
    "qwen3vl": {
        "vision": "model.visual",
        "projector": "model.visual.merger",
    },
    "llava15": {
        "vision": "model.vision_tower",
        "projector": "model.multi_modal_projector",
    },
}

VISION_HOOK_PATHS = {
    "blip2": "vision_model.encoder.layers",
    "qwen3vl": "model.visual.blocks",
    "llava15": "model.vision_tower.vision_model.encoder.layers",
}

PROJECTOR_HOOK_PATHS = {
    "blip2": "text_encoder.encoder.layer",
    "qwen3vl": "model.visual.merger",
    "llava15": "model.multi_modal_projector",
}

BLIP_CROSS_ATTENTION_LAYERS = [9, 10, 11]

ACTIVATION_DIM = {
    "blip2": {"vision": 768, "projector": 768},
    "qwen3vl": {"vision": 1024, "projector": 2048},
    "llava15": {"vision": 1024, "projector": 4096},
}

MODELS = ["blip2", "qwen3vl", "llava15"]
METHODS = ["wanda", "awq"]
COMPONENTS = ["V", "P", "V_P"]
TOKEN_TYPES = ["cls", "patch"]
