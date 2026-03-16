"""Configuration for counterfactual dataset construction (Section 3.1 only)."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source dataset paths
VISUAL_COUNTERFACT_DIR = DATA_DIR / "Visual-Counterfact"
COCO_COUNTERFACTUALS_DIR = DATA_DIR / "COCO-Counterfactuals"
XAITK_DIR = DATA_DIR / "XAITK"

# Quality filter (Phase 2): keep only where base model is correct on original with confidence > threshold
CONFIDENCE_THRESHOLD = 0.80

# Circuit types (Section 3.1 + 4.2)
# Attribute binding: Visual-Counterfact color/size
# Object recognition: XAITK
# Answer generation (image captioning): COCO-Counterfactuals
CIRCUIT_TYPE_ATTRIBUTE_BINDING = "attribute_binding"
CIRCUIT_TYPE_OBJECT_RECOGNITION = "object_recognition"
CIRCUIT_TYPE_ANSWER_GENERATION = "answer_generation"

CIRCUIT_TYPES = [
    CIRCUIT_TYPE_ATTRIBUTE_BINDING,
    CIRCUIT_TYPE_OBJECT_RECOGNITION,
    CIRCUIT_TYPE_ANSWER_GENERATION,
]

# Val fraction per circuit type
VAL_FRACTION = 0.2
RANDOM_SEED = 42

# Model IDs for inference (Section 1)
# Qwen3-VL-2B Instruct: current default lightweight multimodal model.
QWEN3VL_2B_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
# LLaVA-1.5-7B: optional larger model for compatibility experiments.
LLAVA15_7B_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"

# Dataset-specific inference models
# Visual-Counterfact is a VQA task: use a VQA-trained BLIP checkpoint.
BLIP_VQA_MODEL_ID = "Salesforce/blip-vqa-base"

# Backward compatibility aliases (kept mapped to Qwen3-VL for existing scripts).
TINYLLAVA_MODEL_ID = QWEN3VL_2B_MODEL_ID
TINYLLAVA_V1_MODEL_ID = QWEN3VL_2B_MODEL_ID

# HuggingFace dataset
HF_DATASET_ID = "vlm_circuits/counterfactual_unified"  # or your org/repo name
