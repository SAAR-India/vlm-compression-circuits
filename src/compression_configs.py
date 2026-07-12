from pathlib import Path

from preprocessing.config import BLIP_VQA_MODEL_ID, LLAVA15_7B_MODEL_ID, QWEN3VL_2B_MODEL_ID

# Print first N samples per dataset (input, prediction, ground truth) for format debugging
DEBUG_EVAL_SAMPLES = 5

COMP_V = "vision"
COMP_P = "projector"

METHODS = ["wanda", "awq", "sparsegpt", "gptq", "smoothquant"]
METHOD_CONFIGS = {
    "wanda": {"sparsity_ratio": 0.5, "sparsity_type": "unstructured"},
    "awq":   {"w_bit": 4, "q_group_size": 128},
    "sparsegpt": {
        "sparsity_ratio": 0.5,
        "sparsity_type": "unstructured",
        "calib_samples": 128,
        "calib_batch_size": 4,
        "block_size": 128,
        "percdamp": 0.01,
    },
    "gptq": {
        "w_bit": 4,
        "q_group_size": 128,
        "calib_samples": 128,
        "calib_batch_size": 4,
        "block_size": 128,
        "percdamp": 0.01,
        "symmetric": False,
    },
    "smoothquant": {
        "w_bit": 8,
        "a_bit": 8,
        "alpha": 0.5,
        "calib_samples": 128,
        "calib_batch_size": 4,
    },
}

QVLM_MODULE_MAP = {
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

MODULE_MAP = {
    "blip2": {
        COMP_V: "vision_model",
        COMP_P: "text_encoder",
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

QVLM_COMPONENT_COMBOS = {
    "V":   [COMP_V],
    "V+P": [COMP_V, COMP_P],
}

# Only vision and projector compression (no language decoder)
COMPONENT_COMBOS = {
    "V":   [COMP_V],
    "V+P": [COMP_V, COMP_P],
    "P":   [COMP_P]
}

MODEL_CONFIGS = {
    "blip2": {
        "model_id": BLIP_VQA_MODEL_ID,
        "model_class": "BlipForQuestionAnswering",
        "processor_class": "BlipProcessor",
    },
    "qwen3vl": {
        "model_id": QWEN3VL_2B_MODEL_ID,
        "model_class": "Qwen3VLForConditionalGeneration",
        "processor_class": "AutoProcessor",
    },
    "llava15": {
        "model_id": LLAVA15_7B_MODEL_ID,
        "model_class": "LlavaForConditionalGeneration",
        "processor_class": "AutoProcessor",
    },
}

SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = str(SRC_DIR / "compressed_models")
RESULTS_DIR = str(SRC_DIR / "eval_results")
LOG_FILE = str(SRC_DIR / "pipeline_log.json")
