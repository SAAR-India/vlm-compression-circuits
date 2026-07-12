# Toggle model choice
MODEL_CHOICE = "qwen3vl"  # "blip"  |  "qwen3vl"

BLIP_MODEL_ID       = "Salesforce/blip-vqa-base"
QWEN3VL_2B_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

COLOR_DATA_DIR = "/data/veer/testing/new/vlm_circuits/data/Visual-Counterfact/color"
SIZE_DATA_DIR  = "/data/veer/testing/new/vlm_circuits/data/Visual-Counterfact/size"
MAX_NEW_TOKENS = 20
BATCH_SIZE     = 8
IMAGE_MODE     = "original"  # "original" | "counterfact" | "both"

PRED_KEYS = [
    "qwen3vl_pred_original",
    "qwen3vl_confidence_original",
    "qwen3vl_correct_original",
    "blip_pred_original",
    "blip_confidence_original",
    "blip_correct_original",
    # keep these if present (harmless for Visual-Counterfact)
    "qwen3vl_pred_counterfact",
    "qwen3vl_confidence_counterfact",
    "qwen3vl_correct_counterfact",
    "blip_pred_counterfact",
    "blip_confidence_counterfact",
    "blip_correct_counterfact",
]
