import sys 
import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BLIP_VQA_MODEL_ID = "Salesforce/blip-vqa-base"
LLAVA15_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
QWEN3VL_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_OCRBENCH_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "ocrbench_blip_vqa"

MODEL_CONFIGS = {
    "blip": {
        "display_name": "BLIP-VQA",
        "default_baseline": BLIP_VQA_MODEL_ID,
        "default_compressed_arg": "compressed_model_path",
    },
    "llava": {
        "display_name": "LLaVA-1.5",
        "default_baseline": LLAVA15_MODEL_ID,
        "default_compressed_arg": "llava_compressed_model_path",
    },
    "qwen": {
        "display_name": "Qwen3-VL",
        "default_baseline": QWEN3VL_MODEL_ID,
        "default_compressed_arg": "qwen_compressed_model_path",
    },
}


EN_CATEGORY_TYPES = {
    "text_recognition": {
        "text recognition en",
        "fine-grained text recognition en",
        "full-page OCR en",
    },
    "text_detection": {
        "text grounding en",
        "VQA with position en",
    },
    "text_spotting": {
        "text spotting en",
    },
    "relationship_extraction": {
        "key information extraction en",
        "key information mapping en",
    },
    "element_parsing": {
        "document parsing en",
        "chart parsing en",
        "table parsing en",
        "formula recognition en",
    },
    "mathematical_calculation": {
        "math QA en",
        "text counting en",
    },
    "visual_text_understanding": {
        "document classification en",
        "cognition VQA en",
        "diagram QA en",
    },
    "knowledge_reasoning": {
        "reasoning VQA en",
        "science QA en",
        "APP agent en",
        "ASCII art classification en",
    },
}

CN_CATEGORY_TYPES = {
    "text_recognition": {
        "full-page OCR cn",
    },
    "relationship_extraction": {
        "key information extraction cn",
        "handwritten answer extraction cn",
    },
    "element_parsing": {
        "document parsing cn",
        "table parsing cn",
        "formula recognition cn",
    },
    "visual_text_understanding": {
        "cognition VQA cn",
    },
    "knowledge_reasoning": {
        "reasoning VQA cn",
        "text translation cn",
    },
}