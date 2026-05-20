"""
VQA on the Visual-Counterfact Dataset — BLIP VQA vs Qwen3-VL-2B toggle
Dataset: https://huggingface.co/datasets/mgolov/Visual-Counterfact
Models: Salesforce/blip-vqa-base   (BlipForQuestionAnswering + BlipProcessor)
          Qwen/Qwen3-VL-2B-Instruct  (Qwen3VLForConditionalGeneration + AutoProcessor)
Install : pip install transformers accelerate datasets Pillow torch
"""
import ast

import torch
from datasets import load_from_disk

import preprocess_configs

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"Using device  : {device}")
print(f"Model choice  : {preprocess_configs.MODEL_CHOICE}")

if preprocess_configs.MODEL_CHOICE == "blip":
    from transformers import BlipProcessor, BlipForQuestionAnswering

    print(f"Loading {preprocess_configs.BLIP_MODEL_ID} ...")
    blip_processor = BlipProcessor.from_pretrained(preprocess_configs.BLIP_MODEL_ID)
    blip_model = BlipForQuestionAnswering.from_pretrained(preprocess_configs.BLIP_MODEL_ID, torch_dtype=dtype).to(device)
    blip_model.eval()
    print("Model loaded.\n")
elif preprocess_configs.MODEL_CHOICE == "qwen3vl":
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"Loading {preprocess_configs.QWEN3VL_2B_MODEL_ID} ...")
    qwen3vl_processor = AutoProcessor.from_pretrained(preprocess_configs.QWEN3VL_2B_MODEL_ID)
    qwen3vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
        preprocess_configs.QWEN3VL_2B_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    qwen3vl_model.eval()
    print("Model loaded.\n")
else:
    raise ValueError(f"Unknown MODEL_CHOICE='{preprocess_configs.MODEL_CHOICE}'. Use 'blip' or 'qwen3vl'.")

# Question Builder Functions
def build_color_question(obj: str) -> str:
    if preprocess_configs.MODEL_CHOICE == "blip":
        return f"What color is the {obj}?"
    else:
        return f"USER: <image>\nWhat color is the {obj}?\nASSISTANT:"

def build_size_question(obj: str) -> str:
    if preprocess_configs.MODEL_CHOICE == "blip":
        return "Which object appears larger in the image?"
    else:
        return "USER: <image>\nWhich object appears larger in the image?\nASSISTANT:"


# Inference 
def run_single_blip(image, question: str) -> str:
    inputs = blip_processor(images=image, text=question, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        ids = blip_model.generate(**inputs, max_new_tokens=preprocess_configs.MAX_NEW_TOKENS)
    return blip_processor.decode(ids[0], skip_special_tokens=True).strip()


def run_single_qwen3vl(image, question: str) -> str:
    """
    Run VQA for Qwen3-VL-2B using the processor's apply_chat_template with
    tokenize=True so it returns full inputs (input_ids + pixel_values, etc.)
    with image tokens correctly inserted.
    """
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    # Qwen3-VL: apply_chat_template(..., tokenize=True, return_dict=True) does full processing.
    messages = [{"role": "user", "content": 
                [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
                ]}]
    inputs = qwen3vl_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    for k, v in inputs.items():
        if hasattr(v, "to"):
            if v.is_floating_point():
                inputs[k] = v.to(device=device, dtype=dtype)
            else:
                inputs[k] = v.to(device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = qwen3vl_model.generate(
            **inputs,
            max_new_tokens=preprocess_configs.MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Trim to generated tokens only (per official Qwen3-VL example)
    new_ids = generated_ids[0, prompt_len:]
    return qwen3vl_processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_batch(images: list, questions: list) -> list:
    runner = run_single_blip if preprocess_configs.MODEL_CHOICE == "blip" else run_single_qwen3vl
    return [runner(img, q) for img, q in zip(images, questions)]

# Dataset loading
print("Loading datasets from disk ...")
color_ds = load_from_disk(preprocess_configs.COLOR_DATA_DIR)
size_ds  = load_from_disk(preprocess_configs.SIZE_DATA_DIR)
print(f"  color split : {len(color_ds)} rows  |  columns: {color_ds.column_names}")
print(f"  size  split : {len(size_ds)} rows   |  columns: {size_ds.column_names}\n")


def parse_correct_answer(raw) -> list:
    """Parse correct_answer (list or string like \"['green']\" or \"tree\") into list of lowercase strings."""
    if isinstance(raw, list):
        return [str(x).lower() for x in raw]
    s = str(raw).strip()
    if s.startswith("["):
        parsed = ast.literal_eval(s)
        return [str(x).lower() for x in parsed]
    return [s.lower()]


def is_correct(prediction: str, gold_answers: list) -> bool:
    pred = prediction.lower().strip()
    if not pred:
        return False
    return any(gold in pred for gold in gold_answers)


# Evals
def evaluate_split(dataset, question_fn, split_name: str, image_mode: str = preprocess_configs.IMAGE_MODE):
    image_keys = []
    if image_mode in ("original", "both"):
        image_keys.append("original_image")
    if image_mode in ("counterfact", "both"):
        image_keys.append("counterfact_image")

    results = {key: [] for key in image_keys}

    for i in range(0, len(dataset), preprocess_configs.BATCH_SIZE):
        batch     = dataset[i : i + preprocess_configs.BATCH_SIZE]
        objects   = batch["object"]
        raw_golds = batch["correct_answer"]
        questions = [question_fn(obj) for obj in objects]

        for img_key in image_keys:
            images = batch[img_key]
            preds  = run_batch(images, questions)

            for obj, pred, raw_gold in zip(objects, preds, raw_golds):
                gold_list = parse_correct_answer(raw_gold)
                results[img_key].append({
                    "object":    obj,
                    "question":  question_fn(obj),
                    "predicted": pred,
                    "gold":      gold_list,
                    "correct":   is_correct(pred, gold_list),
                })

        if (i // preprocess_configs.BATCH_SIZE) % 10 == 0:
            done = min(i + preprocess_configs.BATCH_SIZE, len(dataset))
            print(f"  [{split_name}] {done}/{len(dataset)} samples processed ...")

    for img_key, rows in results.items():
        n_correct = sum(r["correct"] for r in rows)
        n_empty   = sum(1 for r in rows if not r["predicted"])
        accuracy  = n_correct / len(rows) if rows else 0
        label     = "Original" if img_key == "original_image" else "Counterfactual"

        print(f"\n  [{split_name} | {label}]  Accuracy: {accuracy:.2%}  "
              f"({n_correct}/{len(rows)})  |  Empty predictions: {n_empty}")

        print("  Sample predictions:")
        for r in rows[:5]:
            mark         = "✓" if r["correct"] else "✗"
            pred_display = f'"{r["predicted"]}"' if r["predicted"] else "<empty>"
            print(f"    [{mark}] {r['object']}")
            print(f"         Gold : {r['gold']}")
            print(f"         Pred : {pred_display}")

    return results


# Single sample
print("── Single-sample demo (color split, original image) ──")
sample    = color_ds[0]
obj       = sample["object"]
question  = build_color_question(obj)
pred      = run_batch([sample["original_image"]], [question])[0]
gold_list = parse_correct_answer(sample["correct_answer"])
print(f"  Object  : {obj}")
print(f"  Q       : {question}")
print(f"  Gold    : {gold_list}")
print(f"  Pred    : \"{pred}\"")
print(f"  Correct : {is_correct(pred, gold_list)}\n")


# Full eval
print("── Evaluating COLOR split ──")
color_results = evaluate_split(color_ds, build_color_question, split_name="color")

print("\n── Evaluating SIZE split ──")
size_results = evaluate_split(size_ds, build_size_question, split_name="size")