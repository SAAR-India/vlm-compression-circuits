Before running this benchmark, you need to:

(1) Download `EN_part` and `CN_part` into **`src/ocr_bench/`** (alongside `OCRBench_v2.json`).
(2) **Install OCRBench scorer dependencies** from `src/ocr_bench/requirements.txt`. Inference only uses PyTorch/transformers; **official scoring** needs extra packages (`distance`, `apted`, `zss`, `nltk`, `jieba`, etc.). Missing these fails *after* prediction with `ModuleNotFoundError`.

    Same Python / same Colab runtime as you use to run the script:

        python -m pip install -r src/ocr_bench/requirements.txt

    In Colab notebooks, prefer:

        %pip install -r src/ocr_bench/requirements.txt

(3) Ensure the compressed BLIP-VQA checkpoint already exists under `compressed_models/...`.

To run this eval:

```powershell
python src/ocr_bench/run_blip_vqa_ocrbench.py `
  --compressed-model-path compressed_models/blip2__awq__V_P `
  --output-dir results/ocrbench_blip_vqa_awq_V_P `
  --batch-size 8
```

On Linux/macOS shells, replace line breaks with `\` instead of Powershell `` ` ``.
