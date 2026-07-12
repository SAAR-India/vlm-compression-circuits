# OCRBench_v2

**Disclaimer**: This code is fully functional, but we have decided not to include the results in the manuscript. This is due to the fact that for both BLIP-VQA and LLaVA-1.5-7B the benchmark performance is lackluster. This is a direct result of the models not being trained on OCR tasks. All of the evaluations included in this directory, specifically under the `eval_scripts/` directory come from the original [OCRBench_v2 repository](https://github.com/Yuliang-Liu/MultimodalOCR/blob/main/OCRBench_v2/README.md).

## Results

| Model                               | English Overall Score | Chinese Overall Score |
| ----------------------------------- | --------------------- | --------------------- |
| BLIP-VQA (base)                     | 0.038                 | 0.018                 |
| BLIP-VQA (compressed with INT4)     | 0.034                 | 0.018                 |
| LLaVA-1.5-7B (base)                 | 0.133                 | 0.082                 |
| LLaVA-1.5-7B (compressed with INT4) | 0.131                 | 0.077                 |
|                                     |

Notably, there was not much of a performance drop from the base models. More details of the scores (such as specific task scores) can be provided upon request. If you want to replicate the results, you can follow the steps below.

## Set Up

Before running this benchmark, you need to:

(1) Download `EN_part` and `CN_part` into **`src/ocr_bench/`** (alongside `OCRBench_v2.json`).
(2) **Install OCRBench scorer dependencies** from `src/ocr_bench/requirements.txt`. Inference only uses PyTorch/transformers; **official scoring** needs extra packages (`distance`, `apted`, `zss`, `nltk`, `jieba`, etc.). Missing these fails _after_ prediction with a `ModuleNotFoundError`.

```py
# Same Python / same Colab runtime as you use to run the script:
python -m pip install -r src/ocr_bench/requirements.txt

# In Colab notebooks, prefer:
%pip install -r src/ocr_bench/requirements.txt
```

(3) Ensure the compressed BLIP-VQA checkpoint already exists under `compressed_models/...`.

To run this eval:

```powershell
python src/ocr_bench/run_blip_vqa_ocrbench.py `
  --compressed-model-path compressed_models/blip2__awq__V_P `
  --output-dir results/ocrbench_blip_vqa_awq_V_P `
  --batch-size 8
```

On Linux/macOS shells, replace line breaks with `\` instead of Powershell `` ` ``.
