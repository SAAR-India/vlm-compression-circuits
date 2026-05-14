Before running this benchmark, you need to:

(1) Download EN_part and CN_part into the ocr_bench/ directory.
(2) Install the dependencies found inside `ocr_bench/requirements.txt`
(3) Make sure the compressed BLIP-VQA checkpoint already exists.

To run this eval, you can use the following:

```py
python src/ocr_bench/run_blip_vqa_ocrbench.py `
  --compressed-model-path compressed_models/blip2__awq__V_P `
  --output-dir results/ocrbench_blip_vqa_awq_V_P `
  --batch-size 8
```
