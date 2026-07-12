# Mechanistically Interpreting Compression in Vision-Language Models

Repository containing the code for the paper "[Mechanistically Interpreting Compression in Vision-Language Models](https://arxiv.org/abs/2603.25035)".

## Citation

If you use any part of this code in your research, please cite our paper:

```bibtex
@article{elluru2026mechanistically,
  title={Mechanistically Interpreting Compression in Vision-Language Models},
  author={Elluru, Veeraraju and Singh, Arth and Aguero, Roberto and Agarwal, Ajay and Das, Debojyoti and Paul, Hreetam},
  journal={arXiv preprint arXiv:2603.25035},
  year={2026}
}
```

## 1. Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Run all commands from the repository root. A CUDA-capable GPU is strongly recommended.

### Models

The scripts download the base models automatically from Hugging Face:

- `Salesforce/blip-vqa-base`
- `Qwen/Qwen3-VL-2B-Instruct`
- `llava-hf/llava-1.5-7b-hf`

Compressed checkpoints are created in `src/compressed_models/` and loaded from there by the evaluation and analysis scripts.

## 2. Prepare the Dataset

Download and prepare Visual-Counterfact for the mechanistic analyses:

```bash
python preprocessing/setup_crosscoder_from_hf.py --dataset-only
```

The processed dataset is saved to `output/counterfactual_selected/`.

## 3. Compress and Evaluate

Download, compress, and evaluate the models:

```bash
python src/run_compression_eval.py --stage all
```

Use `--quick` for a smaller test run.

## 4. Run Mechanistic Analyses

After generating the compressed checkpoints, run the cross-coder analysis:

```bash
python -m src.crosscoder.main --model blip2 --method wanda --component V --token_type cls --stage all
```

Run activation patching:

```bash
python -m src.activation_patching.main --model blip2 --all_variants --num_samples 100
```

Evaluation, cross-coder, and activation-patching outputs are saved under their respective directories in `src/`.
