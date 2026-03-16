# Activation Patching Pipeline

Edge activation patching and NOTICE-style analysis for VLMs on the Visual-Counterfact dataset.

## Methodology

Based on:
- **ACDC** / **Edge Pruning** (NeurIPS 2024): Activation patching for circuit discovery; patch clean activations into corrupt run, measure metric recovery.
- **NOTICE** (NAACL 2025): Per-layer, per-token, per-component heatmaps for VLM mechanistic interpretability.

## Dataset

Uses the filtered Visual-Counterfact at `output/counterfactual_selected`:
- `attribute_binding_train` / `attribute_binding_val`
- Pairs: `image_original` (correct) vs `image_counterfact` (wrong attribute)
- VQA: "What color is X?" or "Which object appears larger?"
- `incorrect_answer` from original Visual-Counterfact (data/ or HF)

## Models

| Model | Status | Components |
|-------|--------|------------|
| **blip2** (blip-vqa-base) | Implemented | Q-Former: MLP, Self-Attention, Cross-Attention |
| **qwen3vl** (Qwen3-VL-2B) | Implemented | LLM layers: MLP, Self-Attention |
| **llava** (LLaVA-1.5-7B) | Implemented | LLM layers: MLP, Self-Attention |

## Usage

```bash
source .venv/bin/activate

# BLIP baseline (uncompressed)
python -m src.activation_patching.main --model blip2 --variant baseline --num_samples 100

# BLIP + all variants (baseline, wanda, awq) + comparison metrics
python -m src.activation_patching.main --model blip2 --all_variants --num_samples 100

# GPU optimizations (default: patch_batch_size=16, AMP on for CUDA)
python -m src.activation_patching.main --model blip2 --num_samples 100 --patch_batch_size 32  # more tokens per forward
python -m src.activation_patching.main --model blip2 --num_samples 100 --no_amp              # disable mixed precision (FP32)

# Output: src/activation_patching/results/blip2__{variant}/
#   metrics/patching_metrics.json   # per-layer, per-component
#   metrics/comparison_metrics.json # Jaccard, Spearman, stability (vs baseline)
#   plots/notice_heatmap.png        # NOTICE-style heatmaps
#   plots/layer_importance.png
#   plots/token_importance.png

# GPU optimization (reduce total runtime)
python -m src.activation_patching.main --model blip2 --num_samples 100 --patch_batch_size 32  # more tokens per forward
python -m src.activation_patching.main --model blip2 --num_samples 100 --no_amp               # disable mixed precision if unstable

# LLaVA-1.5-7B (baseline; wanda/awq require compressed checkpoints)
python -m src.activation_patching.main --model llava --variant baseline --num_samples 100

# Qwen3-VL
python -m src.activation_patching.main --model qwen3vl --variant baseline --num_samples 100
```

## GPU optimization

Activation patching uses several GPU optimizations to reduce runtime:

| Optimization | Config | CLI | Effect |
|--------------|--------|-----|--------|
| **Token batching** | `PATCH_BATCH_SIZE` (default 16) | `--patch_batch_size N` | Reduces forward passes from L×3×T to L×3×⌈T/B⌉ per sample |
| **Mixed precision (AMP)** | `USE_AMP` (default True) | `--no_amp` to disable | FP16 forward passes for faster GPU compute |
| **Merged clean forward** | — | — | Single forward captures both logits and activations (avoids duplicate clean run) |

Tune `--patch_batch_size` (e.g. 8–32) based on GPU memory; larger batches = fewer forwards but higher memory.

## Metrics

- **Per-layer, per-component**: Sum of logit-diff recovery over tokens (MLP, Self-Attention, Cross-Attention)
- **Jaccard**: Overlap of important layers between compressed and uncompressed
- **Spearman rho**: Rank correlation of layer importance
- **Stability**: Fraction of top-k layers preserved under compression

## Plots (NOTICE-style)

- **Heatmaps**: Per-layer × per-token, one panel per component (MLP, Self-Attention, Cross-Attention)
- **Layer importance**: Bar chart of total Δ logit diff per layer
- **Token importance**: Line plot of total Δ logit diff per token position
