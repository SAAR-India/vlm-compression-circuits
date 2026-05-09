# EAP for LLaVA-1.5-7B — baseline vs AWQ INT4 (V+P)

Edge Activation Patching results on the Visual-Counterfact dataset. n=50,
patch_batch_size=64, run on 8× NVIDIA B200 with the sample-sharding pipeline
from `scripts/run_phase.sh` + `scripts/merge_shards.py`.

## Layout

```
eap_llava_n50/
├── llava__baseline/                  # canonical merged outputs (50 samples)
│   ├── patching_data.npz             # {mlp, attn, xattn, token_labels}
│   ├── metrics/patching_metrics.json
│   └── plots/{notice_heatmap,layer_importance,token_importance}.png
├── llava__awq/                       # canonical merged outputs (50 samples)
│   ├── patching_data.npz
│   ├── metrics/patching_metrics.json
│   ├── metrics/comparison_metrics.json   ← awq vs baseline
│   └── plots/...
├── llava__baseline__shard{0..7}/     # per-GPU shards w/ raw_sample_results.pkl
└── llava__awq__shard{0..7}/          # per-GPU shards w/ raw_sample_results.pkl
```

## Headline numbers (awq vs baseline, n=50)

| component | jaccard | spearman ρ | stability |
|---|---|---|---|
| MLP            | 0.684 | 0.786 (p≈1e-7) | 0.750 |
| self_attention | 0.684 | 0.746 (p≈1e-6) | 0.750 |
| **aggregate**  | **0.684** | **0.766** | **0.750** |

`metrics/comparison_metrics.json` in the awq dir has full per-component breakdown.

## Reproducing the figures

The plots in each `plots/` dir were saved during the merge step. To regenerate
from `patching_data.npz` only:

```bash
python experiments/eap_llava_n50/regenerate_plots.py
```

To re-run the full aggregation from per-shard raw data (e.g. after changing
`_aggregate_by_semantic_group`):

```bash
# point patching_config.PATCHING_RESULTS_DIR at this directory, or symlink
# the shard dirs into src/activation_patching/results/, then:
python scripts/merge_shards.py --variants baseline awq --with_compare
```

## Reproducing the experiment from scratch

1. Compress the V+P combo of LLaVA-1.5-7B (Wanda + AWQ INT4):
   ```bash
   cd src && python ../scripts/run_llava_compression_only.py && cd ..
   python scripts/rename_to_eap.py
   ```
2. Run all 8-way shards for one variant per phase, then merge:
   ```bash
   HF_HUB_OFFLINE=1 ./scripts/run_phase.sh baseline 50 64
   HF_HUB_OFFLINE=1 ./scripts/run_phase.sh awq 50 64
   python scripts/merge_shards.py --variants baseline awq --with_compare
   ```

Per-sample wall time on B200 with the cached-tokenization patch: ~510 s at
patch_batch_size=64. n=50 split 8-ways → ~50 min per variant phase.
