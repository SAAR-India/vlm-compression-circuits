#!/usr/bin/env bash
# Launch 8 EAP shards for one variant on GPUs 0-7, wait for all to finish.
#
# Usage (from repo root):
#   HF_HUB_OFFLINE=1 ./scripts/run_phase.sh <variant> <num_samples> <patch_batch_size>
#
# After all 3 phases:
#   python scripts/merge_shards.py --variants baseline wanda awq --with_compare
set -u
variant=$1
n=$2
bs=$3

cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p shard_logs
log() { echo "[phase $variant] $*"; }

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  start=$(( gpu * n / 8 ))
  end=$(( (gpu + 1) * n / 8 ))
  if [ "$start" -ge "$end" ]; then
    log "GPU $gpu: empty shard, skipping"
    continue
  fi
  log "GPU $gpu: samples [$start, $end)"
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "
    set -o pipefail
    python -u -m src.activation_patching.main --model llava --variant '$variant' \\
      --num_samples '$n' --patch_batch_size '$bs' \\
      --sample_start '$start' --sample_end '$end' \\
      --output_suffix 'shard${gpu}' \\
      2>&1 | tee shard_logs/'${variant}_gpu${gpu}.log'
    echo EXIT=\${PIPESTATUS[0]} >> shard_logs/'${variant}_gpu${gpu}.log'
  " > /dev/null 2>&1 &
  pids+=($!)
done

log "${#pids[@]} shards launched (pids: ${pids[*]})"
for pid in "${pids[@]}"; do
  wait "$pid" || log "shard pid $pid exited non-zero"
done
log "all shards done"

for f in shard_logs/${variant}_gpu*.log; do
  tail -1 "$f"
done
