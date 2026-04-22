#!/usr/bin/env bash
set -euo pipefail

REPO="/deac/csc/yangGrp/cuij/LLM/TransformerCompression"
MODEL_ID="meta-llama/Llama-3.1-8B"
MODEL_PATH="/deac/csc/yangGrp/cuij/LLM/models/hf_models/meta-llama__Llama-3.1-8B"
RESULT_ROOT="$REPO/results/formal_l31_8b"
LOG_ROOT="$REPO/logs/slurm"
RATIOS=(0.8 0.7 0.6 0.5 0.4)

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

for ratio in "${RATIOS[@]}"; do
  ratio_tag="${ratio/./}"
  out_root="$RESULT_ROOT/keep_${ratio}"
  jobid=$(sbatch --parsable \
    -p gpu_small \
    --gres=gpu:H200_141:1 \
    --cpus-per-task=8 \
    --mem=96G \
    --time=24:00:00 \
    -J "slicegpt_l31_k${ratio_tag}" \
    -o "$LOG_ROOT/slicegpt_l31_k${ratio_tag}_%j.out" \
    -e "$LOG_ROOT/slicegpt_l31_k${ratio_tag}_%j.err" \
    --wrap "$REPO/scripts/slurm/run_slicegpt_ratio.sh '$MODEL_ID' '$MODEL_PATH' '$ratio' '$out_root'")
  echo "keep_ratio=$ratio jobid=$jobid out_root=$out_root"
done
