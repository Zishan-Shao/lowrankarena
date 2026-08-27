#!/usr/bin/env bash
#
# One-shot compression: a single truncation stage with no post-truncation
# quantization and no between-stage correction. This is the simplest
# pipeline configuration — useful for sanity-checking the pipeline end-to-end.
#
# Usage:
#   bash scripts/run_one_shot.sh [PRUNE_RATIO]
#
# Examples:
#   bash scripts/run_one_shot.sh 0.2
#   bash scripts/run_one_shot.sh 0.4
#   bash scripts/run_one_shot.sh 0.6
#
# To compress a different model, set MODEL:
#   MODEL=meta-llama/Llama-2-7b-hf bash scripts/run_one_shot.sh 0.2

set -euo pipefail

PRUNE_RATIO=${1:-0.2}
MODEL=${MODEL:-jeffwan/llama-7b-hf}

# keep_rank_ratio: 0.3 at 0.6 prune, 0 otherwise
if [[ "$PRUNE_RATIO" == "0.6" ]]; then
    KEEP_RANK_RATIO=0.3
else
    KEEP_RANK_RATIO=0.
fi

CUDA_VISIBLE_DEVICES=0 python main_zero_sum.py \
    --model "$MODEL" \
    --save_path . \
    --global_prune_ratio "$PRUNE_RATIO" --keep_rank_ratio "$KEEP_RANK_RATIO" \
    --num_stages 1 --nsamples_gradient_subset 1 \
    --selection_mode zero_sum --importance_seq_len 2048 \
    --sub_with_teacher_module \
    --eval_ppl --save_after_truncation
