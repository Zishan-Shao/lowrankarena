#!/usr/bin/env bash
#
# Multi-stage compression: 6 outer truncation stages with the pull-subspace
# correction applied between stages. No --remap and no --quantize_8_bit
# (i.e. the saved checkpoint stores the raw low-rank factors in fp16).
#
# This configuration is required for any model other than jeffwan/llama-7b-hf
# and generally yields better PPL than the one-shot baseline at the same
# prune ratio.
#
# Usage:
#   bash scripts/run_multistage.sh [PRUNE_RATIO]
#
# Examples:
#   bash scripts/run_multistage.sh 0.2
#   bash scripts/run_multistage.sh 0.4
#   bash scripts/run_multistage.sh 0.6
#
# To compress a different model, set MODEL:
#   MODEL=meta-llama/Llama-2-7b-hf bash scripts/run_multistage.sh 0.2

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
    --num_stages 6 --nsamples_gradient_subset 1 \
    --selection_mode zero_sum --importance_seq_len 2048 \
    --sub_with_teacher_module \
    --pull_subspace --nsamples_subspace_proj 1 --subspace_proj_seq_len 1024 \
    --random_samples_for_pullsubspace \
    --eval_ppl --eval_ppl_per_outer_stage --save_after_truncation
