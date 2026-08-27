#!/usr/bin/env bash
#
# One-shot compression + post-truncation quantization. Same as run_one_shot.sh
# but adds either --remap (bnb int8 row-remap on the saved low-rank factors)
# or --quantize_8_bit (8-bit weight storage) depending on the prune ratio.
#
#   prune_ratio 0.2 / 0.4  ->  --remap
#   prune_ratio 0.6        ->  --quantize_8_bit
#
# Usage:
#   bash scripts/run_remap_quant.sh [PRUNE_RATIO]
#
# Examples:
#   bash scripts/run_remap_quant.sh 0.2
#   bash scripts/run_remap_quant.sh 0.4
#   bash scripts/run_remap_quant.sh 0.6
#
# To compress a different model, set MODEL:
#   MODEL=meta-llama/Llama-2-7b-hf bash scripts/run_remap_quant.sh 0.2

set -euo pipefail

PRUNE_RATIO=${1:-0.2}
MODEL=${MODEL:-jeffwan/llama-7b-hf}

# keep_rank_ratio + quantization flag selection by prune ratio
if [[ "$PRUNE_RATIO" == "0.6" ]]; then
    KEEP_RANK_RATIO=0.3
    QUANT_FLAG=--quantize_8_bit
else
    KEEP_RANK_RATIO=0.
    QUANT_FLAG=--remap
fi

CUDA_VISIBLE_DEVICES=0 python main_zero_sum.py \
    --model "$MODEL" \
    --save_path . \
    --global_prune_ratio "$PRUNE_RATIO" --keep_rank_ratio "$KEEP_RANK_RATIO" \
    --num_stages 1 --nsamples_gradient_subset 1 \
    --selection_mode zero_sum --importance_seq_len 2048 \
    --sub_with_teacher_module \
    --eval_ppl --save_after_truncation "$QUANT_FLAG"
