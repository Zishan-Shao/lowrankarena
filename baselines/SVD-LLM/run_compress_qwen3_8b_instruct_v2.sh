#!/bin/bash
# SVD-LLM V2 compression: whitening_hetero
# Model: Qwen/Qwen3-8B-Instruct
# Ratios: 0.2 0.3 0.4 0.5 0.6
#
# Usage:
#   bash run_compress_qwen3_8b_instruct_v2.sh
#
# NOTE: requires profiling_mat from run_compress_qwen3_8b_instruct.sh (step 1) first.
# NOTE: --local_update is broken (see run_svdllm_v2_compress.py), not used here.

set -eo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-8B-Instruct"
MODEL_TAG="Qwen3-8B-Instruct"
MODEL_PREFIX="Qwen_Qwen3_8B_Instruct"
SAVE_DIR="checkpoints/svdllm/qwen3_8b_instruct"
SEQ_LEN=2048
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" logs

if [ ! -f "$PROF_MAT" ]; then
    echo "ERROR: profiling_mat not found: $PROF_MAT"
    echo "Run run_compress_qwen3_8b_instruct.sh first to generate it."
    exit 1
fi

for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_v2_${KEEP}.pt"

    if [ -f "$CKPT" ]; then
        echo "=== checkpoint exists, skipping: $CKPT ==="
        continue
    fi

    KEEP_DISPLAY=$(python -c "print(round(1 - $RATIO, 1))")
    echo "=== Compress V2 ratio=$RATIO (keep=$KEEP_DISPLAY) ==="
    python run_svdllm_v2_compress.py \
        --model "$MODEL" \
        --ratio $RATIO \
        --profiling_mat_path "$PROF_MAT" \
        --save_path "$SAVE_DIR" \
        --model_seq_len $SEQ_LEN \
        2>&1 | tee "logs/${MODEL_TAG}_v2_${KEEP_DISPLAY}.log"
done

echo ""
echo "=== All done ==="
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_v2_${KEEP}.pt"
    [ -f "$CKPT" ] && echo "  ✓ $CKPT" || echo "  ✗ MISSING: $CKPT"
done
