#!/bin/bash
# SVD-LLM V2 compression: whitening_hetero
# Model: meta-llama/Llama-2-7b-hf
# Ratios: 0.2 0.3 0.4 0.5 0.6
#
# Usage:
#   bash run_compress_llama2_7b_v2.sh [HF_TOKEN]
#
# NOTE: requires profiling_mat from run_compress_llama2_7b.sh (step 1) first.

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-2-7b-hf"
MODEL_TAG="Llama-2-7b"
MODEL_PREFIX="meta_llama_Llama_2_7b_hf"
SAVE_DIR="checkpoints/svdllm/llama2_7b"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

if [ ! -f "$PROF_MAT" ]; then
    echo "ERROR: profiling_mat not found: $PROF_MAT"
    echo "Run run_compress_llama2_7b.sh first to generate it."
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
        $TOKEN_ARG \
        2>&1 | tee "logs/${MODEL_TAG}_v2_${KEEP_DISPLAY}.log"
done

echo ""
echo "=== All done ==="
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_v2_${KEEP}.pt"
    [ -f "$CKPT" ] && echo "  ✓ $CKPT" || echo "  ✗ MISSING: $CKPT"
done
