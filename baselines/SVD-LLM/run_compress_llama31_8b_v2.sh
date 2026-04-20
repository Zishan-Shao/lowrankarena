#!/bin/bash
# SVD-LLM V2 heterogeneous compression: adaptive rank allocation via SVDLLM_v2_hetero
# Model: meta-llama/Llama-3.1-8B
# Ratios: 0.2 0.3 0.4 0.5 0.6
#
# Usage:
#   bash run_compress_llama31_8b_v2.sh [HF_TOKEN]

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG="Llama-3.1-8B"
MODEL_PREFIX="meta_llama_Llama_3.1_8B"
SAVE_DIR="checkpoints/svdllm/llama31_8b"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_wikitext2_256_0.pt"
SUFFIX="v2hetero"

mkdir -p "$SAVE_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

if [ ! -f "$PROF_MAT" ]; then
    echo "ERROR: profiling_mat not found: $PROF_MAT"
    echo "Run run_compress_llama31_8b.sh first to generate it."
    exit 1
fi

for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_${SUFFIX}_${KEEP}.pt"

    if [ -f "$CKPT" ]; then
        echo "=== checkpoint exists, skipping: $CKPT ==="
        continue
    fi

    KEEP_DISPLAY=$(python -c "print(round(1 - $RATIO, 1))")
    echo "=== Compress ${SUFFIX} ratio=$RATIO (keep=$KEEP_DISPLAY) ==="
    python run_svdllm_v2_compress.py \
        --model "$MODEL" \
        --ratio $RATIO \
        --profiling_mat_path "$PROF_MAT" \
        --save_path "$SAVE_DIR" \
        --model_seq_len $SEQ_LEN \
        $TOKEN_ARG \
        2>&1 | tee "logs/${MODEL_TAG}_${SUFFIX}_${KEEP_DISPLAY}.log"
done

echo ""
echo "=== All done ==="
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_${SUFFIX}_${KEEP}.pt"
    [ -f "$CKPT" ] && echo "  ✓ $CKPT" || echo "  ✗ MISSING: $CKPT"
done
