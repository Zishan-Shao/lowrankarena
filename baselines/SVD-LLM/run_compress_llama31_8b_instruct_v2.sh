#!/bin/bash
# SVD-LLM V2 compression: whitening_hetero (+ optional local_update)
# Model: meta-llama/Llama-3.1-8B-Instruct
# Ratios: 0.2 0.3 0.4 0.5 0.6
#
# Usage:
#   bash run_compress_llama31_8b_instruct_v2.sh [HF_TOKEN]          # V2 only
#   bash run_compress_llama31_8b_instruct_v2.sh [HF_TOKEN] update   # V2 + local update
#
# NOTE: requires profiling_mat from run_compress_llama31_8b_instruct.sh (step 1) first.

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TAG="Llama-3.1-8B-Instruct"
MODEL_PREFIX="meta_llama_Llama_3.1_8B_Instruct"
SAVE_DIR="checkpoints/svdllm/llama31_8b_instruct"
HF_TOKEN="${1:-}"
DO_UPDATE="${2:-}"
SEQ_LEN=2048
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

UPDATE_ARG=""
SUFFIX="v2"
[ "$DO_UPDATE" = "update" ] && UPDATE_ARG="--local_update" && SUFFIX="v2_then_update"

if [ ! -f "$PROF_MAT" ]; then
    echo "ERROR: profiling_mat not found: $PROF_MAT"
    echo "Run run_compress_llama31_8b_instruct.sh first to generate it."
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
        $UPDATE_ARG \
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
