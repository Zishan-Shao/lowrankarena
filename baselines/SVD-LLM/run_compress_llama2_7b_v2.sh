#!/bin/bash
# SVD-LLM V2 heterogeneous compression (fully consistent with new repo)
# Model: meta-llama/Llama-2-7b-hf
# Ratios: 0.2 0.3 0.4 0.5 0.6
#
# Uses hetero eigendecomp profiling + adaptive rank allocation.
# Profiling_mat saved separately so it only runs once.
#
# Usage:
#   bash run_compress_llama2_7b_v2.sh [HF_TOKEN]

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-2-7b-hf"
MODEL_TAG="Llama-2-7b"
MODEL_PREFIX="meta_llama_Llama_2_7b_hf"
SAVE_DIR="checkpoints/svdllm/llama2_7b"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
SUFFIX="v2hetero"

# Hetero eigendecomp profiling_mat (separate from V1 Cholesky profiling_mat)
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_hetero_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

# Determine profiling arg: reprofile if hetero profiling_mat doesn't exist
if [ -f "$PROF_MAT" ]; then
    echo "=== Hetero profiling_mat found, reusing: $PROF_MAT ==="
    PROF_ARG="--profiling_mat_path $PROF_MAT"
else
    echo "=== Hetero profiling_mat not found, will reprofile ==="
    PROF_ARG="--reprofile --save_profiling_mat $PROF_MAT"
fi

for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_${SUFFIX}_${KEEP}.pt"

    if [ -f "$CKPT" ]; then
        echo "=== checkpoint exists, skipping: $CKPT ==="
        # After first run profiling_mat is saved, switch to reuse mode
        PROF_ARG="--profiling_mat_path $PROF_MAT"
        continue
    fi

    KEEP_DISPLAY=$(python -c "print(round(1 - $RATIO, 1))")
    echo "=== Compress ${SUFFIX} ratio=$RATIO (keep=$KEEP_DISPLAY) ==="
    python run_svdllm_v2_compress.py \
        --model "$MODEL" \
        --ratio $RATIO \
        $PROF_ARG \
        --save_path "$SAVE_DIR" \
        --model_seq_len $SEQ_LEN \
        $TOKEN_ARG \
        2>&1 | tee "logs/${MODEL_TAG}_${SUFFIX}_${KEEP_DISPLAY}.log"

    # After first run (which saves profiling_mat), switch to reuse mode
    PROF_ARG="--profiling_mat_path $PROF_MAT"
done

echo ""
echo "=== All done ==="
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_${SUFFIX}_${KEEP}.pt"
    [ -f "$CKPT" ] && echo "  ✓ $CKPT" || echo "  ✗ MISSING: $CKPT"
done
