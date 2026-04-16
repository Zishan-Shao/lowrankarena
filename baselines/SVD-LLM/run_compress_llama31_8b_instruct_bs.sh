#!/bin/bash
# SVD-LLM Basis Sharing (step 5)
# Model: meta-llama/Llama-3.1-8B-Instruct
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Usage:
#   bash run_compress_llama31_8b_instruct_bs.sh [HF_TOKEN]
#
# Results CSV: checkpoints/svdllm/llama31_8b_instruct/results_bs.csv

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TAG="Llama-3.1-8B-Instruct"
MODEL_PREFIX="meta_llama_Llama_3.1_8B_Instruct"
SAVE_DIR="checkpoints/svdllm/llama31_8b_instruct"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

keep_file() { python -c "print(1 - $1)"; }
keep_csv()  { python -c "print(round(1 - $1, 1))"; }

# ── BasisSharing compress ─────────────────────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP_FILE}.pt"
    if [ -f "$CKPT" ]; then
        echo "=== BasisSharing checkpoint exists, skipping: $CKPT ==="
    else
        KEEP=$(keep_csv $RATIO)
        echo "=== Compress BasisSharing ratio=$RATIO (keep=$KEEP) ==="
        PROF_ARG=""
        [ -f "$PROF_MAT" ] && PROF_ARG="--profiling_mat_path $PROF_MAT"
        python SVDLLM.py --model "$MODEL" --step 5 --ratio $RATIO \
            $PROF_ARG \
            --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
            2>&1 | tee logs/${MODEL_TAG}_bs_${KEEP}.log
    fi
done

echo ""
echo "=== All done ==="
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP_FILE}.pt"
    [ -f "$CKPT" ] && echo "  ✓ $CKPT" || echo "  ✗ MISSING: $CKPT"
done
