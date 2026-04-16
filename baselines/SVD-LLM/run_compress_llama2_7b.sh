#!/bin/bash
# SVD-LLM V1 (whitening only) + V2 (whitening + local update)
# Model: meta-llama/Llama-2-7b-hf
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Usage:
#   bash run_compress_llama2_7b.sh [HF_TOKEN]
#
# Results CSV: checkpoints/svdllm/llama2_7b/results.csv
# Columns: model,method,keep_ratio,wikitext2_ppl,baseline_ms,flashsvd_ms,speedup

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

# ── helpers ───────────────────────────────────────────────────────────────────

keep_file() { python -c "print(1 - $1)"; }
keep_csv()  { python -c "print(round(1 - $1, 1))"; }

# ── Step 1: V1 (skip if checkpoints already exist) ────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_whitening_only_${KEEP_FILE}.pt"
    if [ -f "$CKPT" ]; then
        echo "=== V1 checkpoint exists, skipping: $CKPT ==="
    else
        KEEP=$(keep_csv $RATIO)
        echo "=== Compress V1 ratio=$RATIO (keep=$KEEP) ==="
        PROF_ARG=""
        [ -f "$PROF_MAT" ] && PROF_ARG="--profiling_mat_path $PROF_MAT"
        python SVDLLM.py --model "$MODEL" --step 1 --ratio $RATIO \
            $PROF_ARG \
            --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
            2>&1 | tee logs/${MODEL_TAG}_v1_${KEEP}.log
    fi
done

# ── Step 2: V1update (whitening_then_update) ──────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_whitening_then_update_${KEEP_FILE}.pt"
    if [ -f "$CKPT" ]; then
        echo "=== V1update checkpoint exists, skipping: $CKPT ==="
    else
        KEEP=$(keep_csv $RATIO)
        echo "=== Compress V1update ratio=$RATIO (keep=$KEEP) ==="
        python SVDLLM.py --model "$MODEL" --step 2 --ratio $RATIO \
            --profiling_mat_path "$PROF_MAT" \
            --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
            2>&1 | tee logs/${MODEL_TAG}_v1update_${KEEP}.log
    fi
done

echo ""
echo "=== All done ==="
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    V1="$SAVE_DIR/${MODEL_PREFIX}_whitening_only_${KEEP_FILE}.pt"
    V1U="$SAVE_DIR/${MODEL_PREFIX}_whitening_then_update_${KEEP_FILE}.pt"
    [ -f "$V1" ]  && echo "  ✓ $V1"  || echo "  ✗ MISSING: $V1"
    [ -f "$V1U" ] && echo "  ✓ $V1U" || echo "  ✗ MISSING: $V1U"
done
