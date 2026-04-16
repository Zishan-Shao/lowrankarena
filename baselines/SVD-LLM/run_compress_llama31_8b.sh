#!/bin/bash
# SVD-LLM V1 (whitening only) + V2 (whitening + local update)
# Model: meta-llama/Llama-3.1-8B
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Usage:
#   bash run_compress_llama31_8b.sh [HF_TOKEN]
#
# Results CSV: checkpoints/svdllm/llama31_8b/results.csv
# Columns: model,method,keep_ratio,wikitext2_ppl,baseline_ms,flashsvd_ms,speedup

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG="Llama-3.1-8B"
MODEL_PREFIX="meta_llama_Llama_3.1_8B"
SAVE_DIR="checkpoints/svdllm/llama31_8b"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

# ── helpers ───────────────────────────────────────────────────────────────────

# SVDLLM saves checkpoint as str(1 - ratio), which may have float precision issues.
# e.g. 1 - 0.3 = 0.7000000000000001 in Python.
# Use this to get the EXACT filename suffix SVDLLM will produce.
keep_file() { python -c "print(1 - $1)"; }          # exact Python float string
keep_csv()  { python -c "print(round(1 - $1, 1))"; } # rounded for display

# ── Step 1: V1 (skip if checkpoints already exist) ────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_whitening_only_${KEEP_FILE}.pt"
    if [ -f "$CKPT" ]; then
        echo "=== V1 checkpoint exists, skipping: $CKPT ==="
    else
        KEEP=$(keep_csv $RATIO)
        echo "=== Compress V1 ratio=$RATIO (保存率=$KEEP) ==="
        PROF_ARG=""
        [ -f "$PROF_MAT" ] && PROF_ARG="--profiling_mat_path $PROF_MAT"
        python SVDLLM.py --model "$MODEL" --step 1 --ratio $RATIO \
            $PROF_ARG \
            --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
            2>&1 | tee logs/${MODEL_TAG}_v1_${KEEP}.log
    fi
done

# ── Step 2: V2 (skip if checkpoint already exists) ───────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_whitening_then_update_${KEEP_FILE}.pt"
    if [ -f "$CKPT" ]; then
        echo "=== V2 checkpoint exists, skipping: $CKPT ==="
    else
        KEEP=$(keep_csv $RATIO)
        echo "=== Compress V2 ratio=$RATIO (保存率=$KEEP) ==="
        python SVDLLM.py --model "$MODEL" --step 2 --ratio $RATIO \
            --profiling_mat_path "$PROF_MAT" \
            --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
            2>&1 | tee logs/${MODEL_TAG}_v2_${KEEP}.log
    fi
done

echo ""
echo "=== All done ==="
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    V1="$SAVE_DIR/${MODEL_PREFIX}_whitening_only_${KEEP_FILE}.pt"
    V2="$SAVE_DIR/${MODEL_PREFIX}_whitening_then_update_${KEEP_FILE}.pt"
    [ -f "$V1" ] && echo "  ✓ $V1" || echo "  ✗ MISSING: $V1"
    [ -f "$V2" ] && echo "  ✓ $V2" || echo "  ✗ MISSING: $V2"
done
