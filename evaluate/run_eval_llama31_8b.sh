#!/bin/bash
# Evaluate all compressed checkpoints for Llama-3.1-8B.
#
# Usage:
#   bash run_eval_llama31_8b.sh [HF_TOKEN]
#
# Output CSV: results/llama31_8b.csv
# Run from: lowrankarena/evaluate/

set -eo pipefail
cd "$(dirname "$0")"

MODEL_TAG="Llama-3.1-8B"
BASE_MODEL="meta-llama/Llama-3.1-8B"
CKPT_BASE="../hf_ckpts/LowRankArena/llama31_8b"
CSV="results/llama31_8b.csv"
DTYPE="bf16"
DEVICE="cuda:0"
BATCH="2"
HF_TOKEN="${1:-}"

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

# keep_ratio values: 0.4 0.5 0.6 0.7 0.8  (= 60/50/40/30/20% compressed)
KEEPS="0.4 0.5 0.6 0.7 0.8"

# ── helper ─────────────────────────────────────────────────────────────────────
eval_one() {
    local ckpt="$1" method="$2" keep="$3"
    if [ ! -d "$ckpt" ] && [ "$ckpt" != "$BASE_MODEL" ]; then
        echo "  [skip] not found: $ckpt"
        return
    fi
    echo ""
    echo ">>> $method keep=$keep"
    python eval_decoder.py \
        --checkpoint  "$ckpt" \
        --model_tag   "$MODEL_TAG" \
        --method      "$method" \
        --keep_ratio  "$keep" \
        --dtype       "$DTYPE" \
        --device      "$DEVICE" \
        --output_csv  "$CSV" \
        --batch_size  "$BATCH" \
        $TOKEN_ARG
}

# ── baseline ───────────────────────────────────────────────────────────────────
eval_one "$BASE_MODEL" baseline 1.0

# ── ASVD ───────────────────────────────────────────────────────────────────────
for K in $KEEPS; do
    eval_one "$CKPT_BASE/ASVD/hf_asvd_raw_${K}" ASVD "$K"
done

# ── SVD-LLM V1 (whitening only) ────────────────────────────────────────────────
for K in $KEEPS; do
    eval_one "$CKPT_BASE/SVDLLM/hf_whitening_only_${K}" SVDLLMv1 "$K"
done

# ── SVD-LLM V2 (whitening + hetero) ────────────────────────────────────────────
for K in $KEEPS; do
    eval_one "$CKPT_BASE/SVDLLM/hf_v2_${K}" SVDLLMv2 "$K"
done

# ── SVD-LLM Basis Sharing ──────────────────────────────────────────────────────
for K in $KEEPS; do
    eval_one "$CKPT_BASE/SVDLLM/hf_basis_sharing_${K}" SVDLLMbs "$K"
done

# ── Dobi-SVD ───────────────────────────────────────────────────────────────────
for K in $KEEPS; do
    eval_one "$CKPT_BASE/Dobi/hf_dobi_${K}" Dobi "$K"
done

echo ""
echo "=== All done. Results: $CSV ==="
