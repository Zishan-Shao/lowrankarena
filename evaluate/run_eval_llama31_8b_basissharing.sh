#!/bin/bash
# Evaluate SVD-LLM Basis Sharing compressed checkpoints for Llama-3.1-8B.
# Loads from safetensors directories (model.safetensors + svd_metadata.json).
#
# Usage:
#   bash run_eval_llama31_8b_basissharing.sh [HF_TOKEN]
#
# Output: results/llama31_8b.csv
# Run from: lowrankarena/evaluate/

set -eo pipefail
cd "$(dirname "$0")"

MODEL_FAMILY="llama"
MODEL_TAG="Llama-3.1-8B"
IS_INSTRUCT="0"
BASE_MODEL="meta-llama/Llama-3.1-8B"
OUTPUT_DIR="/home/ww247/lowrankarena/hf_ckpts/LowRankArena/LLama31_8b/BasisSharing"
MODEL_PREFIX="meta_llama_Llama_3.1_8B"
METHOD="Basis_sharing"
CSV="results/llama31_8b.csv"
DTYPE="bf16"
DEVICE="cuda:0"
BATCH="2"
HF_TOKEN="${1:-}"

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

eval_one() {
    local ckpt="$1" keep="$2"
    echo ""
    echo ">>> $METHOD  keep=$keep  ckpt=$(basename $ckpt)"
    python eval_decoder.py \
        --checkpoint        "$ckpt" \
        --model_family      "$MODEL_FAMILY" \
        --model_tag         "$MODEL_TAG" \
        --is_instruct       "$IS_INSTRUCT" \
        --method            "$METHOD" \
        --keep_ratio        "$keep" \
        --dtype             "$DTYPE" \
        --device            "$DEVICE" \
        --output_csv        "$CSV" \
        --batch_size        "$BATCH" \
        --lmeval_batch_size 2 \
        --tokenizer         "$BASE_MODEL" \
        $TOKEN_ARG
}

# ── baseline ───────────────────────────────────────────────────────────────────
eval_one "$BASE_MODEL" 1.0

# ── Basis Sharing checkpoints (safetensors) ───────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(round(1 - $RATIO, 1))")
    CKPT="$OUTPUT_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP}"
    if [ -f "$CKPT/model.safetensors" ]; then
        eval_one "$CKPT" "$KEEP"
    else
        echo "  [skip] not found: $CKPT/model.safetensors"
    fi
done

echo ""
echo "=== All done. Results: $CSV ==="
