#!/bin/bash
# Evaluate SVD-LLM Basis Sharing compressed checkpoints for Qwen3-8B-Instruct.
# Loads from safetensors directories (model.safetensors + svd_metadata.json).
#
# Usage:
#   bash run_eval_qwen3_8b_instruct_basissharing.sh
#
# Output: results/qwen3_8b_instruct.csv
# Run from: lowrankarena/evaluate/

set -eo pipefail
cd "$(dirname "$0")"

MODEL_FAMILY="qwen"
MODEL_TAG="Qwen3-8B-Instruct"
IS_INSTRUCT="1"
BASE_MODEL="Qwen/Qwen3-8B-Instruct"
OUTPUT_DIR="/home/ww247/lowrankarena/hf_ckpts/LowRankArena/qwen3_8b_instruct/BasisSharing"
MODEL_PREFIX="Qwen_Qwen3_8B_Instruct"
METHOD="Basis_sharing"
CSV="results/qwen3_8b_instruct.csv"
DTYPE="bf16"
DEVICE="cuda:0"
BATCH="2"

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
        --tokenizer         "$BASE_MODEL"
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
