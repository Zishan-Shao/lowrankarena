#!/bin/bash
# Evaluate SVD-LLM V2 compressed checkpoints for Llama-2-7b.
# Auto-discovers all safetensors subdirs under OUTPUT_DIR.
#
# Usage:
#   bash run_eval_llama2_7b_v2.sh [HF_TOKEN]
#
# Output: results/llama2_7b.csv
# Run from: lowrankarena/evaluate/

set -eo pipefail
cd "$(dirname "$0")"

MODEL_FAMILY="llama"
MODEL_TAG="Llama-2-7b"
IS_INSTRUCT="0"
BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="/home/ww247/lowrankarena/hf_ckpts/LowRankArena/llama2_7b/SVDLLMv2"
METHOD="SVDLLMv2"
CSV="results/llama2_7b.csv"
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

# ── auto-discover safetensors checkpoints ─────────────────────────────────────
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "[warn] OUTPUT_DIR not found: $OUTPUT_DIR"
    exit 0
fi

for ckpt_dir in "$OUTPUT_DIR"/*/; do
    [ -d "$ckpt_dir" ] || continue
    [ -f "${ckpt_dir}model.safetensors" ] || continue

    keep=$(basename "$ckpt_dir" | grep -oE '[0-9]+\.[0-9]+$')
    if [ -z "$keep" ]; then
        echo "  [skip] cannot parse keep_ratio from: $(basename $ckpt_dir)"
        continue
    fi

    eval_one "${ckpt_dir%/}" "$keep"
done

echo ""
echo "=== All done. Results: $CSV ==="
