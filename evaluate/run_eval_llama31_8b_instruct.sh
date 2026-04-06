#!/bin/bash
# Evaluate all compressed checkpoints for Llama-3.1-8B-Instruct.
#
# Discovers checkpoints automatically by scanning:
#   ../hf_ckpts/LowRankArena/llama31_8b_instruct/{METHOD}/hf_*_{keep_ratio}
#
# Method name  = subdirectory name  (e.g. ASVD, SVDLLM, Dobi)
# keep_ratio   = trailing float in the checkpoint dir name
#
# Usage:
#   bash run_eval_llama31_8b_instruct.sh [HF_TOKEN]
#
# Output: results/llama31_8b_instruct.csv
# Run from: lowrankarena/evaluate/

set -eo pipefail
cd "$(dirname "$0")"

MODEL_FAMILY="llama"
MODEL_TAG="Llama-3.1-8B-Instruct"
IS_INSTRUCT="1"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
CKPT_BASE="../hf_ckpts/LowRankArena/llama31_8b_instruct"
CSV="results/llama31_8b_instruct.csv"
DTYPE="bf16"
DEVICE="cuda:0"
BATCH="2"
HF_TOKEN="${1:-}"

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

# ── helper ─────────────────────────────────────────────────────────────────────
eval_one() {
    local ckpt="$1" method="$2" keep="$3"
    echo ""
    echo ">>> $method  keep=$keep"
    python eval_decoder.py \
        --checkpoint        "$ckpt" \
        --model_family      "$MODEL_FAMILY" \
        --model_tag         "$MODEL_TAG" \
        --is_instruct       "$IS_INSTRUCT" \
        --method            "$method" \
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
eval_one "$BASE_MODEL" baseline 1.0

# ── scan CKPT_BASE for all method dirs ─────────────────────────────────────────
if [ ! -d "$CKPT_BASE" ]; then
    echo "[warn] checkpoint base not found: $CKPT_BASE"
    exit 0
fi

for method_dir in "$CKPT_BASE"/*/; do
    [ -d "$method_dir" ] || continue
    parent_method=$(basename "$method_dir")

    for ckpt_dir in "$method_dir"*/; do
        [ -d "$ckpt_dir" ] || continue
        ckpt_dir="${ckpt_dir%/}"

        keep=$(basename "$ckpt_dir" | grep -oE '[0-9]+\.[0-9]+$')
        if [ -z "$keep" ]; then
            echo "  [skip] cannot parse keep_ratio from: $(basename "$ckpt_dir")"
            continue
        fi

        # Extract variant tag from dir name (strip hf_ prefix and trailing _keep_ratio)
        # e.g. hf_whitening_only_0.5 → whitening_only
        #      hf_v2_0.5            → v2
        tag=$(basename "$ckpt_dir" | sed 's/^hf_//' | sed 's/_[0-9]*\.[0-9]*$//')
        method="${parent_method}_${tag}"

        eval_one "$ckpt_dir" "$method" "$keep"
    done
done

echo ""
echo "=== All done. Results: $CSV ==="
