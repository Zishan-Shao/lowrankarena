#!/bin/bash
# ASVD compression only (no speed benchmark)
# Model: meta-llama/Llama-3.1-8B
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Usage:
#   bash run_asvd_llama31_8b.sh [HF_TOKEN]
#
# Checkpoints: checkpoints/asvd/llama31_8b/Llama_3.1_8B_asvd_raw_*.pt

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG="Llama-3.1-8B"
SAVE_DIR="checkpoints/asvd/llama31_8b"
HF_TOKEN="${1:-}"

mkdir -p "$SAVE_DIR" logs

# ── Compress ──────────────────────────────────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(round(1 - $RATIO, 1))")
    ASVD_PT="$SAVE_DIR/${MODEL_TAG//-/_}_asvd_raw_${KEEP}.pt"

    if [ -f "$ASVD_PT" ]; then
        echo "=== checkpoint exists, skipping: $ASVD_PT ==="
        continue
    fi

    echo "=== ASVD compress ratio=$RATIO (保存率=$KEEP) ==="
    python asvd.py \
        --model_id "$MODEL" \
        --param_ratio_target "$KEEP" \
        --act_aware \
        --alpha 0.5 \
        --n_calib_samples 32 \
        --calib_dataset wikitext2 \
        --scaling_method abs_mean \
        --sensitivity_metric ppl \
        --eval_ppl wikitext2 \
        --save_path "$ASVD_PT" \
        ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
        2>&1 | tee "logs/${MODEL_TAG}_asvd_${KEEP}.log"
done

echo ""
echo "=== All done. Checkpoints in $SAVE_DIR ==="
ls "$SAVE_DIR"
