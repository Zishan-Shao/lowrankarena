#!/bin/bash
# ASVD compression only (no speed benchmark)
# Model: Qwen/Qwen3-8B-Instruct
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Usage:
#   bash run_asvd_qwen3_8b.sh [HF_TOKEN]
#
# Checkpoints: checkpoints/asvd/qwen3_8b/Qwen3_8B_Instruct_asvd_raw_*.pt

set -eo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-8B-Instruct"
MODEL_TAG="Qwen3-8B-Instruct"
SAVE_DIR="checkpoints/asvd/qwen3_8b"
HF_TOKEN="${1:-}"

[ -n "$HF_TOKEN" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" && export HF_TOKEN="$HF_TOKEN"

mkdir -p "$SAVE_DIR" logs

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
        2>&1 | tee "logs/${MODEL_TAG}_asvd_${KEEP}.log"
done

echo ""
echo "=== All done. Checkpoints in $SAVE_DIR ==="
ls "$SAVE_DIR"
