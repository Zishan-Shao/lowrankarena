#!/bin/bash
# DobiSVD compression only (no speed benchmark)
# Model: meta-llama/Llama-3.1-8B-Instruct
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Pipeline:
#   1. svd_trainer.py   → trains gamma, saves json to results/training_output/
#   2. weight_updater.py → applies gamma, saves DobiSVD_Model.pt (SVDTransformLayer)
#
# Usage:
#   bash run_dobisvd_llama31_8b_instruct.sh [RATIO...]
#   bash run_dobisvd_llama31_8b_instruct.sh          # all 5 ratios
#   bash run_dobisvd_llama31_8b_instruct.sh 0.2 0.4  # selected ratios
#
# Checkpoints: results/compressed_model/Llama-3.1-8B-Instruct/DobiSVD_Noremapping-Llama-3.1-8B-Instruct-{ratio}/DobiSVD_Model.pt

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TAG="Llama-3.1-8B-Instruct"
LOWER_ID="Llama-3.1-8B-Instruct"
SEQ_LEN=2048

mkdir -p logs

find_training_dir() {
    local ratio="$1"
    python -c "
import os, glob
dirs = glob.glob('results/training_output/${LOWER_ID}/Diff-Noremapping-${ratio}_*')
dirs = [d for d in dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'best_gamma.json'))]
if not dirs:
    print('')
else:
    print(os.path.basename(sorted(dirs, key=os.path.getmtime)[-1]))
"
}

RATIOS="${@:-0.2 0.3 0.4 0.5 0.6}"
for RATIO in $RATIOS; do
    KEEP=$(python -c "print(round(1 - $RATIO, 1))")
    DOBI_PT="results/compressed_model/${LOWER_ID}/DobiSVD_Noremapping-${LOWER_ID}-${RATIO}/DobiSVD_Model.pt"

    if [ -f "$DOBI_PT" ]; then
        echo "=== checkpoint exists, skipping: $DOBI_PT ==="
        continue
    fi

    # Step 1: Train gamma
    TRAIN_DIR=$(find_training_dir "$RATIO")
    if [ -n "$TRAIN_DIR" ]; then
        echo "=== training output exists: $TRAIN_DIR, skipping svd_trainer ==="
    else
        echo "=== DobiSVD train gamma ratio=$RATIO (保存率=$KEEP) ==="
        python svd_trainer.py \
            --model_id "$MODEL" \
            --target_ratio "$RATIO" \
            --seq_len $SEQ_LEN \
            --training_dataset wikitext2 \
            --n_train_epochs 20 \
            --n_train_samples 256 \
            --model_dtype bfloat16 \
            --max_grad_norm 1.0 \
            2>&1 | tee "logs/${MODEL_TAG}_dobi_train_${KEEP}.log"
        TRAIN_DIR=$(find_training_dir "$RATIO")
    fi

    if [ -z "$TRAIN_DIR" ]; then
        echo "ERROR: training dir not found for ratio=$RATIO, skipping"
        continue
    fi

    # Step 2: Apply weights
    echo "=== DobiSVD weight_updater ratio=$RATIO (保存率=$KEEP) ==="
    python weight_updater.py \
        --model_id "$MODEL" \
        --training_result_path "$TRAIN_DIR" \
        2>&1 | tee "logs/${MODEL_TAG}_dobi_update_${KEEP}.log"
done

echo ""
echo "=== All done ==="
for RATIO in $RATIOS; do
    DOBI_PT="results/compressed_model/${LOWER_ID}/DobiSVD_Noremapping-${LOWER_ID}-${RATIO}/DobiSVD_Model.pt"
    [ -f "$DOBI_PT" ] && echo "  ✓ $DOBI_PT" || echo "  ✗ MISSING: $DOBI_PT"
done
