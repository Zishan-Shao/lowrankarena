#!/bin/bash
# DobiSVD stability test + compression
# ratio=0.2/0.4/0.6, dtype=bfloat16/float32, grad_clip=1.0
# Trains 5 epochs, checks stability, then runs weight_updater to save checkpoint

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG="Llama-3.1-8B"
LOWER_ID="Llama-3.1-8B"
SEQ_LEN=2048

mkdir -p logs

find_latest_training_dir() {
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

check_gamma() {
    local ratio="$1"
    python -c "
import json, glob, math
dirs = glob.glob('results/training_output/${LOWER_ID}/Diff-Noremapping-${ratio}_*')
dirs = [d for d in dirs if os.path.exists(d+'/best_gamma.json')]
if not dirs:
    print('NO_GAMMA'); exit()
d = json.load(open(sorted(dirs, key=lambda x: os.path.getmtime(x))[-1]))
vals = [v for k,v in d.items() if isinstance(v,(int,float)) and k not in ('ppl','compression_ratio','lr','PPL_ORIG')]
nan_count = sum(1 for v in vals if math.isnan(v))
finite_vals = [v for v in vals if not math.isnan(v)]
print(f'count={len(vals)} nan={nan_count} min={min(finite_vals):.1f} max={max(finite_vals):.1f} mean={sum(finite_vals)/len(finite_vals):.1f}')
"
}

for RATIO in 0.2 0.4 0.6; do
    for DTYPE in bfloat16 float32; do
        KEEP=$(python -c "print(round(1 - $RATIO, 1))")
        DOBI_PT="results/compressed_model/${LOWER_ID}/DobiSVD_Noremapping-${LOWER_ID}-${RATIO}/DobiSVD_Model.pt"

        if [ -f "$DOBI_PT" ]; then
            echo "=== checkpoint exists, skipping: $DOBI_PT ==="
            continue
        fi

        echo ""
        echo "=== [ratio=$RATIO dtype=$DTYPE] Train gamma (5 epochs) ==="
        python svd_trainer.py \
            --model_id "$MODEL" \
            --target_ratio "$RATIO" \
            --seq_len $SEQ_LEN \
            --training_dataset wikitext2 \
            --n_train_epochs 5 \
            --n_train_samples 256 \
            --model_dtype "$DTYPE" \
            --max_grad_norm 1.0 \
            2>&1 | tee "logs/${MODEL_TAG}_dobi_${RATIO}_${DTYPE}.log"

        echo "=== [ratio=$RATIO dtype=$DTYPE] Gamma check: $(check_gamma $RATIO) ==="

        TRAIN_DIR=$(find_latest_training_dir "$RATIO")
        if [ -z "$TRAIN_DIR" ]; then
            echo "ERROR: no training dir found, skipping weight_updater"
            continue
        fi

        echo "=== [ratio=$RATIO dtype=$DTYPE] weight_updater → $DOBI_PT ==="
        python weight_updater.py \
            --model_id "$MODEL" \
            --training_result_path "$TRAIN_DIR" \
            2>&1 | tee "logs/${MODEL_TAG}_dobi_update_${RATIO}_${DTYPE}.log"

        # stop trying other dtypes once checkpoint is saved
        [ -f "$DOBI_PT" ] && break
    done
done

echo ""
echo "=== All done ==="
for RATIO in 0.2 0.4 0.6; do
    DOBI_PT="results/compressed_model/${LOWER_ID}/DobiSVD_Noremapping-${LOWER_ID}-${RATIO}/DobiSVD_Model.pt"
    [ -f "$DOBI_PT" ] && echo "  ✓ $DOBI_PT" || echo "  ✗ MISSING: $DOBI_PT"
done
