#!/bin/bash
# DobiSVD stability test: ratio=0.4/0.6, dtype=bfloat16/float32, grad_clip=1.0
# Checks if training diverges across dtype/ratio combinations

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG="Llama-3.1-8B"
LOWER_ID="Llama-3.1-8B"
SEQ_LEN=2048

mkdir -p logs

find_training_dir() {
    local ratio="$1" dtype="$2"
    python -c "
import os, glob
dirs = glob.glob('results/training_output/${LOWER_ID}/Diff-Noremapping-${ratio}_*_${dtype}')
dirs = [d for d in dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'best_gamma.json'))]
if not dirs:
    print('')
else:
    print(os.path.basename(sorted(dirs, key=os.path.getmtime)[-1]))
"
}

for RATIO in 0.4 0.6; do
    for DTYPE in bfloat16 float32; do
        KEEP=$(python -c "print(round(1 - $RATIO, 1))")
        DOBI_PT="results/compressed_model/${LOWER_ID}/DobiSVD_Noremapping-${LOWER_ID}-${RATIO}/DobiSVD_Model.pt"

        if [ -f "$DOBI_PT" ]; then
            echo "=== checkpoint exists, skipping: $DOBI_PT ==="
            continue
        fi

        echo "=== DobiSVD train ratio=$RATIO dtype=$DTYPE grad_clip=1.0 ==="
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

        echo "=== Check gamma values ==="
        python -c "
import json, glob
dirs = glob.glob('results/training_output/${LOWER_ID}/Diff-Noremapping-${RATIO}_*')
dirs = [d for d in dirs if os.path.exists(d+'/best_gamma.json')]
if dirs:
    d = json.load(open(sorted(dirs, key=lambda x: x)[-1]))
    vals = [v for k,v in d.items() if isinstance(v,(int,float)) and k not in ('ppl','compression_ratio','lr','PPL_ORIG')]
    import math
    nan_count = sum(1 for v in vals if math.isnan(v))
    print(f'  ratio=$RATIO dtype=$DTYPE: count={len(vals)} nan={nan_count} min={min(v for v in vals if not math.isnan(v)):.1f} max={max(v for v in vals if not math.isnan(v)):.1f}')
"
    done
done

echo "=== Done ==="
