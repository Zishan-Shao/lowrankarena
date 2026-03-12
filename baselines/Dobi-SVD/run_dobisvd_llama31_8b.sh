#!/bin/bash
# DobiSVD + conversion to SVD-LLM format
# Model: meta-llama/Llama-3.1-8B
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Pipeline:
#   1. svd_trainer.py   → trains gamma, saves json to results/training_output/
#   2. weight_updater.py → applies gamma, saves SVDTransformLayer model
#   3. convert_asvd_to_svdllm.py → converts to SVD-LLM format
#   4. Eval PPL via SVD-LLM SVDLLM.py --step 4
#   5. Bench speed via bench_flashsvd_vs_svd_decode.py
#
# Usage:
#   bash run_dobisvd_llama31_8b.sh [HF_TOKEN]
#
# Results CSV: results/compressed_model/Llama-3.1-8B/results.csv

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG="Llama-3.1-8B"
LOWER_ID="Llama-3.1-8B"
SVDLLM_DIR="../SVD-LLM"
ASVD_CONVERT="../ASVD/convert_asvd_to_svdllm.py"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
EVAL_BS=2
RESULTS_DIR="results/compressed_model/${LOWER_ID}"
CSV="${RESULTS_DIR}/results.csv"

mkdir -p "$RESULTS_DIR" logs

# ── helpers ───────────────────────────────────────────────────────────────────

parse_ppl() {
    local f="$1"
    python -c "
import re
txt = open('$f').read()
m = re.search(r\"'wikitext2':\\s*([0-9]+\\.[0-9]+)\", txt)
print(m.group(1) if m else 'N/A')
"
}

parse_ms() {
    local f="$1" tag="$2"
    python -c "
import re
txt = open('$f').read()
m = re.search(r'^${tag} decode: ([0-9]+\.?[0-9]*) ms/token', txt, re.MULTILINE)
print(m.group(1) if m else 'N/A')
"
}

# Find the most recently modified training output directory for a given ratio
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

eval_and_log() {
    local ckpt="$1" method="$2" keep="$3"

    echo "=== Eval PPL: $method keep=$keep ==="
    local ppl_out; ppl_out=$(mktemp)
    (cd "$SVDLLM_DIR" && python SVDLLM.py \
        --model "$MODEL" --model_path "$ckpt" \
        --step 4 --model_seq_len $SEQ_LEN --eval_batch_size $EVAL_BS \
        ${HF_TOKEN:+--hf_token "$HF_TOKEN"} 2>&1) | tee "$ppl_out"
    local ppl; ppl=$(parse_ppl "$ppl_out")
    rm -f "$ppl_out"

    local base_ms="N/A" flash_ms="N/A" speedup="N/A"
    echo "=== Bench speed: $method keep=$keep ==="
    local bench_out; bench_out=$(mktemp)
    (cd "$SVDLLM_DIR" && python bench_flashsvd_vs_svd_decode.py \
        --checkpoint "$ckpt" \
        --dtype bf16 --prompt_len 512 --new_tokens 128 --warmup 5 \
        --experimental_flash_dense_attn \
        2>&1) | tee "$bench_out"
    base_ms=$(parse_ms "$bench_out" "SVD")
    flash_ms=$(parse_ms "$bench_out" "FlashSVD")
    speedup=$(python -c "
b, f = '$base_ms', '$flash_ms'
try: print(f'{float(b)/float(f):.3f}')
except: print('N/A')
")
    rm -f "$bench_out"

    echo "$MODEL_TAG,$method,$keep,$ppl,$base_ms,$flash_ms,$speedup" >> "$CSV"
    echo "  → CSV: $MODEL_TAG,$method,$keep ppl=$ppl base=${base_ms}ms flash=${flash_ms}ms speedup=${speedup}x"
}

# ── init CSV ──────────────────────────────────────────────────────────────────
if [ ! -f "$CSV" ]; then
    echo "model,method,keep_ratio,wikitext2_ppl,baseline_ms,flashsvd_ms,speedup" > "$CSV"
fi

# ── Step 1 + 2 + 3: Train gamma → Apply weights → Convert ────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(round(1 - $RATIO, 1))")
    DOBI_PT="results/compressed_model/${LOWER_ID}/DobiSVD_Noremapping-${LOWER_ID}-${RATIO}/DobiSVD_Model.pt"
    SVDLLM_PT="${RESULTS_DIR}/dobisvd_${KEEP}.pt"

    # Step 1: Train gamma (skip if best_gamma.json already exists)
    TRAIN_DIR=$(find_training_dir "$RATIO")
    if [ -n "$TRAIN_DIR" ]; then
        echo "=== DobiSVD training output exists: $TRAIN_DIR, skipping svd_trainer ==="
    else
        echo "=== DobiSVD train gamma ratio=$RATIO (保存率=$KEEP) ==="
        python svd_trainer.py \
            --model_id "$MODEL" \
            --target_ratio "$RATIO" \
            --seq_len $SEQ_LEN \
            --training_dataset wikitext2 \
            --n_train_epochs 20 \
            --n_train_samples 256 \
            2>&1 | tee "logs/${MODEL_TAG}_dobi_train_${KEEP}.log"
        TRAIN_DIR=$(find_training_dir "$RATIO")
    fi

    if [ -z "$TRAIN_DIR" ]; then
        echo "ERROR: training dir not found for ratio=$RATIO after training, skipping"
        continue
    fi

    # Step 2: Apply weights → SVDTransformLayer model
    if [ -f "$DOBI_PT" ]; then
        echo "=== DobiSVD raw checkpoint exists, skipping: $DOBI_PT ==="
    else
        echo "=== DobiSVD weight_updater ratio=$RATIO (保存率=$KEEP) ==="
        python weight_updater.py \
            --model_id "$MODEL" \
            --training_result_path "$TRAIN_DIR" \
            2>&1 | tee "logs/${MODEL_TAG}_dobi_update_${KEEP}.log"
    fi

    # Step 3: Convert to SVD-LLM format
    if [ -f "$SVDLLM_PT" ]; then
        echo "=== SVD-LLM checkpoint exists, skipping: $SVDLLM_PT ==="
    else
        echo "=== Convert DobiSVD → SVD-LLM: keep=$KEEP ==="
        (cd "$SVDLLM_DIR" && python "$(realpath "$ASVD_CONVERT")" \
            --checkpoint "$(realpath "$DOBI_PT")" \
            --out "$(realpath "$SVDLLM_PT")" \
            2>&1) | tee "logs/${MODEL_TAG}_dobi_convert_${KEEP}.log"
    fi
done

# ── Eval all converted checkpoints ───────────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(round(1 - $RATIO, 1))")
    SVDLLM_PT="${RESULTS_DIR}/dobisvd_${KEEP}.pt"
    eval_and_log "$(realpath "$SVDLLM_PT")" "DobiSVD" "$KEEP"
done

echo ""
echo "=== All done. Results: $CSV ==="
cat "$CSV"
