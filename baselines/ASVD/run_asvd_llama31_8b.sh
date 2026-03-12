#!/bin/bash
# ASVD + conversion to SVD-LLM format
# Model: meta-llama/Llama-3.1-8B
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4  (param_ratio_target)
#
# Pipeline:
#   1. asvd.py  → saves ASVD .pt  (SVDLinear model)
#   2. convert_asvd_to_svdllm.py  → converts to SVD-LLM format  (.pt with SVD_LlamaAttention)
#   3. Eval PPL via SVD-LLM SVDLLM.py --step 4
#   4. Bench speed via bench_flashsvd_vs_svd_decode.py
#
# Usage:
#   bash run_asvd_llama31_8b.sh [HF_TOKEN]
#
# Results CSV: checkpoints/asvd/llama31_8b/results.csv

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG="Llama-3.1-8B"
SAVE_DIR="checkpoints/asvd/llama31_8b"
SVDLLM_DIR="../SVD-LLM"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
EVAL_BS=2
CSV="$SAVE_DIR/results.csv"

mkdir -p "$SAVE_DIR" logs

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

# ── Compress + Convert ────────────────────────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(round(1 - $RATIO, 1))")
    ASVD_PT="$SAVE_DIR/${MODEL_TAG//-/_}_asvd_raw_${KEEP}.pt"
    SVDLLM_PT="$SAVE_DIR/${MODEL_TAG//-/_}_asvd_${KEEP}.pt"

    # Step 1: ASVD compression
    if [ -f "$ASVD_PT" ]; then
        echo "=== ASVD raw checkpoint exists, skipping: $ASVD_PT ==="
    else
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
    fi

    # Step 2: Convert to SVD-LLM format
    if [ -f "$SVDLLM_PT" ]; then
        echo "=== SVD-LLM checkpoint exists, skipping: $SVDLLM_PT ==="
    else
        echo "=== Convert ASVD → SVD-LLM: keep=$KEEP ==="
        (cd "$SVDLLM_DIR" && python ../ASVD/convert_asvd_to_svdllm.py \
            --checkpoint "$(realpath "../ASVD/$ASVD_PT")" \
            --out "$(realpath "../ASVD/$SVDLLM_PT")" \
            2>&1) | tee "logs/${MODEL_TAG}_asvd_convert_${KEEP}.log"
    fi
done

# ── Eval all converted checkpoints ───────────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(round(1 - $RATIO, 1))")
    SVDLLM_PT="$SAVE_DIR/${MODEL_TAG//-/_}_asvd_${KEEP}.pt"
    eval_and_log "$(realpath "$SVDLLM_PT")" "ASVD" "$KEEP"
done

echo ""
echo "=== All done. Results: $CSV ==="
cat "$CSV"
