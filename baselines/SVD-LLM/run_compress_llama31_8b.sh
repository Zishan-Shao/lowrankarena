#!/bin/bash
# SVD-LLM V1 (whitening only) + V2 (whitening + local update)
# Model: meta-llama/Llama-3.1-8B
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Usage:
#   bash run_compress_llama31_8b.sh [HF_TOKEN]
#
# Results CSV: checkpoints/svdllm/llama31_8b/results.csv
# Columns: model,method,keep_ratio,wikitext2_ppl,baseline_ms,flashsvd_ms,speedup

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG="Llama-3.1-8B"
MODEL_PREFIX="meta_llama_Llama_3.1_8B"
SAVE_DIR="checkpoints/svdllm/llama31_8b"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
EVAL_BS=2
CSV="$SAVE_DIR/results.csv"
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

# ── helpers ───────────────────────────────────────────────────────────────────

# SVDLLM saves checkpoint as str(1 - ratio), which may have float precision issues.
# e.g. 1 - 0.3 = 0.7000000000000001 in Python.
# Use this to get the EXACT filename suffix SVDLLM will produce.
keep_file() { python -c "print(1 - $1)"; }          # exact Python float string
keep_csv()  { python -c "print(round(1 - $1, 1))"; } # rounded for display

parse_ppl() {
    # $1: tmpfile containing step-4 output
    local f="$1"
    python -c "
import re
txt = open('$f').read()
m = re.search(r\"wikitext2'?[^:]*:\\s*([0-9]+\\.?[0-9]*)\", txt)
print(m.group(1) if m else 'N/A')
"
}

parse_ms() {
    # $1: tmpfile  $2: "SVD" or "FlashSVD"
    # Use ^ anchor (MULTILINE) so 'SVD decode:' won't match inside 'FlashSVD decode:'
    local f="$1" tag="$2"
    python -c "
import re
txt = open('$f').read()
m = re.search(r'^${tag} decode: ([0-9]+\.?[0-9]*) ms/token', txt, re.MULTILINE)
print(m.group(1) if m else 'N/A')
"
}

eval_and_log() {
    # $1: checkpoint path (or "original")
    # $2: method label (baseline / V1 / V2)
    # $3: keep_ratio for CSV display
    local ckpt="$1" method="$2" keep="$3"

    echo "=== Eval PPL: $method keep=$keep ==="
    local ppl_out; ppl_out=$(mktemp)
    python SVDLLM.py \
        --model "$MODEL" --model_path "$ckpt" \
        --step 4 --model_seq_len $SEQ_LEN --eval_batch_size $EVAL_BS \
        $TOKEN_ARG 2>&1 | tee "$ppl_out"
    local ppl; ppl=$(parse_ppl "$ppl_out")
    rm -f "$ppl_out"

    local base_ms="N/A" flash_ms="N/A" speedup="N/A"
    if [ "$ckpt" != "original" ]; then
        echo "=== Bench speed: $method keep=$keep ==="
        local bench_out; bench_out=$(mktemp)
        python bench_flashsvd_vs_svd_decode.py \
            --checkpoint "$ckpt" \
            --dtype bf16 --prompt_len 512 --new_tokens 128 --warmup 5 \
            --experimental_flash_dense_attn \
            2>&1 | tee "$bench_out"
        base_ms=$(parse_ms "$bench_out" "SVD")
        flash_ms=$(parse_ms "$bench_out" "FlashSVD")
        speedup=$(python -c "
b, f = '$base_ms', '$flash_ms'
try: print(f'{float(b)/float(f):.3f}')
except: print('N/A')
")
        rm -f "$bench_out"
    fi

    echo "$MODEL_TAG,$method,$keep,$ppl,$base_ms,$flash_ms,$speedup" >> "$CSV"
    echo "  → CSV: $MODEL_TAG,$method,$keep ppl=$ppl base=${base_ms}ms flash=${flash_ms}ms speedup=${speedup}x"
}

# ── init CSV ──────────────────────────────────────────────────────────────────
echo "model,method,keep_ratio,wikitext2_ppl,baseline_ms,flashsvd_ms,speedup" > "$CSV"

# ── baseline PPL ──────────────────────────────────────────────────────────────
eval_and_log "original" "baseline" "1.0"

# ── Step 1: V1 — first ratio also computes & saves profiling_mat ───────────────
echo "=== Compress V1 ratio=0.2 (保存率=0.8) + profiling_mat ==="
python SVDLLM.py --model "$MODEL" --step 1 --ratio 0.2 \
    --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
    2>&1 | tee logs/${MODEL_TAG}_v1_0.8.log

for RATIO in 0.3 0.4 0.5 0.6; do
    KEEP=$(keep_csv $RATIO)
    echo "=== Compress V1 ratio=$RATIO (保存率=$KEEP) ==="
    python SVDLLM.py --model "$MODEL" --step 1 --ratio $RATIO \
        --profiling_mat_path "$PROF_MAT" \
        --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
        2>&1 | tee logs/${MODEL_TAG}_v1_${KEEP}.log
done

# ── Step 2: V2 ────────────────────────────────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(keep_csv $RATIO)
    echo "=== Compress V2 ratio=$RATIO (保存率=$KEEP) ==="
    python SVDLLM.py --model "$MODEL" --step 2 --ratio $RATIO \
        --profiling_mat_path "$PROF_MAT" \
        --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
        2>&1 | tee logs/${MODEL_TAG}_v2_${KEEP}.log
done

# ── Eval all checkpoints (PPL + decode speed) ─────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)   # exact filename suffix from SVDLLM.py
    KEEP_CSV=$(keep_csv $RATIO)     # rounded for display
    eval_and_log "$SAVE_DIR/${MODEL_PREFIX}_whitening_only_${KEEP_FILE}.pt"        "V1" "$KEEP_CSV"
    eval_and_log "$SAVE_DIR/${MODEL_PREFIX}_whitening_then_update_${KEEP_FILE}.pt" "V2" "$KEEP_CSV"
done

echo ""
echo "=== All done. Results: $CSV ==="
cat "$CSV"
