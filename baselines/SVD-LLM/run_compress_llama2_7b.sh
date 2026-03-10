#!/bin/bash
# SVD-LLM V1 (whitening only) + V2 (whitening + local update)
# Model: meta-llama/Llama-2-7b-hf
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Usage:
#   bash run_compress_llama2_7b.sh [HF_TOKEN]
#
# Results CSV: checkpoints/svdllm/llama2_7b/results.csv
# Columns: model,method,keep_ratio,wikitext2_ppl,baseline_ms,flashsvd_ms,speedup

set -e
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-2-7b-hf"
MODEL_TAG="Llama-2-7b"
SAVE_DIR="checkpoints/svdllm/llama2_7b"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
EVAL_BS=4
CSV="$SAVE_DIR/results.csv"
PROF_MAT="$SAVE_DIR/meta_llama_Llama_2_7b_hf_profiling_wikitext2_256_0.pt"
MODEL_PREFIX="meta_llama_Llama_2_7b_hf"

mkdir -p "$SAVE_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

# ── helper: parse a value from output file ────────────────────────────────────
parse_ppl() {
    python -c "
import re
txt = open('$1').read()
m = re.search(r\"wikitext2'?[^:]*:\\s*([0-9]+\\.?[0-9]*)\", txt)
print(m.group(1) if m else 'N/A')
"
}

parse_ms() {
    python -c "
import re
txt = open('$1').read()
m = re.search(r'$2 decode: ([0-9]+\\.?[0-9]*) ms/token', txt)
print(m.group(1) if m else 'N/A')
"
}

# ── helper: eval PPL + speed, append row to CSV ───────────────────────────────
eval_and_log() {
    local ckpt="$1"
    local method="$2"
    local keep="$3"

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
    echo "  → $MODEL_TAG,$method,$keep,ppl=$ppl,base=${base_ms}ms,flash=${flash_ms}ms,speedup=${speedup}x"
}

# ── init CSV ──────────────────────────────────────────────────────────────────
echo "model,method,keep_ratio,wikitext2_ppl,baseline_ms,flashsvd_ms,speedup" > "$CSV"

# ── baseline ──────────────────────────────────────────────────────────────────
eval_and_log "original" "baseline" "1.0"

# ── Step 1: V1 (whitening only) ───────────────────────────────────────────────
echo "=== Compress V1 ratio=0.2 (保存率=0.8) + profiling_mat ==="
python SVDLLM.py --model "$MODEL" --step 1 --ratio 0.2 \
    --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
    2>&1 | tee logs/${MODEL_TAG}_v1_0.8.log

for RATIO in 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(round(1-$RATIO, 1))")
    echo "=== Compress V1 ratio=$RATIO (保存率=$KEEP) ==="
    python SVDLLM.py --model "$MODEL" --step 1 --ratio $RATIO \
        --profiling_mat_path "$PROF_MAT" \
        --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
        2>&1 | tee logs/${MODEL_TAG}_v1_${KEEP}.log
done

# ── Step 2: V2 (whitening + local update) ─────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(round(1-$RATIO, 1))")
    echo "=== Compress V2 ratio=$RATIO (保存率=$KEEP) ==="
    python SVDLLM.py --model "$MODEL" --step 2 --ratio $RATIO \
        --profiling_mat_path "$PROF_MAT" \
        --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
        2>&1 | tee logs/${MODEL_TAG}_v2_${KEEP}.log
done

# ── Step 4 + bench: eval all checkpoints ──────────────────────────────────────
for KEEP in 0.8 0.7 0.6 0.5 0.4; do
    eval_and_log "$SAVE_DIR/${MODEL_PREFIX}_whitening_only_${KEEP}.pt"        "V1" "$KEEP"
    eval_and_log "$SAVE_DIR/${MODEL_PREFIX}_whitening_then_update_${KEEP}.pt" "V2" "$KEEP"
done

echo ""
echo "=== All done. Results: $CSV ==="
cat "$CSV"
