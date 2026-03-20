#!/bin/bash
# SVD-LLM Basis Sharing (step 5)
# Model: meta-llama/Llama-3.1-8B-Instruct
# 保存率: 0.8, 0.7, 0.6, 0.5, 0.4
#
# Usage:
#   bash run_compress_llama31_8b_instruct_bs.sh [HF_TOKEN]
#
# Results CSV: checkpoints/svdllm/llama31_8b_instruct/results_bs.csv

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TAG="Llama-3.1-8B-Instruct"
MODEL_PREFIX="meta_llama_Llama_3.1_8B_Instruct"
SAVE_DIR="checkpoints/svdllm/llama31_8b_instruct"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
EVAL_BS=2
CSV="$SAVE_DIR/results_bs.csv"
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

keep_file() { python -c "print(1 - $1)"; }
keep_csv()  { python -c "print(round(1 - $1, 1))"; }

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
if [ ! -f "$CSV" ]; then
    echo "model,method,keep_ratio,wikitext2_ppl,baseline_ms,flashsvd_ms,speedup" > "$CSV"
    eval_and_log "original" "baseline" "1.0"
else
    echo "=== CSV already exists, skipping baseline eval ==="
fi

# ── BasisSharing compress ─────────────────────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP_FILE}.pt"
    if [ -f "$CKPT" ]; then
        echo "=== BasisSharing checkpoint exists, skipping: $CKPT ==="
    else
        KEEP=$(keep_csv $RATIO)
        echo "=== Compress BasisSharing ratio=$RATIO (keep=$KEEP) ==="
        PROF_ARG=""
        [ -f "$PROF_MAT" ] && PROF_ARG="--profiling_mat_path $PROF_MAT"
        python SVDLLM.py --model "$MODEL" --step 5 --ratio $RATIO \
            $PROF_ARG \
            --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
            2>&1 | tee logs/${MODEL_TAG}_bs_${KEEP}.log
    fi
done

# ── Eval ──────────────────────────────────────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP_FILE=$(keep_file $RATIO)
    KEEP_CSV=$(keep_csv $RATIO)
    eval_and_log "$SAVE_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP_FILE}.pt" "BasisSharing" "$KEEP_CSV"
done

echo ""
echo "=== All done. Results: $CSV ==="
cat "$CSV"
