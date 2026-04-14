#!/usr/bin/env bash
# run_all_compress.sh
# 顺序跑所有模型的全部压缩方法：V1, V1update, V2, ASVD, Basis Sharing
# 每个子脚本失败后继续执行下一个（set +e）
#
# Usage:
#   bash run_all_compress.sh [HF_TOKEN]
#   bash run_all_compress.sh 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

HF_TOKEN="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_script() {
    local label="$1"
    local dir="$2"
    local script="$3"
    shift 3

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  $label"
    echo "════════════════════════════════════════════════════"
    set +e
    (cd "$dir" && bash "$script" "$@")
    local rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "[WARN] $label exited with code $rc, continuing..."
    fi
    RESULTS+=("$([ $rc -eq 0 ] && echo '✓' || echo '✗')  $label")
}

RESULTS=()

SVDLLM="$ROOT/baselines/SVD-LLM"
ASVD="$ROOT/baselines/ASVD"

# ── Llama-2-7B ────────────────────────────────────────────────────────────────
run_script "Llama-2-7B   V1 + V1update"  "$SVDLLM" run_compress_llama2_7b.sh          "$HF_TOKEN"
run_script "Llama-2-7B   V2"             "$SVDLLM" run_compress_llama2_7b_v2.sh        "$HF_TOKEN"
run_script "Llama-2-7B   Basis Sharing"  "$SVDLLM" run_compress_llama2_7b_basissharing.sh "$HF_TOKEN"
run_script "Llama-2-7B   ASVD"          "$ASVD"   run_asvd_llama2_7b.sh               "$HF_TOKEN"

# ── Llama-3.1-8B ──────────────────────────────────────────────────────────────
run_script "Llama-3.1-8B  V1 + V1update" "$SVDLLM" run_compress_llama31_8b.sh         "$HF_TOKEN"
run_script "Llama-3.1-8B  V2"            "$SVDLLM" run_compress_llama31_8b_v2.sh       "$HF_TOKEN"
run_script "Llama-3.1-8B  Basis Sharing" "$SVDLLM" run_compress_llama31_8b_basissharing.sh "$HF_TOKEN"
run_script "Llama-3.1-8B  ASVD"         "$ASVD"   run_asvd_llama31_8b.sh              "$HF_TOKEN"

# ── Llama-3.1-8B-Instruct ─────────────────────────────────────────────────────
run_script "Llama-3.1-8B-Instruct  V1 + V1update" "$SVDLLM" run_compress_llama31_8b_instruct.sh    "$HF_TOKEN"
run_script "Llama-3.1-8B-Instruct  V2"            "$SVDLLM" run_compress_llama31_8b_instruct_v2.sh "$HF_TOKEN"
run_script "Llama-3.1-8B-Instruct  Basis Sharing" "$SVDLLM" run_compress_llama31_8b_instruct_bs.sh "$HF_TOKEN"
run_script "Llama-3.1-8B-Instruct  ASVD"         "$ASVD"   run_asvd_llama31_8b_instruct.sh        "$HF_TOKEN"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
