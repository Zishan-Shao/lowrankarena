#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all_figures.sh  —  Regenerate all paper figures from experiment CSVs
#
# Usage (from repo root):
#   bash benchmark/figures/run_all_figures.sh
#
# Output: experiments/figs/figures/
#
# Prerequisites:
#   - expA:  experiments/results/glue_summary.csv  (from collect_glue_results.py)
#   - expB:  experiments/results/expB.csv
#   - expC:  experiments/results/expC_seqlen.csv   experiments/results/expC_batch.csv
#   - expD:  experiments/results/expD_mnli_bf16_s512_b32.csv
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTDIR="experiments/figs/figures"
RESULTS="experiments/results"
mkdir -p "${OUTDIR}"

echo "══════════════════════════════════════════════════════════════════════"
echo "  Generating figures → ${OUTDIR}/"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0; FAIL=0

run() {
    local label="$1"; shift
    echo "── ${label}"
    if "$@"; then
        echo "   ✓ done"
        OK=$((OK + 1))
    else
        echo "   ✗ FAILED"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# ── expA: accuracy figures (fig1-6, GLUE G-AVG, MRPC collapse, Pareto) ────────
run "expA: fig01-06 (accuracy + memory + Pareto)" \
    python benchmark/figures/gen_figures.py

# ── expB: backend comparison (fig09-12, fig13) ────────────────────────────────
if [[ -f "${RESULTS}/expB.csv" ]]; then
    run "expB: fig09-12 backend latency/throughput/speedup/memory (mnli+stsb, bf16, seq512)" \
        python benchmark/figures/plot_backend_sweep.py \
            --csv     "${RESULTS}/expB.csv" \
            --tasks   mnli stsb \
            --methods svd fwsvd drone adasvd \
            --dtype   bf16 \
            --seq_len 512 \
            --outdir  "${OUTDIR}"

    run "expB: fig13 FLOPs breakdown (mnli, bf16, seq512)" \
        python benchmark/figures/plot_flops_breakdown.py \
            --csv     "${RESULTS}/expB.csv" \
            --task    mnli \
            --dtype   bf16 \
            --seq_len 512 \
            --outdir  "${OUTDIR}"
else
    echo "── expB: skipped (${RESULTS}/expB.csv not found)"
    echo ""
fi

# ── expC: seq-len + dtype scaling (fig07-08, fig14) ──────────────────────────
run "expC: fig07-08 seqlen/batch scaling" \
    python benchmark/figures/plot_seqlen_scaling.py

if [[ -f "${RESULTS}/expC_seqlen.csv" ]]; then
    run "expC: fig14 dtype×backend memory/throughput scaling" \
        python benchmark/figures/plot_dtype_scaling.py
else
    echo "── expC dtype scaling: skipped (${RESULTS}/expC_seqlen.csv not found)"
    echo ""
fi

# ── expD: nsys kernel analysis (fig15) ────────────────────────────────────────
_EXPD_CSV="${RESULTS}/expD_mnli_bf16_s512_b32.csv"
if [[ ! -f "${_EXPD_CSV}" ]]; then
    _EXPD_CSV="$(ls ${RESULTS}/expD*.csv 2>/dev/null | head -1 || true)"
fi
if [[ -n "${_EXPD_CSV}" && -f "${_EXPD_CSV}" ]]; then
    run "expD: fig15 nsys kernel analysis" \
        python benchmark/figures/plot_nsys_kernel.py \
            --csv     "${_EXPD_CSV}" \
            --outdir  "${OUTDIR}"
else
    echo "── expD: skipped (no expD CSV found in ${RESULTS}/)"
    echo ""
fi

echo "══════════════════════════════════════════════════════════════════════"
echo "  Done  success=${OK}  failed=${FAIL}"
echo "  Figures → ${OUTDIR}/"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
