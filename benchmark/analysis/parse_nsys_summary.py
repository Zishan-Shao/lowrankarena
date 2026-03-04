#!/usr/bin/env python3
"""
Parse nsys_summary.txt produced by the nsys profiling loop and print a
compact comparison table with key metrics per profiling point.

nsys cuda_gpu_kern_sum column layout (tab/space delimited):
  Time(%)  Total Time(ns)  Instances  Avg(ns)  Med(ns)  Min(ns)  Max(ns)  StdDev(ns)  Name

Key metrics extracted:
  total_kernel_calls  – sum of Instances across all rows
  n_unique_kernels    – number of distinct kernel rows in the table
  n_gemm_variants     – distinct GEMM/CUTLASS kernel configs (reflects rank heterogeneity)
  fragmentation_ratio – total_calls / n_unique (repeated-launch density)
  triton_time_pct     – time% in FlashSVD Triton fused kernels (fused_ffn / _demo_attn)
  gemm_time_pct       – time% in cuBLAS/CUTLASS GEMM kernels
  memcpy_time_pct     – time% in memcpy/memset
  sync_time_pct       – time% in sync/barrier/wait
  top1_kernel         – most time-consuming kernel name (truncated to 55 chars)
  top1_time_pct       – its time%

Usage:
    python benchmark/parse_nsys_summary.py \\
        [--input experiments/nsys/nsys_summary_fair.txt] \\
        [--out_csv experiments/expD.csv]
"""

import argparse
import csv
import os
import re


# ── keyword lists ──────────────────────────────────────────────────────────────
# FlashSVD Triton fused kernels (NOT raw GEMMs — handled separately)
_TRITON_KW = ["fused_ffn", "_demo_attn", "triton_"]

# cuBLAS / CUTLASS GEMM kernels (raw matrix-multiply dispatches)
_GEMM_KW   = ["gemm", "cublas", "cutlass", "matmul",
               "volta_sgemm", "ampere_sgemm"]

_MEMCPY_KW = ["memcpy", "memset"]
_SYNC_KW   = ["sync", "barrier", "wait"]


# ── row parser ─────────────────────────────────────────────────────────────────
# nsys data rows start with: <spaces> Time% <spaces> TotalTime(commas) <spaces> Instances ...
# Total Time (ns) uses comma thousands separators → can't use bare \d+
_ROW_RE = re.compile(
    r'^\s*([\d.]+)'        # group 1: Time (%)
    r'\s+([\d,]+)'         # group 2: Total Time (ns)  – may contain commas
    r'\s+([\d,]+)'         # group 3: Instances        – may contain commas
)


def _parse_row(line: str):
    """
    Parse one data row from the nsys cuda_gpu_kern_sum table.

    Returns (pct: float, calls: int, name: str) or None if the line is
    not a data row (header, separator, blank, etc.).

    Name extraction: split the stripped line on runs of 2+ spaces and take
    the last token.  Within a kernel name, spaces appear as single spaces
    (e.g. template argument lists), so splitting on 2+ spaces cleanly
    separates the 8 numeric columns from the Name column.
    """
    m = _ROW_RE.match(line)
    if not m:
        return None

    pct   = float(m.group(1))
    calls = int(m.group(3).replace(',', ''))

    parts = re.split(r'\s{2,}', line.strip())
    name  = parts[-1].strip() if len(parts) >= 2 else ""

    return pct, calls, name


def parse_section(body: str) -> dict:
    """Parse one ════ TAG ════ section from nsys_summary.txt."""
    parsed_rows = []
    for line in body.splitlines():
        r = _parse_row(line)
        if r is not None:
            parsed_rows.append(r)

    total_calls   = 0
    triton_pct    = 0.0
    gemm_pct      = 0.0
    memcpy_pct    = 0.0
    sync_pct      = 0.0
    top1_name     = "n/a"
    top1_pct      = 0.0
    gemm_variants = set()   # distinct GEMM kernel config strings

    for i, (pct, calls, name) in enumerate(parsed_rows):
        total_calls += calls
        lo = name.lower()

        if any(k in lo for k in _TRITON_KW):
            triton_pct += pct
        if any(k in lo for k in _GEMM_KW):
            gemm_pct += pct
            gemm_variants.add(name[:100])   # deduplicate by (truncated) name
        if any(k in lo for k in _MEMCPY_KW):
            memcpy_pct += pct
        if any(k in lo for k in _SYNC_KW):
            sync_pct += pct

        if i == 0:      # rows are sorted by Time% descending → first = top-1
            top1_name = name[:55]
            top1_pct  = pct

    n_unique = len(parsed_rows)
    frag     = round(total_calls / n_unique, 1) if n_unique > 0 else 0.0

    return {
        "total_kernel_calls":  total_calls,
        "n_unique_kernels":    n_unique,
        "n_gemm_variants":     len(gemm_variants),
        # fragmentation: how many repeated launches per unique kernel type
        # higher → more homogeneous workload (one kernel dominates call count)
        # lower  → more heterogeneous (many different kernels, each called once)
        "fragmentation_ratio": frag,
        "triton_time_pct":     round(triton_pct,  1),
        "gemm_time_pct":       round(gemm_pct,    1),
        "memcpy_time_pct":     round(memcpy_pct,  1),
        "sync_time_pct":       round(sync_pct,    1),
        "top1_kernel":         top1_name,
        "top1_time_pct":       round(top1_pct,    1),
    }


def parse_summary_file(path: str) -> list[dict]:
    """Split nsys_summary.txt by ════ TAG ════ headers and parse each block."""
    txt = open(path, encoding="utf-8", errors="replace").read()

    # Section headers written by the profiling shell loop: ════ TAG ════
    parts = re.split(r"═{4,}\s+(\S+)\s+═{4,}", txt)

    # Layout: parts[0]=preamble, parts[1]=tag1, parts[2]=body1, parts[3]=tag2 …
    rows = []
    for i in range(1, len(parts) - 1, 2):
        tag     = parts[i].strip()
        body    = parts[i + 1]
        metrics = parse_section(body)
        rows.append({"point": tag, **metrics})
    return rows


def print_table(rows: list[dict]):
    hdr = (
        f"{'Point':<28} {'Calls':>8} {'Uniq':>5} {'GEMMvar':>8} {'Frag':>7}  "
        f"{'Triton%':>8} {'GEMM%':>6} {'Mem%':>5} {'Sync%':>6}  "
        f"{'Top-1 kernel':<55} {'Top1%':>6}"
    )
    print(hdr)
    print("─" * len(hdr))
    for r in rows:
        print(
            f"{r['point']:<28} "
            f"{r['total_kernel_calls']:>8,} "
            f"{r['n_unique_kernels']:>5} "
            f"{r['n_gemm_variants']:>8} "
            f"{r['fragmentation_ratio']:>7.1f}  "
            f"{r['triton_time_pct']:>7.1f}% "
            f"{r['gemm_time_pct']:>5.1f}% "
            f"{r['memcpy_time_pct']:>4.1f}% "
            f"{r['sync_time_pct']:>5.1f}%  "
            f"{r['top1_kernel']:<55} "
            f"{r['top1_time_pct']:>5.1f}%"
        )
    print()
    # ── interpretation guide ───────────────────────────────────────────────
    print("Notes:")
    print("  GEMMvar  – distinct cuBLAS/CUTLASS kernel configurations.")
    print("             Higher = more rank-heterogeneous workload (e.g. AdaSVD).")
    print("  Frag     – total_calls / n_unique.  High → few kernel types, many calls each.")
    print("  Triton%  – time in FlashSVD fused kernels (fused_ffn / _demo_attn).")
    print("  GEMM%    – time in raw cuBLAS/CUTLASS matrix-multiply kernels.")


def write_csv(rows: list[dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] Written → {path}")


def main():
    p = argparse.ArgumentParser(description="Parse nsys cuda_gpu_kern_sum summary into a comparison table")
    p.add_argument("--input",   default="experiments/nsys/nsys_summary.txt")
    p.add_argument("--out_csv", default="experiments/expD.csv")
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(
            f"[error] File not found: {args.input}\n"
            "Run the nsys profiling loop first."
        )

    rows = parse_summary_file(args.input)
    if not rows:
        raise SystemExit(
            "[error] No ════ TAG ════ sections found.\n"
            "Ensure nsys_summary.txt was written by the profiling shell loop."
        )

    print_table(rows)
    if args.out_csv:
        write_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
