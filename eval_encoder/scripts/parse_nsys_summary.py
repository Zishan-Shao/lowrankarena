#!/usr/bin/env python3
"""
Parse nsys_summary.txt produced by the nsys profiling loop and print a
compact comparison table with 4 key metrics per profiling point.

Usage:
    python eval_encoder/scripts/parse_nsys_summary.py \
        [--input eval_encoder/eval_results/nsys/nsys_summary.txt] \
        [--out_csv eval_encoder/eval_results/nsys/nsys_parsed.csv]
"""

import argparse
import csv
import os
import re


# ── keywords that identify GEMM / matmul kernels ──────────────────────────
_GEMM_KW   = ["gemm", "cublas", "triton", "matmul", "volta_sgemm",
               "ampere_sgemm", "fp16", "bf16_gemm"]
_MEMCPY_KW = ["memcpy", "memset"]
_SYNC_KW   = ["sync", "barrier", "wait"]


def _first_float(s: str):
    """Return the first float found in string s, or None."""
    m = re.search(r"[\d]+\.[\d]*|[\d]*\.[\d]+|[\d]+", s)
    return float(m.group()) if m else None


def _second_int(s: str):
    """Return the second integer token in string s (call-count column), or 0."""
    nums = re.findall(r"\d+", s)
    return int(nums[1]) if len(nums) > 1 else 0


def parse_section(body: str) -> dict:
    """
    Parse one profiling-point section from nsys_summary.txt.

    Returns a dict with:
        total_kernel_calls  – sum of call counts across all unique kernels
        gemm_time_pct       – sum of time% for GEMM-family kernels
        memcpy_time_pct     – sum of time% for memcpy/memset kernels
        sync_time_pct       – sum of time% for sync/barrier kernels
        top1_kernel         – name of the most time-consuming kernel
        top1_time_pct       – its time%
        n_unique_kernels    – number of unique kernel names
    """
    # Lines that start with a number are data rows in the nsys table
    data_lines = [l for l in body.splitlines() if re.match(r"\s*[\d]", l)]

    total_calls   = 0
    gemm_pct      = 0.0
    memcpy_pct    = 0.0
    sync_pct      = 0.0
    top1_name     = "n/a"
    top1_pct      = 0.0

    for i, line in enumerate(data_lines):
        pct  = _first_float(line) or 0.0
        calls = _second_int(line)
        name  = line.strip()

        total_calls += calls

        lo = name.lower()
        if any(k in lo for k in _GEMM_KW):
            gemm_pct += pct
        if any(k in lo for k in _MEMCPY_KW):
            memcpy_pct += pct
        if any(k in lo for k in _SYNC_KW):
            sync_pct += pct

        if i == 0:          # first row = highest time%
            top1_name = name[:60]
            top1_pct  = pct

    n_unique = len(data_lines)
    frag = round(total_calls / n_unique, 2) if n_unique > 0 else 0.0
    return {
        "total_kernel_calls":  total_calls,
        "n_unique_kernels":    n_unique,
        "fragmentation_ratio": frag,   # total_calls / n_unique: higher = more repeated small launches
        "gemm_time_pct":       round(gemm_pct,   1),
        "memcpy_time_pct":     round(memcpy_pct, 1),
        "sync_time_pct":       round(sync_pct,   1),
        "top1_kernel":         top1_name,
        "top1_time_pct":       round(top1_pct,   1),
    }


def parse_summary_file(path: str) -> list[dict]:
    """Split nsys_summary.txt by section headers and parse each block."""
    txt = open(path, encoding="utf-8", errors="replace").read()

    # Section header written by the shell loop: ════ TAG ════
    parts = re.split(r"═{4,}\s+(\S+)\s+═{4,}", txt)

    # parts[0]  = text before first header (discard)
    # parts[1]  = tag1,  parts[2] = body1
    # parts[3]  = tag2,  parts[4] = body2  …
    rows = []
    for i in range(1, len(parts) - 1, 2):
        tag  = parts[i].strip()
        body = parts[i + 1]
        metrics = parse_section(body)
        rows.append({"point": tag, **metrics})
    return rows


def print_table(rows: list[dict]):
    hdr = (f"{'Point':<32} {'Calls':>7} {'Uniq':>5} {'Frag':>6} "
           f"{'GEMM%':>7} {'Memcpy%':>8} {'Sync%':>6}  "
           f"{'Top-1 kernel (truncated)':<50} {'Top1%':>6}")
    print(hdr)
    print("─" * len(hdr))
    for r in rows:
        print(
            f"{r['point']:<32} "
            f"{r['total_kernel_calls']:>7} "
            f"{r['n_unique_kernels']:>5} "
            f"{r['fragmentation_ratio']:>6.2f} "
            f"{r['gemm_time_pct']:>6.1f}% "
            f"{r['memcpy_time_pct']:>7.1f}% "
            f"{r['sync_time_pct']:>5.1f}%  "
            f"{r['top1_kernel']:<50} "
            f"{r['top1_time_pct']:>5.1f}%"
        )


def write_csv(rows: list[dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] Written → {path}")


def main():
    p = argparse.ArgumentParser(description="Parse nsys_summary.txt into a comparison table")
    p.add_argument("--input",   default="eval_encoder/eval_results/nsys/nsys_summary.txt")
    p.add_argument("--out_csv", default="eval_encoder/eval_results/nsys/nsys_parsed.csv")
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"[error] File not found: {args.input}\n"
                         "Run the nsys profiling loop first.")

    rows = parse_summary_file(args.input)
    if not rows:
        raise SystemExit("[error] No sections found in summary file. "
                         "Check that nsys_summary.txt contains ════ TAG ════ headers.")

    print_table(rows)
    if args.out_csv:
        write_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
