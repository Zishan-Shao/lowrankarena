#!/usr/bin/env python3
"""
Parse per-kernel ncu --csv output files produced by expD.sh (ncu phase).

For each ncu_raw_<point_tag>.csv in --ncu_dir, extracts per-kernel occupancy
and CTA parallelism metrics, classifies kernels into categories (triton /
gemm / other), and appends aggregated rows to --out_csv.

ncu --csv column layout (space/comma delimited):
  "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
  "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"

Metrics captured (set by expD.sh NCU_METRICS):
  sm__ctas_active.avg
  sm__warps_active.avg.pct_of_peak_sustained_active
  launch__occupancy_limit_registers
  launch__occupancy_limit_shared_mem
  launch__occupancy_limit_warps
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
  smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct

Output CSV columns:
  point_tag, method, backend, kernel_name, kernel_name_raw, kernel_hash, kernel_category,
  sm__ctas_active_avg, occupancy_pct,
  limit_registers, limit_shared_mem, limit_warps, limit_type,
  stall_mem_pct, stall_math_pct, n_launches

Column notes:
  limit_type    : binding occupancy constraint = argmin(limit_registers,
                  limit_shared_mem, limit_warps); tie-break reg > smem > warps.
                  Source: NCU launch__occupancy_limit_* metrics.
  stall_mem_pct : % of active cycles stalled waiting for memory (scoreboard).
                  Source: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
                  (normalized to active cycles per warp, not wall time).
  stall_math_pct: % of active cycles stalled due to math-pipe throttle.
                  Source: smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
  kernel_category: triton | gemm | other  — classified by kernel name substring
                  (see _TRITON_KW / _GEMM_KW below for exact rules).

Usage
-----
python eval_encoder/scripts/parse_ncu_csv.py \\
    --ncu_dir eval_encoder/eval_results/nsys \\
    --task    mnli \\
    --out_csv eval_encoder/eval_results/expD_ncu.csv
"""

import argparse
import csv
import glob
import os
import re
import sys


# ── Kernel classification ──────────────────────────────────────────────────────
# Rules (applied in priority order):
#   triton : kernel name contains "triton_" | "fused_ffn" | "_demo_attn"
#            — FlashSVD Triton fused attention / FFN kernels
#   gemm   : kernel name contains "gemm" | "cublas" | "cutlass" | "matmul" |
#            "volta_sgemm" | "ampere_sgemm" | "sm80_xmma"
#            — cuBLAS / CUTLASS raw matrix-multiply dispatches
#   other  : everything else (softmax, layernorm, elementwise, memcpy, …)
_TRITON_KW = ["triton_", "fused_ffn", "_demo_attn"]
_GEMM_KW   = ["gemm", "cublas", "cutlass", "matmul",
               "volta_sgemm", "ampere_sgemm", "sm80_xmma"]


def _classify_kernel(name: str) -> str:
    n = name.lower()
    if any(k in n for k in _TRITON_KW):
        return "triton"
    if any(k in n for k in _GEMM_KW):
        return "gemm"
    return "other"


def _parse_point_tag(filename: str, task: str):
    """
    ncu_raw_<point_tag>.csv  →  (point_tag, method, backend)

    Point tag format: <task>_<method>_<backend>
    e.g. mnli_svd_naive, mnli_adasvd_flashsvd15
    """
    base = os.path.basename(filename)
    m = re.match(r"ncu_raw_(.+)\.csv$", base)
    if not m:
        return None, None, None
    tag = m.group(1)
    # strip task prefix
    prefix = task + "_"
    rest = tag[len(prefix):] if tag.startswith(prefix) else tag
    # backend is last token; method is everything before last underscore
    parts = rest.rsplit("_", 1)
    if len(parts) == 2:
        method, backend = parts
    else:
        method, backend = rest, "unknown"
    return tag, method, backend


def _safe_float(v: str) -> float:
    try:
        return float(v.replace(",", ".").strip())
    except (ValueError, AttributeError):
        return float("nan")


def parse_ncu_csv(filepath: str) -> dict:
    """
    Read one ncu raw CSV file.

    Returns: dict keyed by kernel_name → dict of metric → value
             (averaged over multiple launches of the same kernel)

    ncu --csv produces rows like:
      "ID","...","Kernel Name","...","Metric Name","Metric Unit","Metric Value"
    Column indices may vary; we look for the header row to find them.
    """
    kernel_data: dict[str, dict[str, list]] = {}

    with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # ncu sometimes prepends warning lines before the CSV header
    # Find the header line that contains "Kernel Name"
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Kernel Name" in line and "Metric Name" in line:
            header_idx = i
            break

    if header_idx is None:
        return {}

    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        kernel = row.get("Kernel Name", "").strip().strip('"')
        metric = row.get("Metric Name", "").strip().strip('"')
        value  = row.get("Metric Value", "").strip().strip('"')

        if not kernel or not metric:
            continue

        if kernel not in kernel_data:
            kernel_data[kernel] = {}
        if metric not in kernel_data[kernel]:
            kernel_data[kernel][metric] = []
        kernel_data[kernel][metric].append(_safe_float(value))

    # Average over launches
    result = {}
    for kernel, metrics in kernel_data.items():
        result[kernel] = {m: (sum(vs) / len(vs)) for m, vs in metrics.items()
                          if vs}
    return result


def _limit_type(row_metrics: dict) -> str:
    """
    Determine occupancy bottleneck from NCU launch__occupancy_limit_* metrics.

    Each launch__occupancy_limit_<X> reports the theoretical maximum achievable
    warp occupancy (as a fraction of peak) if only constraint X were active.
    The *smallest* value is the binding constraint — it most tightly caps
    the achieved occupancy.  Tie-breaking priority: registers > shared_mem > warps.

    Reference: NVIDIA Nsight Compute metric documentation,
    "Occupancy Limiter" section.
    """
    limits = {
        "registers":  row_metrics.get("launch__occupancy_limit_registers",  float("nan")),
        "shared_mem": row_metrics.get("launch__occupancy_limit_shared_mem", float("nan")),
        "warps":      row_metrics.get("launch__occupancy_limit_warps",      float("nan")),
    }
    # Filter out NaN / zero (metric not collected or not applicable)
    valid = {k: v for k, v in limits.items() if v == v and v > 0}
    if not valid:
        return "unknown"
    min_val = min(valid.values())
    # Priority order for ties: registers > shared_mem > warps
    for key in ("registers", "shared_mem", "warps"):
        if key in valid and valid[key] == min_val:
            return key
    return "unknown"


def process_file(filepath: str, task: str) -> list[dict]:
    """Parse one ncu raw CSV → list of output rows (one per kernel)."""
    tag, method, backend = _parse_point_tag(filepath, task)
    if tag is None:
        return []

    kernel_metrics = parse_ncu_csv(filepath)
    if not kernel_metrics:
        print(f"  [warn] No metrics parsed from {filepath}")
        return []

    rows = []
    for kernel_name, metrics in sorted(kernel_metrics.items()):
        n_launches = max(
            len(v) for v in [metrics] if isinstance(v, dict)
        ) if False else 1  # already averaged; mark n_launches=1 per unique kernel

        rows.append({
            "point_tag":          tag,
            "method":             method,
            "backend":            backend,
            "kernel_name":        kernel_name[:80],
            "kernel_name_raw":    kernel_name,
            "kernel_hash":        f"{hash(kernel_name) & 0xFFFFFFFF:08x}",  # 32-bit hex
            "kernel_category":    _classify_kernel(kernel_name),
            "sm__ctas_active_avg": f"{metrics.get('sm__ctas_active.avg', float('nan')):.2f}",
            "occupancy_pct":      f"{metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active', float('nan')):.2f}",
            "limit_registers":    f"{metrics.get('launch__occupancy_limit_registers', float('nan')):.2f}",
            "limit_shared_mem":   f"{metrics.get('launch__occupancy_limit_shared_mem', float('nan')):.2f}",
            "limit_warps":        f"{metrics.get('launch__occupancy_limit_warps', float('nan')):.2f}",
            "limit_type":         _limit_type(metrics),
            "stall_mem_pct":      f"{metrics.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct', float('nan')):.2f}",
            "stall_math_pct":     f"{metrics.get('smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct', float('nan')):.2f}",
        })
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Parse ncu raw CSVs → expD_ncu.csv summary")
    p.add_argument("--ncu_dir", required=True,
                   help="Directory containing ncu_raw_*.csv files")
    p.add_argument("--task",    default="mnli",
                   help="Task prefix used in point tag filenames (default: mnli)")
    p.add_argument("--out_csv", required=True,
                   help="Output CSV path")
    return p.parse_args()


def main():
    args = parse_args()

    pattern = os.path.join(args.ncu_dir, f"ncu_raw_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[warn] No ncu_raw_*.csv files found in {args.ncu_dir}")
        sys.exit(0)

    all_rows = []
    for f in files:
        print(f"[parse] {f}")
        rows = process_file(f, args.task)
        print(f"        → {len(rows)} kernel rows")
        all_rows.extend(rows)

    if not all_rows:
        print("[warn] No rows to write.")
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"[csv] {len(all_rows)} rows → {args.out_csv}")

    # Print per-category summary table
    from collections import defaultdict
    summary: dict = defaultdict(list)
    for r in all_rows:
        key = (r["point_tag"], r["kernel_category"])
        try:
            occ = float(r["occupancy_pct"])
            if occ == occ:  # not nan
                summary[key].append(occ)
        except ValueError:
            pass

    print("\n  occupancy summary (mean % by category):")
    print(f"  {'point_tag':<35} {'category':<10} {'mean_occ%':>10}  {'n':>4}")
    print("  " + "-" * 65)
    for (tag, cat), vals in sorted(summary.items()):
        mean_occ = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {tag:<35} {cat:<10} {mean_occ:>10.1f}  {len(vals):>4}")


if __name__ == "__main__":
    main()
