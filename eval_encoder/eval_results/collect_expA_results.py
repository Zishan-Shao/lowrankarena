#!/usr/bin/env python3
"""
Collect stage1 (no fine-tune) accuracy from expA.csv into a summary table.

Reads expA.csv (written by run_encoder_benchmark.py), filters backend=naive,
takes the LATEST row per (method, qkv_mode, task), and prints a paper-style
accuracy table covering GLUE + SuperGLUE/HANS/ANLI.

Run from repo root:
    python eval_encoder/eval_results/collect_expA_results.py
    python eval_encoder/eval_results/collect_expA_results.py --csv /path/to/expA.csv
    python eval_encoder/eval_results/collect_expA_results.py --out summary.csv
"""
import os
import csv
import argparse
from collections import defaultdict

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "expA.csv")

GLUE_TASKS      = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"]
SUPERGLUE_TASKS = ["boolq", "rte_sg", "wic", "hans", "anli_r1", "anli_r2", "anli_r3"]
ALL_TASKS       = GLUE_TASKS + SUPERGLUE_TASKS

NORMALIZE_TASKS = {"cola", "stsb"}   # MCC / Pearson → (v+1)/2 before G-AVG

TASK_HDR = {
    "cola":    "CoLA",  "sst2":    "SST-2",  "mrpc":    "MRPC",
    "qqp":     "QQP",   "mnli":    "MNLI",   "qnli":    "QNLI",
    "rte":     "RTE",   "stsb":    "STS-B",
    "boolq":   "BoolQ", "rte_sg":  "RTE-SG", "wic":     "WiC",
    "hans":    "HANS",  "anli_r1": "ANLI-R1","anli_r2": "ANLI-R2","anli_r3": "ANLI-R3",
}

METHOD_ORDER = ["dense", "svd", "fwsvd", "drone", "adasvd"]


def compute_gavg(scores):
    """Normalized G-AVG over GLUE tasks that are present in scores."""
    vals = []
    for t in GLUE_TASKS:
        v = scores.get(t)
        if v is not None:
            vals.append((v + 1) / 2 if t in NORMALIZE_TASKS else v)
    return sum(vals) / len(vals) if vals else None


def load_csv(path, backend_filter, qkv_filter):
    """Return dict: (method, qkv_mode) → task → latest metric_value."""
    # best[(method, qkv_mode, task)] = (timestamp, value)
    best = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("backend", "") != backend_filter:
                continue
            task = row.get("task", "")
            if task not in ALL_TASKS:
                continue
            method   = row.get("method", "")
            qkv_mode = row.get("qkv_mode", "")
            if qkv_filter and qkv_mode != qkv_filter:
                continue
            ts = row.get("timestamp", "")
            try:
                val = float(row["metric_value"])
            except (KeyError, ValueError, TypeError):
                continue
            key = (method, qkv_mode, task)
            if key not in best or ts > best[key][0]:
                best[key] = (ts, val)

    groups = defaultdict(dict)
    for (method, qkv_mode, task), (_, val) in best.items():
        groups[(method, qkv_mode)][task] = val
    return groups


def print_table(groups, backend, present_tasks):
    """Print one table per qkv_mode found in groups."""
    qkv_modes = sorted(set(qkv for _, qkv in groups))

    for qkv in qkv_modes:
        col_tasks = [t for t in ALL_TASKS if t in present_tasks]

        sep = "=" * (9 + len(col_tasks) * 10 + 10)
        print(f"\n{sep}")
        print(f"  qkv_mode={qkv}   backend={backend}   (stage1 / no fine-tune)")
        print(sep)

        hdr = f"{'Method':<9}" + "".join(f"  {TASK_HDR.get(t,t):>8}" for t in col_tasks)
        hdr += f"  {'G-AVG':>7}"
        print(hdr)
        print("-" * len(hdr))

        for method in METHOD_ORDER:
            key = (method, qkv)
            if key not in groups:
                continue
            scores = groups[key]
            row = f"{method:<9}"
            row += "".join(
                (f"  {scores[t]:8.4f}" if t in scores else f"  {'---':>8}")
                for t in col_tasks
            )
            gavg = compute_gavg(scores)
            row += f"  {gavg:7.4f}" if gavg is not None else f"  {'---':>7}"
            print(row)


def save_csv(groups, out_path, present_tasks):
    col_tasks = [t for t in ALL_TASKS if t in present_tasks]
    fieldnames = ["method", "qkv_mode"] + col_tasks + ["G-AVG_glue"]
    rows = []
    for qkv in sorted(set(q for _, q in groups)):
        for method in METHOD_ORDER:
            key = (method, qkv)
            if key not in groups:
                continue
            scores = groups[key]
            rows.append({
                "method":   method,
                "qkv_mode": qkv,
                **{t: round(scores[t], 6) if t in scores else "" for t in col_tasks},
                "G-AVG_glue": round(gavg, 6) if (gavg := compute_gavg(scores)) else "",
            })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}  ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",     default=DEFAULT_CSV,
                   help=f"Path to expA.csv  (default: {DEFAULT_CSV})")
    p.add_argument("--backend", default="naive",
                   help="Backend to filter for accuracy (default: naive)")
    p.add_argument("--qkv_mode", default="",
                   help="Filter by qkv_mode (default: show all)")
    p.add_argument("--out",     default="",
                   help="Output CSV path (default: print only)")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[error] File not found: {args.csv}")
        return

    groups = load_csv(args.csv, args.backend, args.qkv_mode)
    if not groups:
        print(f"[warn] No data found in {args.csv} for backend={args.backend}")
        return

    # collect which tasks actually appear across all rows
    present_tasks = set()
    for scores in groups.values():
        present_tasks |= set(scores.keys())

    print_table(groups, args.backend, present_tasks)

    if args.out:
        save_csv(groups, args.out, present_tasks)


if __name__ == "__main__":
    main()
