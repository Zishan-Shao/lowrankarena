#!/usr/bin/env python3
"""
Collect stage1 (no fine-tune) accuracy from expA.csv into a summary table.

Reads expA.csv (written by run_encoder_benchmark.py), filters backend=naive,
takes the LATEST row per (method, qkv_mode, task), and prints a paper-style
accuracy table covering GLUE + SuperGLUE/HANS/ANLI.

Task tiers:
  GLUE tasks         : cola sst2 mrpc qqp mnli qnli rte stsb  → G-AVG
  SuperGLUE-Core     : boolq rte_sg wic copa                  → SG-Core-AVG
  Diagnostic         : cb  (high-variance, 56 examples, NOT in any average)
  Robustness         : hans anli_r1 anli_r2 anli_r3           → Rob-AVG

Run from repo root:
    python experiments/collect_expA_results.py
    python experiments/collect_expA_results.py --csv /path/to/expA.csv
    python experiments/collect_expA_results.py --out summary.csv
"""
import os
import csv
import argparse
from collections import defaultdict

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_CSV = os.path.join(REPO_ROOT, "experiments", "expA.csv")

GLUE_TASKS      = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"]
SG_CORE_TASKS   = ["boolq", "rte_sg", "wic", "copa"]   # SuperGLUE-Core (enter average)
DIAGNOSTIC_TASKS = ["cb"]                               # High-variance, NOT in any average
ROBUST_TASKS    = ["hans", "anli_r1", "anli_r2", "anli_r3"]
ALL_TASKS       = GLUE_TASKS + SG_CORE_TASKS + DIAGNOSTIC_TASKS + ROBUST_TASKS

NORMALIZE_TASKS = {"cola", "stsb"}   # MCC / Pearson → (v+1)/2 before G-AVG

TASK_HDR = {
    "cola":    "CoLA",  "sst2":    "SST-2",  "mrpc":    "MRPC",
    "qqp":     "QQP",   "mnli":    "MNLI",   "qnli":    "QNLI",
    "rte":     "RTE",   "stsb":    "STS-B",
    "boolq":   "BoolQ", "rte_sg":  "RTE-SG", "wic":     "WiC",
    "copa":    "COPA",
    "cb":      "CB†",   # † = high-variance diagnostic task (56 examples)
    "hans":    "HANS",  "anli_r1": "ANLI-R1","anli_r2": "ANLI-R2","anli_r3": "ANLI-R3",
}

METHOD_ORDER = ["dense", "svd", "fwsvd", "drone", "adasvd"]


def compute_glue_avg(scores):
    """Normalized G-AVG over GLUE tasks present in scores."""
    vals = []
    for t in GLUE_TASKS:
        v = scores.get(t)
        if v is not None:
            vals.append((v + 1) / 2 if t in NORMALIZE_TASKS else v)
    return sum(vals) / len(vals) if vals else None


def compute_sg_core_avg(scores):
    """Simple average over SuperGLUE-Core tasks present in scores."""
    vals = [scores[t] for t in SG_CORE_TASKS if t in scores]
    return sum(vals) / len(vals) if vals else None


def compute_rob_avg(scores):
    """Simple average over Robustness tasks present in scores."""
    vals = [scores[t] for t in ROBUST_TASKS if t in scores]
    return sum(vals) / len(vals) if vals else None


def load_csv(path, backend_filter, qkv_filter):
    """Return dict: (method, qkv_mode) → task → latest metric_value."""
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


def _fmt(v):
    return f"{v:8.4f}" if v is not None else f"{'---':>8}"


def _print_section(title, col_tasks, groups, qkv, avg_fn, avg_label, present_tasks):
    tasks = [t for t in col_tasks if t in present_tasks]
    if not tasks:
        return
    hdr = f"  {'Method':<9}" + "".join(f"  {TASK_HDR.get(t,t):>8}" for t in tasks)
    hdr += f"  {avg_label:>9}"
    print(f"\n  ── {title}")
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for method in METHOD_ORDER:
        key = (method, qkv)
        if key not in groups:
            continue
        scores = groups[key]
        row = f"  {method:<9}" + "".join(
            f"  {_fmt(scores.get(t))}" for t in tasks
        )
        avg = avg_fn(scores)
        row += f"  {_fmt(avg)}"
        print(row)


def print_table(groups, backend, present_tasks):
    """Print structured three-tier table per qkv_mode found in groups."""
    qkv_modes = sorted(set(qkv for _, qkv in groups))

    for qkv in qkv_modes:
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  qkv_mode={qkv}   backend={backend}   (stage1 / no fine-tune)")
        print(sep)

        # Section 1: GLUE
        _print_section(
            "GLUE  (G-AVG: normalized MCC/Pearson)",
            GLUE_TASKS, groups, qkv,
            compute_glue_avg, "G-AVG", present_tasks,
        )

        # Section 2: SuperGLUE-Core
        _print_section(
            "SuperGLUE-Core  (SG-Core-AVG)",
            SG_CORE_TASKS, groups, qkv,
            compute_sg_core_avg, "SG-Core", present_tasks,
        )

        # Section 3: Diagnostic (CB) — not in any average
        diag_tasks = [t for t in DIAGNOSTIC_TASKS if t in present_tasks]
        if diag_tasks:
            hdr = f"  {'Method':<9}" + "".join(
                f"  {TASK_HDR.get(t,t):>8}" for t in diag_tasks
            ) + f"  {'(no avg)':>9}"
            print(f"\n  ── Diagnostic  (NOT in average; high-variance)")
            print("  " + "-" * (len(hdr) - 2))
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for method in METHOD_ORDER:
                key = (method, qkv)
                if key not in groups:
                    continue
                scores = groups[key]
                row = f"  {method:<9}" + "".join(
                    f"  {_fmt(scores.get(t))}" for t in diag_tasks
                ) + f"  {'---':>9}"
                print(row)
            print(f"  † CB: dev=56 examples, high-variance — treat as diagnostic only")

        # Section 4: Robustness
        _print_section(
            "Robustness  (Rob-AVG)",
            ROBUST_TASKS, groups, qkv,
            compute_rob_avg, "Rob-AVG", present_tasks,
        )

    print()


def save_csv(groups, out_path, present_tasks):
    col_tasks = [t for t in ALL_TASKS if t in present_tasks]
    fieldnames = ["method", "qkv_mode"] + col_tasks + [
        "G-AVG_glue", "SG-Core-AVG", "Rob-AVG"
    ]
    rows = []
    for qkv in sorted(set(q for _, q in groups)):
        for method in METHOD_ORDER:
            key = (method, qkv)
            if key not in groups:
                continue
            scores = groups[key]
            rows.append({
                "method":      method,
                "qkv_mode":    qkv,
                **{t: round(scores[t], 6) if t in scores else "" for t in col_tasks},
                "G-AVG_glue":  round(v, 6) if (v := compute_glue_avg(scores)) else "",
                "SG-Core-AVG": round(v, 6) if (v := compute_sg_core_avg(scores)) else "",
                "Rob-AVG":     round(v, 6) if (v := compute_rob_avg(scores)) else "",
            })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}  ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",      default=DEFAULT_CSV,
                   help=f"Path to expA.csv  (default: {DEFAULT_CSV})")
    p.add_argument("--backend",  default="naive",
                   help="Backend to filter for accuracy (default: naive)")
    p.add_argument("--qkv_mode", default="",
                   help="Filter by qkv_mode (default: show all)")
    p.add_argument("--out",      default="",
                   help="Output CSV path (default: print only)")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[error] File not found: {args.csv}")
        return

    groups = load_csv(args.csv, args.backend, args.qkv_mode)
    if not groups:
        print(f"[warn] No data found in {args.csv} for backend={args.backend}")
        return

    present_tasks = set()
    for scores in groups.values():
        present_tasks |= set(scores.keys())

    print_table(groups, args.backend, present_tasks)

    if args.out:
        save_csv(groups, args.out, present_tasks)


if __name__ == "__main__":
    main()
