#!/usr/bin/env python3
"""
Collect GLUE benchmark results from JSON files into a summary CSV.

Scans experiments/glue/*.json, groups by (method, qkv_mode, seq_len),
picks the LATEST run per stage (s1/s2 independently), and writes:
  experiments/results/glue_summary.csv

Strategy: for each (method, qkv_mode, seq_len) group:
  - s1: use the single latest skip_finetuning=True  JSON
  - s2: use the single latest skip_finetuning=False JSON
This avoids stale partial runs contaminating the merged result.

Run from repo root:
    python experiments/collect_glue_results.py
"""
import os
import json
import glob
import csv
from collections import defaultdict

GLUE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "glue")
OUT_CSV          = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results", "glue_summary.csv")

TASKS_8  = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"]
METRIC   = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "f1",
    "qqp":  "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte":  "accuracy",
    "stsb": "pearson",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _primary_metric(task, metrics_dict):
    """Return primary metric value from a metrics sub-dict."""
    primary = METRIC.get(task)
    if primary and primary in metrics_dict:
        return metrics_dict[primary]
    # fallback: first numeric value
    for v in metrics_dict.values():
        if isinstance(v, (int, float)):
            return v
    return None


def parse_json(path):
    d = json.load(open(path, encoding="utf-8"))
    cfg            = d.get("config", {})
    method         = cfg.get("method", "unknown")
    qkv_mode       = cfg.get("qkv_mode", "per_head")
    timestamp      = d.get("timestamp", "00000000_000000")
    skip_finetuning = cfg.get("skip_finetuning", True)   # conservative default

    results = d.get("results", [])
    if not results:
        raise ValueError("no task results (aborted/empty run)")

    task_scores = {}
    for r in results:
        task    = r.get("task", "")
        metrics = r.get("metrics", {})
        s1 = _primary_metric(task, metrics.get("initial", {}))
        # s2 is only meaningful when finetuning was actually performed
        s2 = None if skip_finetuning else _primary_metric(task, metrics.get("final", {}))
        task_scores[task] = (s1, s2)

    summary  = d.get("summary", {})
    g_avg_s1 = summary.get("G-Avg", {}).get("initial")
    # g_avg_s2 is only meaningful when finetuning was actually performed
    g_avg_s2 = None if skip_finetuning else summary.get("G-Avg", {}).get("final")

    seq_len = cfg.get("seq_len", 128)

    dtype = cfg.get("dtype", "fp32")   # old JSONs pre-dating --dtype arg default to fp32

    return dict(method=method, qkv_mode=qkv_mode, seq_len=seq_len, dtype=dtype,
                timestamp=timestamp, skip_finetuning=skip_finetuning,
                task_scores=task_scores, g_avg_s1=g_avg_s1, g_avg_s2=g_avg_s2)


# Tasks whose raw metric is in [-1, 1] and must be mapped to [0, 1] before averaging
_NORMALIZE_TASKS = {"cola", "stsb"}


def compute_gavg(scores_dict, stage):
    """Normalized mean over available GLUE tasks.

    MCC (CoLA) and Pearson (STS-B) are mapped from [-1,1] to [0,1] via
    (score+1)/2 before averaging, matching glue_pipeline.py's formula.
    None values (tasks not yet run) are skipped; the denominator is the
    number of tasks actually present, so partial runs yield a valid partial
    G-AVG rather than being discarded.
    """
    vals = []
    for t in TASKS_8:
        s1, s2 = scores_dict.get(t, (None, None))
        v = s1 if stage == 0 else s2
        if v is not None:
            if t in _NORMALIZE_TASKS:
                v = (v + 1) / 2
            vals.append(v)
    return round(sum(vals) / len(vals), 6) if vals else None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    files = sorted(glob.glob(os.path.join(GLUE_RESULTS_DIR, "glue_results_*.json")))
    if not files:
        print(f"No JSON files found in {GLUE_RESULTS_DIR}")
        return

    groups = defaultdict(list)
    for f in files:
        try:
            groups[(*(parse_json(f)["method"],), parse_json(f)["qkv_mode"])]  # force parse twice — refactor below
        except Exception:
            pass

    # Re-parse properly
    groups = defaultdict(list)
    skipped = 0
    for f in files:
        try:
            info = parse_json(f)
            groups[(info["method"], info["qkv_mode"], info["seq_len"], info["dtype"])].append(info)
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}")
            skipped += 1

    rows = []
    for (method, qkv_mode, seq_len, dtype), runs in sorted(groups.items()):
        # Pick the single latest run per stage independently:
        #   s1 source: latest skip_finetuning=True  run
        #   s2 source: latest skip_finetuning=False run
        # This prevents stale partial runs from contaminating the result.
        # s1: prefer the latest COMPLETE (all 8 tasks) run; fall back to latest partial
        # Both skip_ft=True and skip_ft=False files store initial (pre-finetune) scores.
        def _n_s1(r):
            return sum(1 for t in TASKS_8 if r["task_scores"].get(t, (None, None))[0] is not None)
        complete_s1_runs = sorted([r for r in runs if _n_s1(r) == len(TASKS_8)], key=lambda x: x["timestamp"])
        partial_s1_runs  = sorted([r for r in runs if _n_s1(r) > 0],             key=lambda x: x["timestamp"])
        latest_s1_run = complete_s1_runs[-1] if complete_s1_runs else (partial_s1_runs[-1] if partial_s1_runs else None)

        # s2: use latest COMPLETE (all 8 tasks) skip_ft=False run; ignore partial runs
        s2_runs = sorted([r for r in runs if not r["skip_finetuning"]], key=lambda x: x["timestamp"])
        complete_s2_runs = [r for r in s2_runs
                            if sum(1 for t in TASKS_8 if r["task_scores"].get(t, (None, None))[1] is not None) == len(TASKS_8)]
        latest_s2_run = complete_s2_runs[-1] if complete_s2_runs else None

        merged_s1 = {t: s1 for t, (s1, _) in (latest_s1_run["task_scores"].items() if latest_s1_run else {}.items()) if s1 is not None}
        merged_s2 = {t: s2 for t, (_, s2) in (latest_s2_run["task_scores"].items() if latest_s2_run else {}.items()) if s2 is not None}

        # G-Avg: prefer value from JSON (already normalized); fall back to compute
        g_s1 = latest_s1_run["g_avg_s1"] if latest_s1_run else None
        g_s2 = latest_s2_run["g_avg_s2"] if latest_s2_run else None

        merged = {t: (merged_s1.get(t), merged_s2.get(t)) for t in set(merged_s1) | set(merged_s2)}

        # Count how many tasks are present
        n_s1 = sum(1 for t in TASKS_8 if merged_s1.get(t) is not None)
        n_s2 = sum(1 for t in TASKS_8 if merged_s2.get(t) is not None)

        # Fall back to compute_gavg() if JSON didn't store it
        if g_s1 is None and n_s1 > 0:
            g_s1 = compute_gavg(merged, stage=0)
        if g_s2 is None and n_s2 > 0:
            g_s2 = compute_gavg(merged, stage=1)

        row = {
            "method":      method,
            "qkv_mode":    qkv_mode,
            "seq_len":     seq_len,
            "dtype":       dtype,
            "g_avg_s1":    round(g_s1, 6) if g_s1 is not None else "",
            "g_avg_s2":    round(g_s2, 6) if g_s2 is not None else "",
            "n_tasks_s1":  n_s1,
            "n_tasks_s2":  n_s2,
        }
        for task in TASKS_8:
            s1, s2 = merged.get(task, (None, None))
            row[f"{task}_s1"] = round(s1, 6) if s1 is not None else ""
            row[f"{task}_s2"] = round(s2, 6) if s2 is not None else ""
        rows.append(row)

    fieldnames = ["method", "qkv_mode", "seq_len", "dtype",
                  "g_avg_s1", "g_avg_s2", "n_tasks_s1", "n_tasks_s2"] + \
                 [f"{t}_{s}" for t in TASKS_8 for s in ("s1", "s2")]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUT_CSV}  ({len(rows)} rows, {skipped} skipped)")
    for r in rows:
        n1, n2 = r['n_tasks_s1'], r['n_tasks_s2']
        s1_tag = f"{r['g_avg_s1']} ({n1}/8)" if n1 < 8 else str(r['g_avg_s1'])
        s2_tag = f"{r['g_avg_s2']} ({n2}/8)" if 0 < n2 < 8 else str(r['g_avg_s2'])
        print(f"  {r['method']:8s} {r['qkv_mode']:10s} seq={r['seq_len']:3}  "
              f"G-AVG s1={s1_tag}  s2={s2_tag}")


if __name__ == "__main__":
    main()
