#!/usr/bin/env python3
"""
Collect GLUE benchmark results from JSON files into a summary CSV.

Scans eval_encoder/glue_results/*.json, groups by (method, qkv_mode, seq_len),
merges task scores (newer runs overwrite older), and writes:
  eval_encoder/eval_results/glue_summary.csv

Run from repo root:
    python eval_encoder/eval_results/collect_glue_results.py
"""
import os
import json
import glob
import csv
from collections import defaultdict

GLUE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "glue_results")
OUT_CSV          = os.path.join(os.path.dirname(__file__), "glue_summary.csv")

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

    return dict(method=method, qkv_mode=qkv_mode, seq_len=seq_len, timestamp=timestamp,
                skip_finetuning=skip_finetuning,
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
            groups[(info["method"], info["qkv_mode"], info["seq_len"])].append(info)
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}")
            skipped += 1

    rows = []
    for (method, qkv_mode, seq_len), runs in sorted(groups.items()):
        # merge task scores: newer timestamps overwrite older ones
        # s1 (compress-eval): always valid — take newest
        # s2 (compress+finetune): only valid when skip_finetuning=False
        runs_sorted = sorted(runs, key=lambda x: x["timestamp"])
        merged_s1 = {}   # task → s1 value
        merged_s2 = {}   # task → s2 value  (only from finetune runs)
        g_s1 = g_s2 = None
        for run in runs_sorted:
            for task, (s1, s2) in run["task_scores"].items():
                if s1 is not None:
                    merged_s1[task] = s1
                if s2 is not None:   # already None for skip_finetuning runs
                    merged_s2[task] = s2
            # Only trust G-Avg from runs that completed all 8 tasks (avoids partial-run G-Avg contamination)
            run_n_s1 = sum(1 for t in TASKS_8 if run["task_scores"].get(t, (None,None))[0] is not None)
            run_n_s2 = sum(1 for t in TASKS_8 if run["task_scores"].get(t, (None,None))[1] is not None)
            if run["g_avg_s1"] is not None and run_n_s1 == len(TASKS_8):
                g_s1 = run["g_avg_s1"]
            if run["g_avg_s2"] is not None and run_n_s2 == len(TASKS_8):  # already None for skip_finetuning runs
                g_s2 = run["g_avg_s2"]
        merged = {t: (merged_s1.get(t), merged_s2.get(t)) for t in set(merged_s1) | set(merged_s2)}

        # Count how many tasks are present in the merged result
        n_s1 = sum(1 for t in TASKS_8 if merged_s1.get(t) is not None)
        n_s2 = sum(1 for t in TASKS_8 if merged_s2.get(t) is not None)

        # G-AVG: prefer the JSON value from a complete 8-task run (already
        # normalized by glue_pipeline.py); fall back to compute_gavg() which
        # applies the same normalization over however many tasks are present.
        if g_s1 is None and n_s1 > 0:
            g_s1 = compute_gavg(merged, stage=0)
        if g_s2 is None and n_s2 > 0:
            g_s2 = compute_gavg(merged, stage=1)

        row = {
            "method":      method,
            "qkv_mode":    qkv_mode,
            "seq_len":     seq_len,
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

    fieldnames = ["method", "qkv_mode", "seq_len",
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
