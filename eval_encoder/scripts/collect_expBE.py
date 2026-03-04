#!/usr/bin/env python3
"""
collect_expBE.py  —  Extract E-1 / E-2 / E-3 sub-CSVs from a combined expBE run.

Usage:
    python eval_encoder/scripts/collect_expBE.py \
        --input  eval_encoder/eval_results/expBE.csv \
        --outdir eval_encoder/eval_results

Outputs:
    expE1_alignment.csv      E-1: max|Δlogit|, mean|Δlogit| per backend/task
    expE2_padding.csv        E-2: real vs synthetic latency / speedup / seq_pad_pct
    expE3_repeatability.csv  E-3: latency mean±std, throughput mean±std
"""

import argparse
import os
import sys

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in ["latency_ms", "latency_ms_std", "throughput_sps", "throughput_sps_std",
                "logit_max_diff", "logit_mean_abs_diff", "seq_pad_pct", "n_repeats"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[csv] {len(df):>3d} rows → {path}")


# ── E-1: Alignment ────────────────────────────────────────────────────────────

def extract_e1(df: pd.DataFrame, outdir: str) -> None:
    """
    Correctness gate: max|Δlogit| and mean|Δlogit| per (task, method, backend).

    - Uses synthetic input_mode (alignment check always runs on synthetic batch).
    - Excludes naive (diff = 0 by definition).
    - Threshold column: pass = max_diff < 0.05 (bf16 default).
    """
    THRESH = 0.05

    has_align = df["logit_max_diff"].notna() & (df["logit_max_diff"] != "")
    sub = df[has_align & (df["backend"] != "naive")].copy()

    if sub.empty:
        print("[E-1] No alignment rows found (run with ALIGN=1).")
        return

    # prefer synthetic rows; fall back to whatever is present
    synth = sub[sub["input_mode"] == "synthetic"]
    sub = synth if not synth.empty else sub

    out = sub[["task", "method", "backend", "dtype", "seq_len", "batch_size",
               "logit_max_diff", "logit_mean_abs_diff"]].copy()
    out = out.sort_values(["task", "method", "backend"]).reset_index(drop=True)
    out["pass_bf16"] = (out["logit_max_diff"] < THRESH).map({True: "✓", False: "✗"})

    _save(out, os.path.join(outdir, "expE1_alignment.csv"))

    # pretty-print
    print("\n── E-1 Alignment ──")
    print(out.to_string(index=False))
    print()


# ── E-2: Padding sensitivity ──────────────────────────────────────────────────

def extract_e2(df: pd.DataFrame, outdir: str) -> None:
    """
    Real-data vs synthetic speedup comparison.

    speedup_X = latency_naive_X / latency_backend_X   (X = real | synthetic)
    seq_pad_pct: from real rows (0 for synthetic by construction).
    """
    needed = {"real", "synthetic"}
    present = set(df["input_mode"].dropna().unique())
    if not needed.issubset(present):
        print(f"[E-2] Need both real+synthetic rows; found: {present}. Skipping.")
        return

    key = ["task", "method", "dtype", "seq_len", "batch_size"]

    def _pivot(mode: str) -> pd.DataFrame:
        sub = df[df["input_mode"] == mode][key + ["backend", "latency_ms", "seq_pad_pct"]].copy()
        sub = sub.rename(columns={"latency_ms": f"lat_{mode}", "seq_pad_pct": f"seq_pad_{mode}"})
        return sub

    real  = _pivot("real")
    synth = _pivot("synthetic")

    merged = real.merge(synth, on=key + ["backend"], how="outer")

    # speedup against naive baseline (within same task/dtype/seq_len/batch)
    naive_real  = real[real["backend"] == "naive"][key + ["lat_real"]].rename(
        columns={"lat_real": "naive_lat_real"})
    naive_synth = synth[synth["backend"] == "naive"][key + ["lat_synthetic"]].rename(
        columns={"lat_synthetic": "naive_lat_synth"})

    merged = merged.merge(naive_real,  on=key, how="left")
    merged = merged.merge(naive_synth, on=key, how="left")

    merged["speedup_real"]  = (merged["naive_lat_real"]  / merged["lat_real"]).round(3)
    merged["speedup_synth"] = (merged["naive_lat_synth"] / merged["lat_synthetic"]).round(3)
    merged["speedup_delta"] = (merged["speedup_synth"] - merged["speedup_real"]).round(3)

    out_cols = key + ["backend", "lat_real", "lat_synthetic",
                      "speedup_real", "speedup_synth", "speedup_delta", "seq_pad_real"]
    out = merged[out_cols].sort_values(key + ["backend"]).reset_index(drop=True)

    _save(out, os.path.join(outdir, "expE2_padding.csv"))

    print("\n── E-2 Padding Sensitivity ──")
    print(out[out["backend"] != "naive"].to_string(index=False))
    print()


# ── E-3: Repeatability ────────────────────────────────────────────────────────

def extract_e3(df: pd.DataFrame, outdir: str) -> None:
    """
    Mean±std across n_repeats independent runs.
    Keeps rows where latency_ms_std is present (n_repeats > 1).
    """
    sub = df[df["latency_ms_std"].notna() & (df["n_repeats"] > 1)].copy()

    if sub.empty:
        print("[E-3] No repeatability rows found (run with REPEAT>=2).")
        return

    out = sub[["task", "method", "backend", "input_mode", "dtype",
               "seq_len", "batch_size", "n_repeats",
               "latency_ms", "latency_ms_std",
               "throughput_sps", "throughput_sps_std"]].copy()
    out = out.sort_values(["task", "method", "backend", "input_mode"]).reset_index(drop=True)

    # coefficient of variation (CV) — useful for reviewers
    out["cv_latency_pct"] = (out["latency_ms_std"] / out["latency_ms"] * 100).round(2)

    _save(out, os.path.join(outdir, "expE3_repeatability.csv"))

    print("\n── E-3 Repeatability ──")
    print(out.to_string(index=False))
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  default="eval_encoder/eval_results/expBE.csv",
                   help="Path to the combined expBE CSV (output of expB.sh with ALIGN=1).")
    p.add_argument("--outdir", default="eval_encoder/eval_results",
                   help="Directory to write expE1/E2/E3 CSVs.")
    args = p.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[error] Input not found: {args.input}")

    df = _load(args.input)
    print(f"[load] {len(df)} rows from {args.input}")
    print(f"       tasks:    {sorted(df['task'].dropna().unique())}")
    print(f"       backends: {sorted(df['backend'].dropna().unique())}")
    print(f"       modes:    {sorted(df['input_mode'].dropna().unique())}")
    print()

    extract_e1(df, args.outdir)
    extract_e2(df, args.outdir)
    extract_e3(df, args.outdir)


if __name__ == "__main__":
    main()
