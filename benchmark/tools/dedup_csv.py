#!/usr/bin/env python3
"""
Deduplicate a benchmark CSV by key columns, keeping the last (= most recent) row
per unique combination.

Designed for CSVs that are written with analyze_compute.py's append strategy:
  rows are appended in run order, so "last occurrence" = most recent run.

Default key columns (cover expB.csv / expC_seqlen.csv / expC_batch.csv):
  task, method, backend, dtype, seq_len, batch_size

Usage
-----
  # Deduplicate expB.csv in-place:
  python eval_encoder/scripts/utils/dedup_csv.py \
      --csv eval_encoder/eval_results/expB.csv

  # Dry-run (print stats only, don't overwrite):
  python eval_encoder/scripts/utils/dedup_csv.py \
      --csv eval_encoder/eval_results/expB.csv --dry_run

  # Custom key columns:
  python eval_encoder/scripts/utils/dedup_csv.py \
      --csv some.csv --keys task method seq_len
"""

import argparse
import os
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("[error] pandas is required:  pip install pandas")


DEFAULT_KEYS = ["task", "method", "backend", "dtype", "seq_len", "batch_size"]


def dedup(path: str, keys: list[str], dry_run: bool) -> None:
    if not os.path.exists(path):
        sys.exit(f"[error] File not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    n_before = len(df)

    # Validate keys
    missing = [k for k in keys if k not in df.columns]
    if missing:
        sys.exit(f"[error] Key columns not in CSV: {missing}\n"
                 f"        Available columns: {list(df.columns)}")

    # keep='last' → keep most recent run per key combination
    df_dedup = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    n_after  = len(df_dedup)
    n_removed = n_before - n_after

    print(f"[dedup] {path}")
    print(f"  rows before : {n_before}")
    print(f"  rows after  : {n_after}")
    print(f"  removed     : {n_removed}")

    if n_removed == 0:
        print("  (already clean, nothing to do)")
        return

    if dry_run:
        print("  [dry_run] not writing (pass --dry_run=false to overwrite)")
        return

    df_dedup.to_csv(path, index=False)
    print(f"  [written]  → {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     required=True, help="CSV file to deduplicate (in-place)")
    p.add_argument("--keys",    nargs="+",     default=DEFAULT_KEYS,
                   help=f"Key columns (default: {DEFAULT_KEYS})")
    p.add_argument("--dry_run", action="store_true",
                   help="Print stats only, do not write")
    args = p.parse_args()
    dedup(args.csv, args.keys, args.dry_run)


if __name__ == "__main__":
    main()
