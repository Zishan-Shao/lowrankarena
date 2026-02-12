#!/usr/bin/env python3
"""
Organize and clean CSV files in eval_results/
- Merge adasvd_flashsvd_complete.csv + adasvd_naive_complete.csv
- Archive old/broken data
- Create clean final benchmarks
"""

import os
import pandas as pd
from datetime import datetime

# Paths
EVAL_DIR = "eval_results"
FINAL_DIR = f"{EVAL_DIR}/final"
ARCHIVE_DIR = f"{EVAL_DIR}/archived_csvs"

def main():
    print("=" * 60)
    print("CSV Organization & Cleanup")
    print("=" * 60)
    print()

    # Create directories
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # 1. Merge AdaSVD FlashSVD + Naive results
    print("Step 1: Merging AdaSVD results...")
    flashsvd_csv = f"{FINAL_DIR}/adasvd_flashsvd_complete.csv"
    naive_csv = f"{FINAL_DIR}/adasvd_naive_complete.csv"

    dfs = []
    if os.path.exists(flashsvd_csv):
        df_flash = pd.read_csv(flashsvd_csv)
        print(f"  ✅ Loaded {len(df_flash)} FlashSVD results")
        dfs.append(df_flash)
    else:
        print(f"  ⚠️  FlashSVD CSV not found: {flashsvd_csv}")

    if os.path.exists(naive_csv):
        df_naive = pd.read_csv(naive_csv)
        print(f"  ✅ Loaded {len(df_naive)} Naive results")
        dfs.append(df_naive)
    else:
        print(f"  ⚠️  Naive CSV not found: {naive_csv}")

    if dfs:
        df_merged = pd.concat(dfs, ignore_index=True)
        # Sort by backend (flashsvd first) then budget
        df_merged = df_merged.sort_values(['backend', 'budget'])

        output_path = f"{FINAL_DIR}/adasvd_complete_benchmarks.csv"
        df_merged.to_csv(output_path, index=False)
        print(f"  ✅ Merged {len(df_merged)} results → {output_path}")

        # Print summary
        print("\n  Summary:")
        summary = df_merged.groupby(['backend', 'budget']).size()
        for (backend, budget), count in summary.items():
            print(f"    {backend:10s} budget={budget:.1f}: {count} runs")
    print()

    # 2. Archive old encoder_runs.csv
    print("Step 2: Archiving old encoder_runs.csv...")
    main_csv = f"{EVAL_DIR}/encoder_runs.csv"
    if os.path.exists(main_csv):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f"{ARCHIVE_DIR}/encoder_runs_pre_cleanup_{timestamp}.csv"

        df = pd.read_csv(main_csv)
        df.to_csv(archive_path, index=False)
        print(f"  ✅ Archived {len(df)} rows → {archive_path}")

        # Keep only recent valid data (after budget fix)
        # Filter: method=adasvd AND param_ratio within ±10% of budget
        df_clean = df[df['method'] == 'adasvd'].copy()
        if 'budget' in df_clean.columns and 'param_ratio' in df_clean.columns:
            # Calculate deviation
            df_clean['deviation'] = abs(df_clean['param_ratio'] - df_clean['budget']) / df_clean['budget']
            # Keep only data with <10% deviation (valid budget control)
            df_valid = df_clean[df_clean['deviation'] < 0.1].copy()
            df_valid = df_valid.drop('deviation', axis=1)

            print(f"  ✅ Filtered {len(df_clean)} → {len(df_valid)} valid AdaSVD results")
            print(f"     (Removed {len(df_clean) - len(df_valid)} broken budget control data)")

            df_valid.to_csv(main_csv, index=False)
            print(f"  ✅ Cleaned encoder_runs.csv ({len(df_valid)} rows)")
    else:
        print(f"  ⚠️  encoder_runs.csv not found")
    print()

    # 3. Summary
    print("Step 3: Final file structure...")
    print(f"\n  Final CSVs in {FINAL_DIR}:")
    for fname in sorted(os.listdir(FINAL_DIR)):
        if fname.endswith('.csv'):
            fpath = os.path.join(FINAL_DIR, fname)
            if os.path.isfile(fpath):
                df = pd.read_csv(fpath)
                print(f"    ✅ {fname:40s} ({len(df):3d} rows)")
    print()

    print("=" * 60)
    print("✅ CSV Organization Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
