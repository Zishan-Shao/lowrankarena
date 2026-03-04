#!/usr/bin/env python3
"""
Comprehensive cleanup and consolidation of all encoder benchmark CSV files.
Cleans up both root eval_results/ and eval_encoder/eval_results/
"""

import pandas as pd
import os
import shutil
from pathlib import Path
from datetime import datetime

# Define paths
ROOT_EVAL = Path("/mnt/e/learning/SVD-Benchmark/lowrankarena/lowrankarena-main/lowrankarena-main/eval_results")
ENCODER_EVAL = Path("/mnt/e/learning/SVD-Benchmark/lowrankarena/lowrankarena-main/lowrankarena-main/eval_encoder/eval_results")

# Create directories
FINAL_DIR = ENCODER_EVAL / "final"
ARCHIVE_DIR = ENCODER_EVAL / "archived_csvs"
FINAL_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(exist_ok=True)

print("╔" + "="*70 + "╗")
print("║" + " "*15 + "CSV Complete Cleanup & Consolidation" + " "*20 + "║")
print("╚" + "="*70 + "╝\n")

# Step 1: Identify all CSV files
print("Step 1: Scanning for CSV files...")
print("-" * 70)

all_csvs = []

# Root eval_results
if ROOT_EVAL.exists():
    root_csvs = list(ROOT_EVAL.glob("*.csv"))
    print(f"Found {len(root_csvs)} files in {ROOT_EVAL}")
    all_csvs.extend([(f, "root") for f in root_csvs])

# eval_encoder/eval_results
encoder_csvs = list(ENCODER_EVAL.glob("*.csv"))
print(f"Found {len(encoder_csvs)} files in {ENCODER_EVAL}")
all_csvs.extend([(f, "encoder") for f in encoder_csvs])

print(f"\nTotal CSV files found: {len(all_csvs)}\n")

# Step 2: Categorize files
print("Step 2: Categorizing files...")
print("-" * 70)

backups = []
valid_results = []
duplicates = []

for csv_file, source in all_csvs:
    filename = csv_file.name

    # Identify backups
    if 'backup' in filename.lower():
        backups.append((csv_file, source))
        print(f"  📦 Backup: {filename}")
    # Identify likely duplicates
    elif any(x in filename for x in ['_rerun', '_clean_', '_no_reload', '_correct']):
        duplicates.append((csv_file, source))
        print(f"  🔄 Duplicate: {filename}")
    # Valid results
    else:
        valid_results.append((csv_file, source))
        print(f"  ✅ Valid: {filename}")

print(f"\nCategorized:")
print(f"  ✅ Valid results: {len(valid_results)}")
print(f"  🔄 Duplicates: {len(duplicates)}")
print(f"  📦 Backups: {len(backups)}\n")

# Step 3: Archive backups and duplicates
print("Step 3: Archiving old files...")
print("-" * 70)

for csv_file, source in backups + duplicates:
    dest = ARCHIVE_DIR / csv_file.name
    shutil.move(str(csv_file), str(dest))
    print(f"  📦 → archived/{csv_file.name}")

print(f"\n✓ Archived {len(backups) + len(duplicates)} files\n")

# Step 4: Read and consolidate valid results
print("Step 4: Consolidating valid results...")
print("-" * 70)

all_data = []
for csv_file, source in valid_results:
    try:
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            all_data.append(df)
            print(f"  ✓ Read {csv_file.name}: {len(df)} rows")
        else:
            print(f"  ⚠ Empty: {csv_file.name}")
            # Archive empty files too
            shutil.move(str(csv_file), str(ARCHIVE_DIR / csv_file.name))
    except Exception as e:
        print(f"  ✗ Error reading {csv_file.name}: {e}")

if not all_data:
    print("\n⚠️ No valid data found!")
    exit(1)

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n✓ Combined {len(all_data)} files → {len(combined_df)} total rows\n")

# Step 5: Remove duplicates
print("Step 5: Removing duplicate rows...")
print("-" * 70)

# Consider rows duplicates if they have same model, task, method, rank, budget, backend
dedupe_cols = ['model_id', 'task', 'method', 'rank', 'budget', 'backend', 'seq_len', 'batch_size']
existing_cols = [c for c in dedupe_cols if c in combined_df.columns]

before_count = len(combined_df)
combined_df = combined_df.drop_duplicates(subset=existing_cols, keep='last')
after_count = len(combined_df)

print(f"  Removed {before_count - after_count} duplicate rows")
print(f"  Remaining: {after_count} unique rows\n")

# Step 6: Split by method and save
print("Step 6: Saving organized results...")
print("-" * 70)

# Save complete dataset
complete_file = FINAL_DIR / "all_encoder_benchmarks.csv"
combined_df = combined_df.sort_values(
    ['method', 'seq_len', 'batch_size', 'backend', 'rank', 'budget'],
    na_position='last'
)
combined_df.to_csv(complete_file, index=False)
print(f"  ✓ Saved: final/all_encoder_benchmarks.csv ({len(combined_df)} rows)")

# Save by method
methods = combined_df['method'].unique()
for method in methods:
    method_df = combined_df[combined_df['method'] == method].copy()
    method_df = method_df.sort_values(['seq_len', 'batch_size', 'backend', 'rank', 'budget'], na_position='last')

    method_file = FINAL_DIR / f"{method}_benchmarks.csv"
    method_df.to_csv(method_file, index=False)
    print(f"  ✓ Saved: final/{method}_benchmarks.csv ({len(method_df)} rows)")

print("\n" + "="*70)
print("✅ Cleanup Complete!")
print("="*70 + "\n")

# Summary
print("📊 Final Structure:")
print(f"  eval_encoder/eval_results/final/")
print(f"    ├── all_encoder_benchmarks.csv ({len(combined_df)} rows)")
for method in methods:
    count = len(combined_df[combined_df['method'] == method])
    print(f"    ├── {method}_benchmarks.csv ({count} rows)")
print(f"\n  eval_encoder/eval_results/archived_csvs/")
print(f"    └── {len(backups) + len(duplicates)} old files archived")

# Move old files from root to archive too
if ROOT_EVAL.exists():
    root_remaining = list(ROOT_EVAL.glob("*.csv"))
    if root_remaining:
        print(f"\n⚠️  Note: {len(root_remaining)} CSV files still in root eval_results/")
        print(f"   Consider moving them to eval_encoder/eval_results/archived_csvs/")
