#!/usr/bin/env python3
"""
Analyze GLUE benchmark results with detailed task-level information.

Usage:
    python eval_encoder/scripts/analyze_results.py eval_encoder/glue_results/glue_results_*.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_results(file_path: str) -> Dict:
    """Load results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def print_summary(data: Dict):
    """Print summary with G-Avg and A-Avg."""
    print("\n" + "="*80)
    print(" "*25 + "GLUE Benchmark Summary")
    print("="*80)

    config = data.get("config", {})
    print(f"\nConfiguration:")
    print(f"  Method:        {config.get('method', 'N/A')}")
    print(f"  Rank:          {config.get('rank', 'N/A')}")
    print(f"  Retention:     {config.get('retention', 'N/A')}")
    print(f"  Backend:       {config.get('backend', 'N/A')}")
    print(f"  Model:         {config.get('model_id', 'N/A')}")
    print(f"  Tasks:         {len(data.get('results', []))}")

    summary = data.get("summary", {})
    if summary:
        print(f"\nFinal Scores:")
        g_avg = summary.get("G-Avg", {})
        a_avg = summary.get("A-Avg", {})
        print(f"  G-Avg:  {g_avg.get('initial', 0):.4f} → {g_avg.get('final', 0):.4f} ({g_avg.get('improvement', 0):+.4f})")
        print(f"  A-Avg:  {a_avg.get('initial', 0):.4f} → {a_avg.get('final', 0):.4f} ({a_avg.get('improvement', 0):+.4f})")


def print_task_details(data: Dict):
    """Print detailed task-level information."""
    results = data.get("results", [])

    print("\n" + "="*80)
    print(" "*28 + "Task-Level Details")
    print("="*80)

    for result in results:
        task = result.get("task", "unknown").upper()
        print(f"\n{task}")
        print("-"*80)

        # Dataset info
        dataset = result.get("dataset", {})
        if dataset:
            print(f"  Dataset:")
            print(f"    Train size:     {dataset.get('train_size', 'N/A'):,}")
            print(f"    Val size:       {dataset.get('val_size', 'N/A'):,}")
            print(f"    Num labels:     {dataset.get('num_labels', 'N/A')}")
            print(f"    Type:           {'Regression' if dataset.get('is_regression') else 'Classification'}")

        # Metrics
        metrics = result.get("metrics", {})
        if metrics:
            print(f"  Metrics:")
            print(f"    Primary metric: {metrics.get('primary_metric', 'N/A')}")
            print(f"    Initial:        {metrics.get('initial', {})}")
            print(f"    Final:          {metrics.get('final', {})}")
            print(f"    Best value:     {metrics.get('best_value', 0):.4f}")
            print(f"    Improvement:    {metrics.get('improvement', 0):+.4f}")

        # Training info
        training = result.get("training", {})
        if training:
            print(f"  Training:")
            print(f"    Epochs:         {training.get('num_epochs', 'N/A')}")
            print(f"    Batch size:     {training.get('batch_size', 'N/A')}")
            print(f"    Learning rate:  {training.get('learning_rate', 'N/A')}")
            print(f"    Total steps:    {training.get('total_steps', 'N/A'):,}")
            print(f"    Time:           {training.get('time_minutes', 0):.1f} minutes")


def print_comparison_table(data: Dict):
    """Print comparison table of all tasks."""
    results = data.get("results", [])

    print("\n" + "="*100)
    print(" "*38 + "Task Comparison Table")
    print("="*100)

    header = f"{'Task':<8} | {'Metric':<22} | {'Initial':>8} | {'Final':>8} | {'Δ':>8} | {'Train':>6} | {'Val':>6} | {'Time':>8}"
    print(header)
    print("-"*100)

    total_time = 0
    for result in results:
        task = result.get("task", "unknown").upper()
        metric = result.get("best_metric", "N/A")
        initial = result.get("initial_results", {}).get(metric, 0)
        final = result.get("best_value", 0)
        improvement = final - initial

        dataset = result.get("dataset", {})
        train_size = dataset.get("train_size", 0)
        val_size = dataset.get("val_size", 0)

        training = result.get("training", {})
        time_min = training.get("time_minutes", 0)
        total_time += time_min

        print(f"{task:<8} | {metric:<22} | {initial:8.4f} | {final:8.4f} | {improvement:+8.4f} | "
              f"{train_size:6,} | {val_size:6,} | {time_min:7.1f}m")

    print("-"*100)
    print(f"{'TOTAL':<8} | {'':<22} | {'':<8} | {'':<8} | {'':<8} | {'':<6} | {'':<6} | {total_time:7.1f}m")
    print("="*100)


def export_csv(data: Dict, output_file: str):
    """Export results to CSV format."""
    import csv

    results = data.get("results", [])

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Task', 'Metric', 'Initial', 'Final', 'Improvement',
            'Train_Size', 'Val_Size', 'Num_Labels', 'Type',
            'Epochs', 'Batch_Size', 'LR', 'Time_Minutes'
        ])

        # Data
        for result in results:
            task = result.get("task", "unknown")
            metric = result.get("best_metric", "N/A")
            initial = result.get("initial_results", {}).get(metric, 0)
            final = result.get("best_value", 0)
            improvement = final - initial

            dataset = result.get("dataset", {})
            training = result.get("training", {})

            writer.writerow([
                task,
                metric,
                f"{initial:.4f}",
                f"{final:.4f}",
                f"{improvement:+.4f}",
                dataset.get("train_size", 0),
                dataset.get("val_size", 0),
                dataset.get("num_labels", 0),
                "Regression" if dataset.get("is_regression") else "Classification",
                training.get("num_epochs", 0),
                training.get("batch_size", 0),
                training.get("learning_rate", 0),
                f"{training.get('time_minutes', 0):.1f}",
            ])

    print(f"\n✅ Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GLUE benchmark results")
    parser.add_argument("result_file", help="Path to results JSON file")
    parser.add_argument("--export-csv", help="Export to CSV file")
    parser.add_argument("--summary-only", action="store_true", help="Show summary only")
    parser.add_argument("--details-only", action="store_true", help="Show details only")

    args = parser.parse_args()

    # Load results
    try:
        data = load_results(args.result_file)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {args.result_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON file: {args.result_file}")
        sys.exit(1)

    # Print analysis
    if not args.details_only:
        print_summary(data)
        print_comparison_table(data)

    if not args.summary_only:
        print_task_details(data)

    # Export CSV if requested
    if args.export_csv:
        export_csv(data, args.export_csv)

    print("\n")


if __name__ == "__main__":
    main()
