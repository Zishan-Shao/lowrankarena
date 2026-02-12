#!/usr/bin/env python3
"""
Generate README.md table from fp32 benchmark results
"""

import pandas as pd
import sys

def format_memory_delta(naive_mem, flashsvd_mem):
    """Calculate memory delta with percentage"""
    delta = flashsvd_mem - naive_mem
    pct = (delta / naive_mem) * 100
    return f"{delta:+.0f} ({pct:+.1f}%)"

def main():
    # Read CSV
    csv_path = '/mnt/e/learning/SVD-Benchmark/lowrankarena/eval_encoder/eval_results/encoder_runs.csv'
    df = pd.read_csv(csv_path)

    # Filter for fp32, seq=128, batch=32, today's tests
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    fp32_df = df[
        (df['dtype'] == 'fp32') &
        (df['seq_len'] == 128) &
        (df['batch_size'] == 32) &
        (df['timestamp'].dt.date == pd.Timestamp.now().date())
    ].copy()

    print(f"Found {len(fp32_df)} fp32 test results")

    # Use peak_mem_infer_mb for inference memory (推理阶段内存)
    fp32_df['infer_mem'] = fp32_df['peak_mem_infer_mb']

    # Build table rows
    rows = []

    # Dense baseline
    dense = fp32_df[fp32_df['method'] == 'dense']
    if len(dense) > 0:
        d = dense.iloc[0]
        rows.append({
            'method': 'dense',
            'rank': '-',
            'backend': '-',
            'accuracy': f"{d['metric_value']:.2%}",
            'latency': f"{d['latency_ms']:.1f}",
            'throughput': f"{d['throughput_sps']:.1f}",
            'infer_mem': f"{d['infer_mem']:.0f}",
            'delta_mem': '-',
            'param_ratio': f"{d['param_ratio']:.2%}"
        })

    # SVD, FWSVD, DRONE
    for method in ['svd', 'fwsvd', 'drone']:
        method_df = fp32_df[fp32_df['method'] == method]

        # Determine rank list based on method
        if method == 'drone':
            ranks = [32, 64, 128, 256, 512]
        else:
            ranks = [32, 64, 128, 256, 512]

        for rank in ranks:
            rank_df = method_df[method_df['rank'] == rank]

            # Naive
            naive_rows = rank_df[rank_df['backend'] == 'naive']
            if len(naive_rows) > 0:
                n = naive_rows.iloc[0]
                rows.append({
                    'method': method,
                    'rank': str(rank),
                    'backend': 'naive',
                    'accuracy': f"{n['metric_value']:.2%}",
                    'latency': f"{n['latency_ms']:.1f}",
                    'throughput': f"{n['throughput_sps']:.1f}",
                    'infer_mem': f"{n['infer_mem']:.0f}",
                    'delta_mem': '-',
                    'param_ratio': f"{n['param_ratio']:.2%}"
                })

            # FlashSVD
            flashsvd_rows = rank_df[rank_df['backend'] == 'flashsvd']
            if len(flashsvd_rows) > 0 and len(naive_rows) > 0:
                f = flashsvd_rows.iloc[0]
                n = naive_rows.iloc[0]
                delta_mem = format_memory_delta(n['infer_mem'], f['infer_mem'])
                rows.append({
                    'method': method,
                    'rank': str(rank),
                    'backend': 'flashsvd',
                    'accuracy': f"{f['metric_value']:.2%}",
                    'latency': f"{f['latency_ms']:.1f}",
                    'throughput': f"{f['throughput_sps']:.1f}",
                    'infer_mem': f"{f['infer_mem']:.0f}",
                    'delta_mem': delta_mem,
                    'param_ratio': f"{f['param_ratio']:.2%}"
                })

    # AdaSVD
    adasvd_df = fp32_df[fp32_df['method'] == 'adasvd']
    for budget in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        budget_df = adasvd_df[abs(adasvd_df['budget'] - budget) < 0.01]

        # Naive
        naive_rows = budget_df[budget_df['backend'] == 'naive']
        if len(naive_rows) > 0:
            n = naive_rows.iloc[0]
            rows.append({
                'method': 'adasvd',
                'rank': f"{budget:.1f}",
                'backend': 'naive',
                'accuracy': f"{n['metric_value']:.2%}",
                'latency': f"{n['latency_ms']:.1f}",
                'throughput': f"{n['throughput_sps']:.1f}",
                'infer_mem': f"{n['infer_mem']:.0f}",
                'delta_mem': '-',
                'param_ratio': f"{n['param_ratio']:.2%}"
            })

        # FlashSVD
        flashsvd_rows = budget_df[budget_df['backend'] == 'flashsvd']
        if len(flashsvd_rows) > 0 and len(naive_rows) > 0:
            f = flashsvd_rows.iloc[0]
            n = naive_rows.iloc[0]
            delta_mem = format_memory_delta(n['infer_mem'], f['infer_mem'])
            rows.append({
                'method': 'adasvd',
                'rank': f"{budget:.1f}",
                'backend': 'flashsvd',
                'accuracy': f"{f['metric_value']:.2%}",
                'latency': f"{f['latency_ms']:.1f}",
                'throughput': f"{f['throughput_sps']:.1f}",
                'infer_mem': f"{f['infer_mem']:.0f}",
                'delta_mem': delta_mem,
                'param_ratio': f"{f['param_ratio']:.2%}"
            })

    # Generate markdown table
    print("\n" + "="*80)
    print("README Table (fp32 data)")
    print("="*80)
    print()
    print("| Method | Rank/Budget | Backend | Accuracy | Latency (ms) | Throughput (sps) | Infer Mem (MB) | Δ Mem (vs Naive) | Param Ratio |")
    print("|--------|-------------|---------|----------|--------------|------------------|----------------|------------------|-------------|")

    for row in rows:
        print(f"| {row['method']} | {row['rank']} | {row['backend']} | {row['accuracy']} | {row['latency']} | {row['throughput']} | {row['infer_mem']} | {row['delta_mem']} | {row['param_ratio']} |")

    print()
    print("="*80)
    print(f"Total rows: {len(rows)}")
    print("="*80)

if __name__ == '__main__':
    main()
