#!/usr/bin/env python3
"""
Generate comparison table in the format of academic papers.

Usage:
    python eval_encoder/scripts/generate_comparison_table.py \
        eval_encoder/glue_results/glue_results_*.json \
        --format markdown
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List


# Model parameter counts (in millions)
BERT_BASE_PARAMS = 109.5  # Million parameters


def calculate_compression_ratio(rank: int, hidden_size: int = 768, num_layers: int = 12) -> float:
    """
    Calculate parameter compression ratio.

    For attention layers (Q, K, V projections):
    - Original: 3 * (hidden_size * hidden_size) = 3 * 768^2 per layer
    - Compressed: 3 * 2 * (hidden_size * rank) per layer

    Total for all layers:
    - Original params ~= 109.5M
    - Compressed params = calculate based on rank
    """
    # Attention Q/K/V projections per layer
    original_attn_params = 3 * hidden_size * hidden_size  # Q, K, V: 768x768 each
    compressed_attn_params = 3 * 2 * hidden_size * rank   # Q, K, V: (768xR + Rx768) each

    # FFN per layer (also compressed)
    ffn_intermediate = 3072  # BERT-base FFN intermediate size
    original_ffn_params = hidden_size * ffn_intermediate + ffn_intermediate * hidden_size
    compressed_ffn_params = 2 * hidden_size * rank + 2 * rank * ffn_intermediate

    # Attention output projection (also compressed)
    original_out_params = hidden_size * hidden_size
    compressed_out_params = 2 * hidden_size * rank

    # Per-layer compression
    original_per_layer = original_attn_params + original_ffn_params + original_out_params
    compressed_per_layer = compressed_attn_params + compressed_ffn_params + compressed_out_params

    # Total for all layers
    total_original = original_per_layer * num_layers
    total_compressed = compressed_per_layer * num_layers

    # Add embedding and other non-compressed params (roughly 30M)
    embedding_params = 30 * 1e6  # Approximate
    total_model_original = total_original + embedding_params
    total_model_compressed = total_compressed + embedding_params

    # Compression ratio as percentage of original
    ratio = total_model_compressed / (BERT_BASE_PARAMS * 1e6) * 100

    return ratio, total_model_compressed / 1e6  # Return ratio and absolute params in millions


def load_results(file_path: str) -> Dict:
    """Load results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def extract_task_scores(results: List[Dict]) -> Dict[str, tuple]:
    """
    Extract initial and final scores for each task.

    Returns:
        Dict mapping task name to (initial_score, final_score)
    """
    scores = {}

    task_metrics = {
        "cola": "matthews_correlation",
        "sst2": "accuracy",
        "mrpc": "f1",  # Primary metric
        "qqp": "f1",   # Primary metric
        "mnli": "accuracy",
        "qnli": "accuracy",
        "rte": "accuracy",
        "stsb": "pearson",
    }

    for result in results:
        task = result.get("task", "").lower()
        if task not in task_metrics:
            continue

        metric = task_metrics[task]
        initial = result.get("initial_results", {}).get(metric, 0.0)
        final = result.get("final_results", {}).get(metric, 0.0)

        # Convert MCC and Pearson from [-1, 1] to [0, 100] scale
        if metric in ["matthews_correlation", "pearson"]:
            initial = (initial + 1) / 2 * 100
            final = (final + 1) / 2 * 100
        else:
            initial = initial * 100
            final = final * 100

        scores[task] = (initial, final)

    return scores


def format_score(score: float, decimal: int = 1) -> str:
    """Format score with specified decimal places."""
    if score == 0.0:
        return "−"
    return f"{score:.{decimal}f}"


def generate_markdown_table(data_list: List[Dict]) -> str:
    """Generate Markdown table."""
    lines = []

    # Header
    header = "| Model | #Param | CoLA | MNLI | MRPC | QNLI | QQP | SST-2 | STS-B | G-Avg | A-Avg |"
    separator = "|-------|--------|------|------|------|------|-----|-------|-------|-------|-------|"

    lines.append(header)
    lines.append(separator)

    # Original BERT baseline
    lines.append("| **Original BERT-base** | 100.0% | 56.2 | 84.7 | 87.4 | 91.3 | 87.8 | 93.0 | 88.5 | **84.1** | **85.4** |")

    # Each method
    for data in data_list:
        config = data['config']
        scores = data['scores']
        summary = data['summary']

        method = config['method'].upper()
        rank = config.get('rank', 'N/A')
        retention = config.get('retention')

        # Calculate compression ratio
        if isinstance(rank, int):
            ratio, params_m = calculate_compression_ratio(rank)
            param_str = f"{ratio:.1f}%"
        else:
            param_str = "N/A"

        # Model name
        if retention:
            model_name = f"{method} (r={rank}, ρ={retention:.1f})"
        else:
            model_name = f"{method} (r={rank})"

        # Initial scores (after compression)
        initial_line = f"| {model_name} | {param_str} |"
        for task in ["cola", "mnli", "mrpc", "qnli", "qqp", "sst2", "stsb"]:
            if task in scores:
                initial_line += f" {format_score(scores[task][0])} |"
            else:
                initial_line += " − |"

        # G-Avg and A-Avg initial
        g_avg_init = summary['G-Avg']['initial'] * 100
        a_avg_init = summary['A-Avg']['initial'] * 100
        initial_line += f" {format_score(g_avg_init)} | {format_score(a_avg_init)} |"
        lines.append(initial_line)

        # Final scores (after fine-tuning)
        final_line = f"| + fine-tuning | {param_str} |"
        for task in ["cola", "mnli", "mrpc", "qnli", "qqp", "sst2", "stsb"]:
            if task in scores:
                final_line += f" {format_score(scores[task][1])} |"
            else:
                final_line += " − |"

        # G-Avg and A-Avg final
        g_avg_final = summary['G-Avg']['final'] * 100
        a_avg_final = summary['A-Avg']['final'] * 100
        final_line += f" **{format_score(g_avg_final)}** | **{format_score(a_avg_final)}** |"
        lines.append(final_line)

    return "\n".join(lines)


def generate_latex_table(data_list: List[Dict]) -> str:
    """Generate LaTeX table."""
    lines = []

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of compression methods on GLUE benchmark}")
    lines.append("\\begin{tabular}{lcccccccccc}")
    lines.append("\\toprule")
    lines.append("Model & \\#Param & CoLA & MNLI & MRPC & QNLI & QQP & SST-2 & STS-B & G-Avg & A-Avg \\\\")
    lines.append("\\midrule")

    # Original BERT baseline
    lines.append("Original BERT-base & 100.0\\% & 56.2 & 84.7 & 87.4 & 91.3 & 87.8 & 93.0 & 88.5 & \\textbf{84.1} & \\textbf{85.4} \\\\")

    # Each method
    for data in data_list:
        config = data['config']
        scores = data['scores']
        summary = data['summary']

        method = config['method'].upper()
        rank = config.get('rank', 'N/A')

        # Calculate compression ratio
        if isinstance(rank, int):
            ratio, params_m = calculate_compression_ratio(rank)
            param_str = f"{ratio:.1f}\\%"
        else:
            param_str = "N/A"

        # Model name
        model_name = f"{method} (r={rank})"

        # Initial scores
        initial_vals = [param_str]
        for task in ["cola", "mnli", "mrpc", "qnli", "qqp", "sst2", "stsb"]:
            if task in scores:
                initial_vals.append(format_score(scores[task][0]))
            else:
                initial_vals.append("−")

        g_avg_init = summary['G-Avg']['initial'] * 100
        a_avg_init = summary['A-Avg']['initial'] * 100
        initial_vals.extend([format_score(g_avg_init), format_score(a_avg_init)])

        lines.append(f"{model_name} & {' & '.join(initial_vals)} \\\\")

        # Final scores
        final_vals = [param_str]
        for task in ["cola", "mnli", "mrpc", "qnli", "qqp", "sst2", "stsb"]:
            if task in scores:
                final_vals.append(format_score(scores[task][1]))
            else:
                final_vals.append("−")

        g_avg_final = summary['G-Avg']['final'] * 100
        a_avg_final = summary['A-Avg']['final'] * 100
        final_vals.extend([f"\\textbf{{{format_score(g_avg_final)}}}", f"\\textbf{{{format_score(a_avg_final)}}}"])

        lines.append(f"+ fine-tuning & {' & '.join(final_vals)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_csv_table(data_list: List[Dict]) -> str:
    """Generate CSV table."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Model", "#Param", "CoLA", "MNLI", "MRPC", "QNLI", "QQP", "SST-2", "STS-B", "G-Avg", "A-Avg"])

    # Original BERT
    writer.writerow(["Original BERT-base", "100.0%", "56.2", "84.7", "87.4", "91.3", "87.8", "93.0", "88.5", "84.1", "85.4"])

    # Each method
    for data in data_list:
        config = data['config']
        scores = data['scores']
        summary = data['summary']

        method = config['method'].upper()
        rank = config.get('rank', 'N/A')

        if isinstance(rank, int):
            ratio, _ = calculate_compression_ratio(rank)
            param_str = f"{ratio:.1f}%"
        else:
            param_str = "N/A"

        model_name = f"{method} (r={rank})"

        # Initial row
        initial_row = [model_name, param_str]
        for task in ["cola", "mnli", "mrpc", "qnli", "qqp", "sst2", "stsb"]:
            if task in scores:
                initial_row.append(format_score(scores[task][0]))
            else:
                initial_row.append("−")

        g_avg_init = summary['G-Avg']['initial'] * 100
        a_avg_init = summary['A-Avg']['initial'] * 100
        initial_row.extend([format_score(g_avg_init), format_score(a_avg_init)])
        writer.writerow(initial_row)

        # Final row
        final_row = ["+ fine-tuning", param_str]
        for task in ["cola", "mnli", "mrpc", "qnli", "qqp", "sst2", "stsb"]:
            if task in scores:
                final_row.append(format_score(scores[task][1]))
            else:
                final_row.append("−")

        g_avg_final = summary['G-Avg']['final'] * 100
        a_avg_final = summary['A-Avg']['final'] * 100
        final_row.extend([format_score(g_avg_final), format_score(a_avg_final)])
        writer.writerow(final_row)

    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Generate comparison table")
    parser.add_argument("result_files", nargs="+", help="Result JSON files")
    parser.add_argument("--format", choices=["markdown", "latex", "csv"], default="markdown")
    parser.add_argument("--output", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Load all results
    data_list = []
    for pattern in args.result_files:
        files = glob.glob(pattern)
        for file in files:
            try:
                data = load_results(file)
                scores = extract_task_scores(data.get('results', []))
                data_list.append({
                    'config': data.get('config', {}),
                    'scores': scores,
                    'summary': data.get('summary', {}),
                })
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")

    if not data_list:
        print("Error: No valid result files found")
        return 1

    # Generate table
    if args.format == "markdown":
        table = generate_markdown_table(data_list)
    elif args.format == "latex":
        table = generate_latex_table(data_list)
    else:  # csv
        table = generate_csv_table(data_list)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(table)
        print(f"✅ Table saved to: {args.output}")
    else:
        print(table)

    return 0


if __name__ == "__main__":
    exit(main())
