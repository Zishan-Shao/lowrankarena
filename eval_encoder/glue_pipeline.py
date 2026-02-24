#!/usr/bin/env python3
"""
Complete GLUE benchmark pipeline: Compress → Fine-tune → Evaluate

Usage:
    python eval_encoder/glue_pipeline.py \
        --method fwsvd \
        --rank 300 \
        --backend naive \
        --tasks sst2 cola mrpc \
        --model_id bert-base-uncased

Supported GLUE tasks:
    - cola: Corpus of Linguistic Acceptability (single sentence, classification)
    - sst2: Stanford Sentiment Treebank (single sentence, classification)
    - mrpc: Microsoft Research Paraphrase Corpus (sentence pair, classification)
    - qqp: Quora Question Pairs (sentence pair, classification)
    - mnli: Multi-Genre Natural Language Inference (sentence pair, 3-class)
    - qnli: Question Natural Language Inference (sentence pair, classification)
    - rte: Recognizing Textual Entailment (sentence pair, classification)
    - stsb: Semantic Textual Similarity Benchmark (sentence pair, regression)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm

# Add repo root to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval_encoder.load_compressed_model import load_compressed_model


# ═══════════════════════════════════════════════════════════════════════════
# GLUE Task Configuration
# ═══════════════════════════════════════════════════════════════════════════

GLUE_TASKS = {
    "cola": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "sentence_keys": ("sentence",),
        "metric": "matthews_correlation",
        "is_regression": False,
        "pretrained_models": [
            "textattack/bert-base-uncased-CoLA",
            "howey/bert-base-uncased-cola",
        ]
    },
    "sst2": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "sentence_keys": ("sentence",),
        "metric": "accuracy",
        "is_regression": False,
        "pretrained_models": [
            "textattack/bert-base-uncased-SST-2",
            "howey/bert-base-uncased-sst2",
        ]
    },
    "mrpc": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric": "f1",
        "is_regression": False,
        "pretrained_models": [
            "textattack/bert-base-uncased-MRPC",
            "howey/bert-base-uncased-mrpc",
        ]
    },
    "qqp": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "sentence_keys": ("question1", "question2"),
        "metric": "f1",
        "is_regression": False,
        "pretrained_models": [
            "textattack/bert-base-uncased-QQP",
            "howey/bert-base-uncased-qqp",
        ]
    },
    "mnli": {
        "num_labels": 3,
        "train_split": "train",
        "val_split": "validation_matched",
        "test_split": "test_matched",
        "sentence_keys": ("premise", "hypothesis"),
        "metric": "accuracy",
        "is_regression": False,
        "pretrained_models": [
            "textattack/bert-base-uncased-MNLI",  # Non-standard label mapping (handled by evaluate_task)
            "howey/bert-base-uncased-mnli",
        ]
    },
    "qnli": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "sentence_keys": ("question", "sentence"),
        "metric": "accuracy",
        "is_regression": False,
        "pretrained_models": [
            "textattack/bert-base-uncased-QNLI",
            "howey/bert-base-uncased-qnli",
        ]
    },
    "rte": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric": "accuracy",
        "is_regression": False,
        "pretrained_models": [
            "textattack/bert-base-uncased-RTE",
            "howey/bert-base-uncased-rte",
        ]
    },
    "stsb": {
        "num_labels": 1,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric": "pearson",
        "is_regression": True,
        "pretrained_models": [
            "textattack/bert-base-uncased-STS-B",
            "howey/bert-base-uncased-stsb",
        ]
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SuperGLUE Tasks
# ═══════════════════════════════════════════════════════════════════════════

SUPER_GLUE_TASKS = {
    "boolq": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "validation",
        "sentence_keys": ("passage", "question"),
        "metric": "accuracy",
        "is_regression": False,
        "dataset_name": "super_glue",
        "dataset_config": "boolq",
        # super_glue/boolq ClassLabel: 0=False, 1=True
        "canonical_label2id": {"False": 0, "True": 1},
        # Probe confirmed: howey model has class 0=True, class 1=False → flip [1,0]
        "model_remap_overrides": {"howey/bert-base-uncased-boolq": [1, 0]},
        "pretrained_models": [
            "howey/bert-base-uncased-boolq",
        ]
    },
    "cb": {
        "num_labels": 3,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "validation",
        "sentence_keys": ("premise", "hypothesis"),
        "metric": "accuracy",
        "is_regression": False,
        "dataset_name": "super_glue",
        "dataset_config": "cb",
        # super_glue/cb ClassLabel: 0=entailment, 1=contradiction, 2=neutral
        "canonical_label2id": {"entailment": 0, "contradiction": 1, "neutral": 2},
        # textattack MNLI generic LABEL_X; ordering: 0=contradiction, 1=entailment, 2=neutral
        # CB dataset: 0=entailment, 1=contradiction, 2=neutral → remap [1,0,2]
        "model_remap_overrides": {"textattack/bert-base-uncased-MNLI": [1, 0, 2]},
        "pretrained_models": [
            "textattack/bert-base-uncased-MNLI",  # closest 3-class NLI model
        ]
    },
    "rte_sg": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "validation",
        "sentence_keys": ("premise", "hypothesis"),
        "metric": "accuracy",
        "is_regression": False,
        "dataset_name": "super_glue",
        "dataset_config": "rte",
        # super_glue/rte ClassLabel: 0=entailment, 1=not_entailment
        "canonical_label2id": {"entailment": 0, "not_entailment": 1},
        "pretrained_models": [
            "textattack/bert-base-uncased-RTE",
            "howey/bert-base-uncased-rte",
        ]
    },
    "wic": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "validation",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric": "accuracy",
        "is_regression": False,
        "dataset_name": "super_glue",
        "dataset_config": "wic",
        # super_glue/wic ClassLabel: 0=false, 1=true
        "canonical_label2id": {"false": 0, "true": 1},
        "pretrained_models": [
            "rycecorn/Bert-fine-tuned-WiC",  # verified: acc=0.6881, label ordering matches
            "bert-base-uncased",             # fine-tune from scratch fallback
        ]
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Robustness / Adversarial Tasks (HANS + ANLI)
# ═══════════════════════════════════════════════════════════════════════════

ROBUST_TASKS = {
    "hans": {
        "num_labels": 2,
        "train_split": None,       # eval-only; no fine-tuning
        "val_split": "validation",
        "test_split": "validation",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric": "accuracy",
        "is_regression": False,
        "dataset_name": "hans",
        "dataset_config": None,
        "label_map": {"entailment": 0, "non_entailment": 1, "non-entailment": 1, "nonentailment": 1},  # gold_label is string
        "requires_label_fold": True,  # MNLI 3-class → HANS 2-class
        "pretrained_models": [
            "textattack/bert-base-uncased-MNLI",
        ]
    },
    "anli_r1": {
        "num_labels": 3,
        "train_split": None,   # eval-only: fine-tuning on ANLI defeats adversarial purpose
                               # AND has label-ordering conflict with textattack MNLI base
        "val_split": "test_r1",
        "test_split": "test_r1",
        "sentence_keys": ("premise", "hypothesis"),
        "metric": "accuracy",
        "is_regression": False,
        "dataset_name": "anli",
        "dataset_config": None,
        # ANLI ClassLabel: 0=e(ntailment), 1=n(eutral), 2=c(ontradiction)
        # textattack MNLI: class 0=contradiction, 1=entailment, 2=neutral (LABEL_X generic)
        "canonical_label2id": {"entailment": 0, "neutral": 1, "contradiction": 2},
        "model_remap_overrides": {"textattack/bert-base-uncased-MNLI": [2, 0, 1]},
        "pretrained_models": [
            "textattack/bert-base-uncased-MNLI",
        ]
    },
    "anli_r2": {
        "num_labels": 3,
        "train_split": None,   # eval-only (see anli_r1 comment)
        "val_split": "test_r2",
        "test_split": "test_r2",
        "sentence_keys": ("premise", "hypothesis"),
        "metric": "accuracy",
        "is_regression": False,
        "dataset_name": "anli",
        "dataset_config": None,
        "canonical_label2id": {"entailment": 0, "neutral": 1, "contradiction": 2},
        "model_remap_overrides": {"textattack/bert-base-uncased-MNLI": [2, 0, 1]},
        "pretrained_models": [
            "textattack/bert-base-uncased-MNLI",
        ]
    },
    "anli_r3": {
        "num_labels": 3,
        "train_split": None,   # eval-only (see anli_r1 comment)
        "val_split": "test_r3",
        "test_split": "test_r3",
        "sentence_keys": ("premise", "hypothesis"),
        "metric": "accuracy",
        "is_regression": False,
        "dataset_name": "anli",
        "dataset_config": None,
        "canonical_label2id": {"entailment": 0, "neutral": 1, "contradiction": 2},
        "model_remap_overrides": {"textattack/bert-base-uncased-MNLI": [2, 0, 1]},
        "pretrained_models": [
            "textattack/bert-base-uncased-MNLI",
        ]
    },
}

# Unified task registry
ALL_TASKS = {**GLUE_TASKS, **SUPER_GLUE_TASKS, **ROBUST_TASKS}


# ═══════════════════════════════════════════════════════════════════════════
# Command Line Arguments
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete GLUE pipeline: Compress → Fine-tune → Evaluate"
    )

    # Model configuration
    parser.add_argument("--model_id", default="bert-base-uncased",
                        help="Base model to compress (default: bert-base-uncased)")
    parser.add_argument("--use_task_models", action="store_true",
                        help="Use task-specific pre-trained models from HuggingFace "
                             "(e.g., textattack/bert-base-uncased-SST-2)")
    parser.add_argument("--task_model_prefix", default="textattack",
                        help="Prefix for task-specific models (default: textattack)")
    parser.add_argument("--local_pretrained_dir", default=None,
                        help="Directory containing locally fine-tuned task checkpoints. "
                             "Expects {dir}/{task}/pretrained_base/ per task "
                             "(produced by --pretrain_before_compress). "
                             "Takes priority over --use_task_models.")
    parser.add_argument("--tasks", nargs="+",
                        choices=list(ALL_TASKS.keys()),
                        default=["sst2"],
                        help="Tasks to evaluate (GLUE / SuperGLUE / HANS / ANLI)")

    # Compression configuration
    parser.add_argument("--method",
                        choices=["dense", "svd", "fwsvd", "drone", "adasvd"],
                        default="fwsvd",
                        help="Compression method")
    parser.add_argument("--rank", type=int, default=None,
                        help="Rank for SVD-based methods (mutually exclusive with --retention). "
                             "If --rank_attn/rank_ffn/rank_wo are not specified, this applies to all components.")
    parser.add_argument("--rank_attn", type=int, default=None,
                        help="Rank for Q/K/V attention matrices. Overrides --rank for attention.")
    parser.add_argument("--rank_ffn", type=int, default=None,
                        help="Rank for FFN matrices (Wi, Wo). Overrides --rank for FFN.")
    parser.add_argument("--rank_wo", type=int, default=None,
                        help="Rank for attention output projection (Wo). Overrides --rank for Wo.")
    parser.add_argument("--retention", type=float, default=None,
                        help="Retention rate (0.0-1.0) to automatically calculate rank. "
                             "For BERT-base: retention=0.5 → rank=384, retention=0.3 → rank=230")
    parser.add_argument("--budget", type=float, default=None,
                        help="Budget for AdaSVD (0.0-1.0)")
    parser.add_argument("--qkv_mode", choices=["per_head", "full"], default="per_head",
                        help="QKV factorization mode: 'per_head' (rank limited to head_dim=64) "
                             "or 'full' (paper-style, rank can be 256+). FlashSVD only supports per_head.")
    parser.add_argument("--backend", choices=["naive", "flashsvd"],
                        default="naive",
                        help="Execution backend")
    parser.add_argument("--calib_batches", type=int, default=4,
                        help="Number of calibration batches for FWSVD/DRONE/AdaSVD (default: 4)")
    parser.add_argument("--calib_task", default=None,
                        choices=list(ALL_TASKS.keys()),
                        help="Task to use for calibration data (default: same as each --task). "
                             "Use this to calibrate on a different task than the one being evaluated. "
                             "E.g. --tasks hans anli_r1 --calib_task mnli: "
                             "evaluate on HANS/ANLI, but calibrate on MNLI train for all tasks.")
    # AdaSVD-specific
    parser.add_argument("--adasvd_calib_samples", type=int, default=4000,
                        help="Max calibration samples for AdaSVD ARS (paper: ~4000, batch-level shuffle)")
    parser.add_argument("--adasvd_steps", type=int, default=800,
                        help="AdaSVD hypernetwork training steps (paper ARS default: 800)")
    parser.add_argument("--adasvd_engineering_stable", action="store_true",
                        help="Use learned alpha_z gate in PaperHN (ablation only, not paper-strict)")

    # Fine-tuning configuration
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training and evaluation batch size (increased from 16 for better throughput)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Data configuration
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    # Pipeline control
    parser.add_argument("--skip_compression", action="store_true",
                        help="Skip compression step (use existing checkpoint)")
    parser.add_argument("--skip_finetuning", action="store_true",
                        help="Skip fine-tuning step (evaluate compressed model)")
    parser.add_argument("--checkpoint", default=None,
                        help="Existing checkpoint to use (if skip_compression)")
    parser.add_argument("--reuse_checkpoint", action="store_true",
                        help="Automatically reuse existing checkpoint if available (non-interactive)")
    parser.add_argument("--pretrain_before_compress", action="store_true",
                        help="Fine-tune base model on task first, then compress the fine-tuned model, "
                             "then fine-tune the compressed model. Pipeline: base → finetune → compress → finetune")

    # Output configuration
    parser.add_argument("--output_dir", default="eval_encoder/glue_results")
    parser.add_argument("--save_models", action="store_true",
                        help="Save fine-tuned models")

    args = parser.parse_args()

    # Validate rank/retention mutual exclusivity and set defaults
    if args.method != "adasvd" and args.method != "dense":
        if args.rank is None and args.retention is None:
            # Skip default if all component ranks are explicitly provided
            _all_components_set = (
                args.rank_attn is not None and
                args.rank_ffn is not None and
                args.rank_wo is not None
            )
            if not _all_components_set:
                args.rank = 300
                print(f"[info] Using default rank=300")
            # else: all component ranks specified; global rank not needed
        elif args.rank is not None and args.retention is not None:
            parser.error("--rank and --retention are mutually exclusive. Please specify only one.")
        elif args.retention is not None:
            if not 0.0 < args.retention <= 1.0:
                parser.error("--retention must be between 0.0 and 1.0")
            # Will calculate actual rank later based on model architecture
            print(f"[info] Using retention rate: {args.retention:.2%}")
        else:
            print(f"[info] Using fixed rank: {args.rank}")

    return args


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Calculate rank from retention rate
# ═══════════════════════════════════════════════════════════════════════════

def calculate_rank_from_retention(retention: float, model_id: str = "bert-base-uncased") -> int:
    """
    Calculate rank from retention rate based on model architecture.

    Retention rate is defined as: rank / hidden_size

    Args:
        retention: Retention rate (0.0 - 1.0)
        model_id: Model identifier to determine architecture

    Returns:
        Calculated rank (integer)
    """
    # Model dimension mapping
    MODEL_DIMS = {
        "bert-base-uncased": 768,
        "bert-base-cased": 768,
        "bert-large-uncased": 1024,
        "bert-large-cased": 1024,
        "roberta-base": 768,
        "roberta-large": 1024,
        "albert-base-v2": 768,
        "albert-large-v2": 1024,
    }

    # Extract base model name
    base_model = model_id.split("/")[-1].lower()

    # Determine hidden size
    hidden_size = 768  # Default for BERT-base
    for model_key, dim in MODEL_DIMS.items():
        if model_key in model_id.lower() or base_model in model_key:
            hidden_size = dim
            break

    # Calculate rank
    rank = int(hidden_size * retention)

    # Estimate actual parameter retention for square matrices (attention)
    # For M=N=hidden: param_ratio ≈ 2R/M = 2×retention
    estimated_param_retention = 2 * rank / hidden_size

    print(f"[retention] Model: {model_id}")
    print(f"[retention] Hidden size: {hidden_size}")
    print(f"[retention] Retention rate: {retention:.2%} (input parameter)")
    print(f"[retention] Calculated rank: {rank}")
    print(f"[retention] Estimated actual parameter retention: ~{estimated_param_retention:.1%}")
    print(f"[retention] Note: This is an estimate for square matrices (attention layers)")
    print(f"[retention]       Actual ratio will be calculated after compression")

    return rank


# ═══════════════════════════════════════════════════════════════════════════
# Step 0 (optional): Pre-train base model before compression
# ═══════════════════════════════════════════════════════════════════════════

def _write_csv_row(row_data: dict):
    """Append a row to encoder_runs.csv, creating it with header if needed."""
    import csv as csv_mod
    csv_path = "eval_encoder/eval_results/encoder_runs.csv"
    fields = [
        "timestamp", "model_id", "task", "dataset_split", "dataset_size",
        "seq_len", "batch_size", "dtype",
        "method", "rank", "budget", "scope", "backend", "seed",
        "calib_dataset", "calib_split", "calib_samples", "calib_batches", "calib_seed", "calib_seq_len",
        "metric_name", "metric_value",
        "latency_ms", "throughput_sps",
        "peak_mem_infer_mb", "peak_mem_e2e_mb", "peak_mem_mb",
        "param_ratio", "original_params", "compressed_params",
        "total_param_ratio", "total_original_params", "total_compressed_params",
        "notes", "git_commit",
    ]
    row = {f: "" for f in fields}
    row.update(row_data)
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_pretrain_csv_row(pretrain_dir, task, metric_name, metric_value, args):
    """Write pretrain baseline score directly to encoder_runs.csv."""
    _write_csv_row({
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "model_id": str(pretrain_dir),
        "task": task,
        "dataset_split": "validation",
        "seq_len": str(args.seq_len),
        "batch_size": str(args.batch_size),
        "dtype": "fp32",
        "method": "pretrained_base",
        "backend": "naive",
        "seed": str(args.seed),
        "metric_name": metric_name,
        "metric_value": f"{metric_value:.6f}",
        "notes": f"pretrain_before_compress epochs={args.num_epochs} lr={args.learning_rate}",
    })
    print(f"[pretrain] CSV row written: {metric_name}={metric_value:.4f}")


def _write_finetune_csv_row(checkpoint_path, task, results: dict, args):
    """Write post-compression fine-tuning result to encoder_runs.csv.

    Writes two rows when flashsvd accuracy is available: one for naive, one for flashsvd.
    """
    cfg = ALL_TASKS[task]
    metric_name = cfg["metric"]
    best_value_naive = results.get("best_value", results.get("metrics", {}).get("best_value", 0.0))
    best_value_flash = results.get("best_value_flashsvd")

    # Build rank/budget label from args
    if args.method == "adasvd":
        rank_str = ""
        budget_str = str(args.budget) if args.budget is not None else ""
    else:
        if args.rank_attn is not None or args.rank_ffn is not None or args.rank_wo is not None:
            ra = args.rank_attn if args.rank_attn is not None else args.rank
            rf = args.rank_ffn if args.rank_ffn is not None else args.rank
            rw = args.rank_wo if args.rank_wo is not None else args.rank
            rank_str = f"ra{ra}_rf{rf}_rw{rw}"
        else:
            rank_str = str(args.rank) if args.rank is not None else ""
        budget_str = ""

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    base_row = {
        "timestamp": ts,
        "model_id": str(checkpoint_path),
        "task": task,
        "dataset_split": "validation",
        "seq_len": str(args.seq_len),
        "batch_size": str(args.batch_size),
        "dtype": "fp32",
        "method": f"{args.method}_finetuned",
        "rank": rank_str,
        "budget": budget_str,
        "seed": str(args.seed),
        "notes": f"post_compress_finetune epochs={args.num_epochs} lr={args.learning_rate}",
    }

    # Row 1: naive backend
    _write_csv_row({**base_row,
                    "backend": "naive",
                    "metric_name": metric_name,
                    "metric_value": f"{best_value_naive:.6f}"})
    print(f"[finetune] CSV row written (naive):    {metric_name}={best_value_naive:.4f}")

    # Row 2: flashsvd backend (only if successfully measured)
    if best_value_flash is not None:
        _write_csv_row({**base_row,
                        "backend": "flashsvd",
                        "metric_name": metric_name,
                        "metric_value": f"{best_value_flash:.6f}"})
        print(f"[finetune] CSV row written (flashsvd): {metric_name}={best_value_flash:.4f}")

def pretrain_base_model(args, task: str) -> Path:
    """Fine-tune the base model (bert-base-uncased) on a task before compression.

    Saves to eval_encoder/models/{task}/pretrained_base/.
    Returns the path to the saved checkpoint.
    """
    print(f"\n{'='*70}")
    print(f"STEP 0: Pre-training base model on {task.upper()}")
    print("="*70)

    pretrain_dir = Path("eval_encoder/models") / task / "pretrained_base"
    pretrain_info_file = pretrain_dir / "pretrain_info.json"

    # Reuse if already exists (pretrain is expensive and deterministic; always reuse)
    if pretrain_dir.exists():
        print(f"[exists] Pre-trained checkpoint found: {pretrain_dir}")
        print("[info] Reusing existing pre-trained checkpoint")
        if pretrain_info_file.exists():
            with open(pretrain_info_file) as f:
                saved_info = json.load(f)
            pretrain_metric = saved_info.get("best_metric_value", 0.0)
            print(f"[info] Pre-trained {saved_info.get('metric', '?')}: {pretrain_metric:.4f}")
        else:
            pretrain_metric = 0.0
        return pretrain_dir, pretrain_metric

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ALL_TASKS[task]
    import time
    start_time = time.time()

    # Load base model with correct number of labels
    print(f"[model] Loading base model: {args.model_id}")
    config = AutoConfig.from_pretrained(args.model_id, num_labels=cfg["num_labels"])
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, config=config)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Prepare data
    print(f"\n[data] Loading {task} dataset...")
    train_loader, val_loader = prepare_data(task, tokenizer, args.seq_len, args.batch_size)
    print(f"[data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Initial evaluation
    initial_results, _ = evaluate_task(model, val_loader, task, device)
    print(f"[eval] Initial (base model): {initial_results}")

    best_metric_value = initial_results.get(cfg["metric"], 0)
    best_model_state = None

    # Training loop (ensure grad is enabled — finetune_on_task() disables it globally after cleanup)
    torch.set_grad_enabled(True)
    print(f"\n[train] Pre-training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        results, _ = evaluate_task(model, val_loader, task, device)
        metric_value = results.get(cfg["metric"], 0)
        print(f"\n[epoch {epoch+1}] Results: {results}")
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[epoch {epoch+1}] ✓ New best {cfg['metric']}: {best_metric_value:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Final evaluation
    final_results, _ = evaluate_task(model, val_loader, task, device)
    elapsed = time.time() - start_time
    print(f"\n[pretrain] Final results: {final_results}")
    print(f"[pretrain] Best {cfg['metric']}: {best_metric_value:.4f}  ({elapsed/60:.1f} min)")

    # Save
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(pretrain_dir), safe_serialization=False)
    tokenizer.save_pretrained(str(pretrain_dir))

    # Save pretrain metadata (so reuse can read the metric value)
    pretrain_info = {
        "task": task,
        "model_id": args.model_id,
        "metric": cfg["metric"],
        "best_metric_value": best_metric_value,
        "final_results": final_results,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": elapsed,
    }
    with open(pretrain_dir / "pretrain_info.json", "w") as f:
        json.dump(pretrain_info, f, indent=2)

    print(f"[save] Pre-trained checkpoint saved to: {pretrain_dir}")
    print(f"[save] Best {cfg['metric']}: {best_metric_value:.4f}")

    # Cleanup
    del optimizer, scheduler
    model = model.cpu()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Write pretrain score directly to encoder_runs.csv (score already computed above)
    print(f"\n[pretrain] Recording pretrain score to CSV...")
    _write_pretrain_csv_row(
        pretrain_dir=pretrain_dir,
        task=task,
        metric_name=cfg["metric"],
        metric_value=best_metric_value,
        args=args,
    )

    return pretrain_dir, best_metric_value


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Compression
# ═══════════════════════════════════════════════════════════════════════════

def get_task_model_id(task: str, args) -> str:
    """Get model ID for a specific task.

    Priority:
      1. --local_pretrained_dir  {dir}/{task}/pretrained_base  (local fine-tuned ckpts)
      2. --use_task_models + --task_model_prefix               (HuggingFace task models)
      3. --model_id                                            (base model fallback)
    """
    # 1. Local pretrained checkpoint directory (highest priority)
    local_dir = getattr(args, 'local_pretrained_dir', None)
    if local_dir:
        local_path = Path(local_dir) / task / "pretrained_base"
        if local_path.exists():
            print(f"[model] Using local pretrained checkpoint: {local_path}")
            return str(local_path)
        else:
            print(f"[warn] local_pretrained_dir set but {local_path} not found — falling through")

    # 2. HuggingFace task-specific model
    if not args.use_task_models:
        return args.model_id

    task_config = ALL_TASKS.get(task, {})
    pretrained_models = task_config.get("pretrained_models", [])

    if not pretrained_models:
        print(f"[warn] No task-specific model found for {task}, using base model")
        return args.model_id

    # Find model matching the specified prefix
    for model in pretrained_models:
        if args.task_model_prefix in model:
            return model

    # Fallback to first available
    return pretrained_models[0]


def compress_model(args, task: str = None, model_id_override: str = None) -> Path:
    """Compress model using run_encoder_benchmark.py

    Args:
        args: Command line arguments
        task: Specific task (if using task-specific models)
        model_id_override: Local checkpoint path to compress instead of HuggingFace model
                           (used when pretrain_before_compress=True)
    """
    print("\n" + "="*70)
    print("STEP 1: Model Compression")
    print("="*70)

    if args.skip_compression and args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        print(f"[skip] Using existing checkpoint: {checkpoint_path}")
        return checkpoint_path

    # Get model ID: override (pre-trained local ckpt) > task-specific HF model > base model
    if model_id_override:
        model_id = model_id_override
        print(f"[model] Compressing from local pre-trained checkpoint: {model_id}")
    else:
        model_id = get_task_model_id(task, args) if task else args.model_id
        print(f"[model] Using model: {model_id}")

    # Validate incompatible option combinations
    if getattr(args, "qkv_mode", "per_head") == "full" and getattr(args, "backend", "naive") == "flashsvd":
        raise ValueError(
            "--qkv_mode full is not compatible with --backend flashsvd. "
            "FlashSVD kernels require per-head format (use --qkv_mode per_head)."
        )

    # Calculate rank from retention rate if specified
    if args.retention is not None and args.method not in ["dense", "adasvd"]:
        args.rank = calculate_rank_from_retention(args.retention, model_id)

    # Determine model name based on compression method
    # NOTE: checkpoint name always ends with "_naive" regardless of --backend.
    # Compression (SVD factorisation) is backend-independent; the backend only
    # affects the forward-pass implementation.  Pinning the name to "naive"
    # ensures that flashsvd and naive runs share a single checkpoint so that:
    #   (a) compression runs only once even when BACKENDS="flashsvd naive", and
    #   (b) SVD matrices are guaranteed identical across backends.
    # The backend is applied after loading the checkpoint in evaluate/finetune.
    if args.method == "adasvd":
        model_name = f"{args.method}_b{args.budget}_{args.qkv_mode}_naive"
    elif args.method == "dense":
        model_name = "dense_naive"
    else:
        # For SVD-based methods, include rank info in name
        # Use component-specific naming if specified
        if args.rank_attn is not None or args.rank_ffn is not None or args.rank_wo is not None:
            ra = args.rank_attn if args.rank_attn is not None else args.rank
            rf = args.rank_ffn if args.rank_ffn is not None else args.rank
            rw = args.rank_wo if args.rank_wo is not None else args.rank
            model_name = f"{args.method}_ra{ra}_rf{rf}_rw{rw}_{args.qkv_mode}_naive"
        elif args.rank is not None:
            model_name = f"{args.method}_r{args.rank}_{args.qkv_mode}_naive"
        else:
            model_name = f"{args.method}_rNone_{args.qkv_mode}_naive"

    # Use subdirectory structure when using task-specific models to prevent conflicts
    # Structure: eval_encoder/models/{task}/{model_name}
    # Also required when pretrain_before_compress (model_id_override set): each task
    # has its own pretrained base, so compressed checkpoints must be task-specific too.
    # Same applies when local_pretrained_dir is set (each task has its own local ckpt).
    _has_per_task_model = (
        args.use_task_models or
        model_id_override is not None or
        getattr(args, 'local_pretrained_dir', None) is not None
    )
    if task and _has_per_task_model:
        checkpoint_path = Path("eval_encoder/models") / task / model_name
        save_dir = str(Path("eval_encoder/models") / task)
    else:
        checkpoint_path = Path("eval_encoder/models") / model_name
        save_dir = "eval_encoder/models"

    # Check if already exists
    if checkpoint_path.exists():
        print(f"[exists] Checkpoint already exists: {checkpoint_path}")
        if args.reuse_checkpoint:
            print("[info] Reusing existing checkpoint")
            return checkpoint_path
        else:
            print("[info] Overwriting existing checkpoint")

    # Build compression command
    # Use task-specific validation if available
    validation_task = task if task else "sst2"

    # Resolve effective calibration task for this run
    # --calib_task overrides per-task; falls back to validation_task if not set
    effective_calib_task = args.calib_task or validation_task

    # Early check: calibration-based methods need a train split on the calib task
    _calib_cfg = ALL_TASKS.get(effective_calib_task, {})
    if _calib_cfg.get("train_split") is None and args.method in ["fwsvd", "drone", "adasvd"]:
        raise ValueError(
            f"Calibration task '{effective_calib_task}' has no train split; "
            f"calibration-based methods (fwsvd/drone/adasvd) are not supported. "
            f"Use --calib_task <task_with_train_split> (e.g. --calib_task mnli) "
            f"or switch to --method dense or --method svd."
        )

    cmd = [
        "python", "eval_encoder/run_encoder_benchmark.py",
        "--model_id", model_id,
        "--method", args.method,
        "--backend", "naive",  # always compress with naive; backend applied at eval/benchmark time
        "--task", validation_task,
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--dtype", "fp32",
        "--save_model",
        "--save_dir", save_dir,
        "--full_validation",
    ]

    if args.method != "dense":
        if args.method == "adasvd":
            cmd.extend(["--budget", str(args.budget)])
            cmd.extend(["--adasvd_calib_samples", str(args.adasvd_calib_samples)])
            cmd.extend(["--adasvd_steps", str(args.adasvd_steps)])
            if args.adasvd_engineering_stable:
                cmd.append("--adasvd_engineering_stable")
        else:
            # Add component-specific ranks if specified
            if args.rank_attn is not None:
                cmd.extend(["--rank_attn", str(args.rank_attn)])
            if args.rank_ffn is not None:
                cmd.extend(["--rank_ffn", str(args.rank_ffn)])
            if args.rank_wo is not None:
                cmd.extend(["--rank_wo", str(args.rank_wo)])

            # Always pass base rank when set — run_encoder_benchmark.py uses it
            # as fallback for any component rank not explicitly specified
            if args.rank is not None:
                cmd.extend(["--rank", str(args.rank)])

        # Add qkv_mode
        cmd.extend(["--qkv_mode", args.qkv_mode])

        # Add calibration batches and calib_task for methods that need calibration
        if args.method in ["fwsvd", "drone", "adasvd"]:
            cmd.extend(["--calib_batches", str(args.calib_batches)])
            if effective_calib_task != validation_task:
                cmd.extend(["--calib_task", effective_calib_task])

    print(f"\n[cmd] {' '.join(cmd)}\n")

    # Run compression (naive — also saves checkpoint)
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Compression failed with exit code {result.returncode}")

    print(f"\n[✓] Compression complete: {checkpoint_path}")
    return checkpoint_path


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Fine-tuning on specific task
# ═══════════════════════════════════════════════════════════════════════════

def _load_hans_dataset():
    """
    Load HANS evaluation set from GitHub (or local cache).
    HuggingFace datasets library dropped loading-script support (>= 3.0),
    so we fetch the original TSV directly.

    Returns a HuggingFace Dataset with columns: gold_label, sentence1, sentence2.
    """
    import urllib.request
    from datasets import Dataset as HFDataset

    cache_path = os.path.join(os.path.expanduser("~/.cache"), "hans_eval.tsv")
    if not os.path.exists(cache_path):
        print("[hans] Downloading evaluation set from GitHub...")
        url = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read().decode("utf-8")
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(data)
            print(f"[hans] Cached to {cache_path}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot fetch HANS evaluation set from GitHub: {e}\n"
                f"Please pre-download it manually and place it at: {cache_path}\n"
                f"  wget -O {cache_path} "
                f"https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
            ) from e
    else:
        print(f"[hans] Loading from cache: {cache_path}")

    rows = []
    with open(cache_path, encoding="utf-8") as f:
        header = None
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            rows.append({"gold_label": row["gold_label"],
                         "sentence1": row["sentence1"],
                         "sentence2": row["sentence2"]})

    ds = HFDataset.from_list(rows)
    print(f"[hans] Loaded {len(ds)} examples  (labels: {set(ds['gold_label'])})")
    return ds


def _load_task_dataset(task: str, split: str):
    """Load a dataset for any supported task (GLUE / SuperGLUE / HANS / ANLI)."""
    cfg = ALL_TASKS[task]
    ds_name = cfg.get("dataset_name", "glue")
    ds_cfg = cfg.get("dataset_config", task if ds_name == "glue" else None)
    # HANS uses a loading script not supported by datasets >= 3.0; load directly from GitHub
    if task == "hans":
        return _load_hans_dataset()
    if ds_cfg is not None:
        return load_dataset(ds_name, ds_cfg, split=split)
    return load_dataset(ds_name, split=split)


def prepare_data(task, tokenizer, seq_len, batch_size):
    """Prepare train and validation dataloaders for any supported task."""
    cfg = ALL_TASKS[task]
    label_map = cfg.get("label_map", None)  # e.g. HANS: {"entailment": 0, ...}

    # HANS has no train split — return (None, val_loader)
    train_loader = None
    if cfg["train_split"] is not None:
        train_raw = _load_task_dataset(task, cfg["train_split"])
    val_raw = _load_task_dataset(task, cfg["val_split"])

    def tokenize(examples):
        keys = cfg["sentence_keys"]
        # Remap string labels → int (e.g. HANS gold_label)
        if label_map is not None and "label" not in examples and "gold_label" in examples:
            examples["label"] = [label_map[g] for g in examples["gold_label"]]
        if len(keys) == 1:
            return tokenizer(
                examples[keys[0]],
                padding="max_length",
                truncation=True,
                max_length=seq_len
            )
        else:
            return tokenizer(
                examples[keys[0]], examples[keys[1]],
                padding="max_length",
                truncation=True,
                max_length=seq_len
            )

    def _process(raw):
        # Add integer label column for HANS
        if label_map is not None and "gold_label" in raw.column_names:
            raw = raw.map(lambda ex: {"label": label_map[ex["gold_label"]]})
        keep = "label"
        ds = raw.map(
            tokenize,
            batched=True,
            remove_columns=[c for c in raw.column_names if c != keep]
        )
        ds.set_format("torch")
        return ds

    val_dataset = _process(val_raw)

    def collate_fn(batch):
        result = {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["label"] for x in batch]) if cfg["is_regression"]
                     else torch.tensor([x["label"] for x in batch]),
        }
        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = torch.stack([x["token_type_ids"] for x in batch])
        return result

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    if cfg["train_split"] is not None:
        train_dataset = _process(train_raw)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════
# Label remap helpers
# ═══════════════════════════════════════════════════════════════════════════

def _compute_label_remap(model, task, device, original_model_id=None):
    """
    Compute label remap: model output class index → dataset label index.
    Returns None if no remapping is needed.

    Priority chain (same as evaluate_task):
      1. canonical_label2id name-matching
      2. model_remap_overrides per-model dict
      3. Legacy MNLI textattack hardcoded fallback
    """
    cfg = ALL_TASKS[task]
    canon = cfg.get("canonical_label2id")
    label_remap = None

    if canon is not None:
        model_id2label = getattr(model.config, 'id2label', {})
        if model_id2label:
            try:
                canon_lower = {k.lower(): v for k, v in canon.items()}
                n = len(model_id2label)
                remap_dict = {
                    int(i): canon_lower[str(name).lower()]
                    for i, name in model_id2label.items()
                    if str(name).lower() in canon_lower
                }
                if len(remap_dict) == n:
                    remap_list = [remap_dict[i] for i in range(n)]
                    if remap_list != list(range(n)):
                        label_remap = torch.tensor(remap_list, dtype=torch.long, device=device)
            except Exception as e:
                print(f"[eval] Warning: could not build canonical remap: {e}")

    if label_remap is None:
        overrides = cfg.get("model_remap_overrides", {})
        _model_name = original_model_id or getattr(model.config, '_name_or_path', '')
        for model_key, remap_list in overrides.items():
            if model_key.lower() in _model_name.lower() or _model_name.lower() in model_key.lower():
                label_remap = torch.tensor(remap_list, dtype=torch.long, device=device)
                break

    if task == "mnli" and label_remap is None:
        model_name = original_model_id or getattr(model.config, '_name_or_path', '')
        if 'textattack' in model_name.lower():
            label_remap = torch.tensor([2, 0, 1], dtype=torch.long, device=device)

    return label_remap


def _invert_label_remap(eval_remap: torch.Tensor) -> torch.Tensor:
    """
    Invert an eval-direction remap for use during training.

    Eval remap:  model_class_idx  → dataset_label_idx
    Train remap: dataset_label_idx → model_class_idx
    """
    n = eval_remap.shape[0]
    inv = torch.zeros(n, dtype=torch.long, device=eval_remap.device)
    for model_class in range(n):
        inv[eval_remap[model_class].item()] = model_class
    return inv


def evaluate_task(model, val_loader, task, device, original_model_id=None):
    """Evaluate model on any supported task (GLUE / SuperGLUE / HANS / ANLI).

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        task: Task name (key in ALL_TASKS)
        device: Device to run on
        original_model_id: Optional original model ID (for models loaded from checkpoint)
    """
    cfg = ALL_TASKS[task]
    ds_name = cfg.get("dataset_name", "glue")

    # Choose metric loader
    # MVP: use generic accuracy for all non-GLUE tasks (super_glue/hans/anli) for stability
    if ds_name == "glue":
        metric = load_metric("glue", task)
    else:
        # SuperGLUE, HANS, ANLI — generic accuracy (sufficient for compression trend analysis)
        metric = load_metric("accuracy")

    # ── Canonical label remap ────────────────────────────────────────────────
    # Automatically aligns model output class indices with the dataset's label ordering.
    canon = cfg.get("canonical_label2id")
    label_remap = None
    if canon is not None:
        model_id2label = getattr(model.config, 'id2label', {})
        if model_id2label:
            try:
                canon_lower = {k.lower(): v for k, v in canon.items()}
                n = len(model_id2label)
                remap_dict = {
                    int(i): canon_lower[str(name).lower()]
                    for i, name in model_id2label.items()
                    if str(name).lower() in canon_lower
                }
                if len(remap_dict) == n:
                    remap_list = [remap_dict[i] for i in range(n)]
                    if remap_list != list(range(n)):
                        label_remap = torch.tensor(remap_list, dtype=torch.long, device=device)
            except Exception as e:
                print(f"[eval] Warning: could not build canonical remap: {e}")

    # Fallback: per-model override for models with generic LABEL_X id2label
    if label_remap is None:
        overrides = cfg.get("model_remap_overrides", {})
        _model_name = original_model_id or getattr(model.config, '_name_or_path', '')
        for model_key, remap_list in overrides.items():
            if model_key.lower() in _model_name.lower() or _model_name.lower() in model_key.lower():
                label_remap = torch.tensor(remap_list, dtype=torch.long, device=device)
                break

    # Legacy fallback: MNLI textattack hardcoded remap
    if task == "mnli" and label_remap is None:
        model_name = original_model_id or getattr(model.config, '_name_or_path', '')
        if 'textattack' in model_name.lower():
            label_remap = torch.tensor([2, 0, 1], dtype=torch.long, device=device)

    # HANS: fold MNLI 3-class output → 2-class
    # textattack MNLI: 0=contradiction, 1=entailment, 2=neutral
    # HANS target:     0=entailment,    1=non_entailment
    requires_label_fold = cfg.get("requires_label_fold", False)

    # ── Sanity logging ───────────────────────────────────────────────────────
    _id2label = getattr(model.config, 'id2label', {})
    _num_labels = getattr(model.config, 'num_labels', '?')
    print(f"[eval] id2label={_id2label}  num_labels={_num_labels}")
    if label_remap is not None:
        print(f"[eval] label_remap={label_remap.tolist()}  (canonical={canon})")
    else:
        print(f"[eval] label_remap=None  (labels already canonical or no canonical defined)")
    if requires_label_fold:
        print(f"[eval] fold_rule: mnli_3→{task}_2  (pred==1→0=entailment, else→1=non_entailment)")

    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            if cfg["is_regression"]:
                preds = logits.squeeze()
            else:
                preds = torch.argmax(logits, dim=-1)

                if label_remap is not None:
                    preds = label_remap[preds]

                if requires_label_fold:
                    # pred==1 (entailment) → 0; pred∈{0,2} → 1 (non-entailment)
                    preds = torch.where(
                        preds == 1,
                        torch.zeros_like(preds),
                        torch.ones_like(preds)
                    )

            metric.add_batch(
                predictions=preds.cpu(),
                references=batch["labels"].cpu()
            )

            total_loss += loss.item()
            num_batches += 1

    results = metric.compute()
    avg_loss = total_loss / max(num_batches, 1)

    return results, avg_loss


def benchmark_inference_speed(model, val_loader, device, warmup_steps=10, measure_steps=50):
    """Benchmark inference speed (throughput and latency).

    Args:
        model: Model to benchmark
        val_loader: Validation data loader
        device: Device to run on
        warmup_steps: Number of warmup iterations
        measure_steps: Number of measurement iterations

    Returns:
        dict: Contains throughput (samples/s), latency (ms/batch), and memory (MB)
    """
    import gc
    model.eval()

    # Get batch size from loader
    batch_size = val_loader.batch_size

    # Create iterator
    data_iter = iter(val_loader)

    # Warmup
    print(f"[benchmark] Warmup: {warmup_steps} steps...")
    with torch.no_grad():
        for _ in range(warmup_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(val_loader)
                batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)

    # Synchronize and reset memory stats (ONLY measure inference, not training)
    if device == "cuda" or str(device) == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # Reset here to only measure inference
        gc.collect()

    # Measure
    print(f"[benchmark] Measuring: {measure_steps} steps...")
    start_time = time.time()

    with torch.no_grad():
        for _ in range(measure_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(val_loader)
                batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)

    if device == "cuda" or str(device) == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    elapsed = end_time - start_time

    # Calculate metrics
    total_samples = batch_size * measure_steps
    throughput = total_samples / elapsed  # samples/s
    latency = (elapsed / measure_steps) * 1000  # ms/batch

    # Memory usage (ONLY inference memory, reset before measurement)
    if device == "cuda" or str(device) == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_memory_mb = 0.0

    gc.collect()

    print(f"[benchmark] Throughput: {throughput:.1f} samples/s")
    print(f"[benchmark] Latency: {latency:.2f} ms/batch")
    print(f"[benchmark] Peak Inference Memory: {peak_memory_mb:.1f} MB")

    return {
        "throughput_samples_per_sec": throughput,
        "latency_ms_per_batch": latency,
        "peak_memory_mb": peak_memory_mb,
    }

def _append_flashsvd_csv_row(task, comp_info, speed_metrics, metric_name, metric_value,
                              csv_path="eval_encoder/eval_results/encoder_runs.csv"):
    """Append a flashsvd benchmark row to encoder_runs.csv.

    Finds the most recent naive row matching (model_id, task, method) and copies
    all parameter/dataset fields — only backend, speed metrics, and accuracy differ.
    FlashSVD shares parameters with naive, so param counts are identical.
    """
    import csv as csv_mod
    if not os.path.exists(csv_path):
        print(f"[csv] Skipping flashsvd row: {csv_path} not found")
        return

    model_id = comp_info.get('model_id', '')
    method   = comp_info.get('method', '')

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv_mod.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Find the last naive row that matches this model/task/method
    matching_row = None
    for row in reversed(rows):
        if (row.get('model_id') == model_id and
                row.get('task') == task and
                row.get('method') == method and
                row.get('backend') == 'naive'):
            matching_row = row
            break

    if matching_row is None:
        print(f"[csv] Skipping flashsvd row: no matching naive row for "
              f"model={model_id} task={task} method={method}")
        return

    # Build flashsvd row: copy naive row, update only what differs
    flash_row = dict(matching_row)
    flash_row['timestamp']        = datetime.now().isoformat(timespec='seconds')
    flash_row['backend']          = 'flashsvd'
    flash_row['metric_value']     = f"{metric_value:.6f}"
    flash_row['latency_ms']       = f"{speed_metrics['latency_ms_per_batch']:.2f}"
    flash_row['throughput_sps']   = f"{speed_metrics['throughput_samples_per_sec']:.1f}"
    _pmb = speed_metrics['peak_memory_mb']
    flash_row['peak_mem_infer_mb'] = f"{_pmb:.1f}"
    flash_row['peak_mem_e2e_mb']   = f"{_pmb:.1f}"
    flash_row['peak_mem_mb']       = f"{_pmb:.1f}"  # legacy field

    # DictReader may produce a None key for rows that have more columns than the header.
    # Strip it before writing to avoid "dict contains fields not in fieldnames: None".
    flash_row.pop(None, None)

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(flash_row)
    print(f"[csv] ✅ FlashSVD row appended to {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# FlashSVD save / restore helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_encoder_layers(model):
    """Return encoder layer list for BERT / RoBERTa models (None for others)."""
    if hasattr(model, 'bert'):
        return model.bert.encoder.layer
    if hasattr(model, 'roberta'):
        return model.roberta.encoder.layer
    return None


def _enable_flashsvd_save(model):
    """Enable FlashSVD in-place; return saved (layer, naive_block) pairs for restore.

    The FlashSVDBlock shares all nn.Parameter objects with the original NaiveSVDBlock
    (no copies), so the saved naive_block references remain valid after training.
    """
    from eval_encoder.flashsvd_backend import enable_flashsvd
    layers = _get_encoder_layers(model)
    saved = []
    if layers is not None:
        for layer in layers:
            block = getattr(layer, 'block', None)
            if block is not None and type(block).__name__ in ('NaiveSVDBlock', 'MinimalSVDBlock'):
                saved.append((layer, block))
    enable_flashsvd(model)

    # Verify parameter identity: FlashSVDBlock must hold the SAME Parameter objects
    # as NaiveSVDBlock (not copies). If this assert fires, FlashSVDBlock.__init__
    # started copying tensors — training would NOT update the flash block's weights.
    _SHARED = ('Pq', 'Vq', 'Pk', 'Vk', 'Pv', 'Vv', 'Uo', 'Vo', 'U1', 'V1', 'U2', 'V2')
    for layer, naive_block in saved[:1]:   # one layer is enough to catch the bug
        flash_block = layer.block
        for attr in _SHARED:
            n_p = getattr(naive_block, attr, None)
            f_p = getattr(flash_block, attr, None)
            if n_p is not None and f_p is not None:
                assert id(n_p) == id(f_p), (
                    f"[BUG] FlashSVDBlock.{attr} is a COPY, not a reference to NaiveSVDBlock.{attr}. "
                    f"Training will NOT update FlashSVD weights. Check FlashSVDBlock.__init__."
                )

    return saved


def _restore_naive(saved):
    """Restore NaiveSVDBlock instances saved by _enable_flashsvd_save (undo FlashSVD swap)."""
    for layer, naive_block in saved:
        layer.block = naive_block
    if saved:
        print(f"[flashsvd] Restored {len(saved)} layers to NaiveSVDBlock (for training).")


def evaluate_compressed_model(checkpoint_path: Path, task: str, args) -> Dict:
    """Evaluate compressed model on a specific GLUE task WITHOUT fine-tuning."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Evaluating on {task.upper()} (No Fine-tuning)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ALL_TASKS[task]

    # Record start time
    import time
    start_time = time.time()

    # Load compressed model
    model, tokenizer, comp_info = load_compressed_model(
        str(checkpoint_path),
        device=device,
        dtype=torch.float32,
    )

    # Prepare data
    print(f"\n[data] Loading {task} dataset...")
    train_loader, val_loader = prepare_data(task, tokenizer, args.seq_len, args.batch_size)
    train_batches = len(train_loader) if train_loader is not None else 0
    print(f"[data] Train batches: {train_batches}, Val batches: {len(val_loader)}")

    # Get dataset sizes
    train_size = len(train_loader.dataset) if train_loader is not None else 0
    val_size = len(val_loader.dataset)

    # Evaluate with naive backend (always first)
    original_model_id = comp_info.get('model_id', None)
    _comp_method = comp_info.get('method', args.method)
    print(f"\n[eval] Evaluating with naive backend...")
    results_naive, loss = evaluate_task(model, val_loader, task, device, original_model_id=original_model_id)
    print(f"[eval] naive: {results_naive}")

    # Benchmark naive speed (before enabling FlashSVD)
    print(f"\n{'='*70}")
    print("STEP 3a: Inference Speed Benchmark (naive)")
    print("="*70)
    speed_metrics_naive = benchmark_inference_speed(
        model, val_loader, device,
        warmup_steps=10, measure_steps=50
    )

    # Evaluate + benchmark with flashsvd backend (SVD methods only, per_head only)
    results_flashsvd = None
    speed_metrics_flash = None
    if _comp_method != 'dense' and getattr(args, 'qkv_mode', 'per_head') == 'per_head':
        try:
            from eval_encoder.flashsvd_backend import enable_flashsvd
            enable_flashsvd(model)
            print(f"\n[eval] Evaluating with flashsvd backend...")
            results_flashsvd, _ = evaluate_task(model, val_loader, task, device, original_model_id=original_model_id)
            print(f"[eval] flashsvd: {results_flashsvd}")

            print(f"\n{'='*70}")
            print("STEP 3b: Inference Speed Benchmark (flashsvd)")
            print("="*70)
            speed_metrics_flash = benchmark_inference_speed(
                model, val_loader, device,
                warmup_steps=10, measure_steps=50
            )
        except RuntimeError as e:
            print(f"[eval] FlashSVD unavailable: {e}")

    results = results_naive

    # Record end time
    end_time = time.time()
    eval_time = end_time - start_time

    print(f"\n{'='*70}")
    print(f"Evaluation Complete: {task.upper()}")
    print("="*70)
    print(f"Naive:    {results_naive}")
    if results_flashsvd is not None:
        print(f"FlashSVD: {results_flashsvd}")
    print(f"Time:    {eval_time:.1f} seconds")
    print("="*70)

    # Since no fine-tuning, initial = final
    metric_value = results_naive.get(cfg["metric"], 0)
    metric_value_flash = (results_flashsvd.get(cfg["metric"], 0)
                          if results_flashsvd is not None else None)

    # Write flashsvd row to CSV
    if results_flashsvd is not None and speed_metrics_flash is not None:
        _append_flashsvd_csv_row(
            task, comp_info, speed_metrics_flash,
            cfg["metric"], metric_value_flash,
        )

    return {
        "task": task,
        "dataset": {
            "train_size": train_size,
            "val_size": val_size,
            "num_labels": cfg["num_labels"],
            "is_regression": cfg["is_regression"],
        },
        "metrics": {
            "primary_metric": cfg["metric"],
            "initial": results_naive,
            "initial_naive": results_naive,
            "initial_flashsvd": results_flashsvd,
            "final": results_naive,
            "final_naive": results_naive,
            "final_flashsvd": results_flashsvd,
            "best_value": metric_value,
            "best_value_flashsvd": metric_value_flash,
            "improvement": 0.0,  # No improvement without fine-tuning
        },
        "training": {
            "num_epochs": 0,  # No training
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "total_steps": 0,
            "time_seconds": eval_time,
            "time_minutes": eval_time / 60,
        },
        "inference_speed": speed_metrics_naive,
        "inference_speed_naive": speed_metrics_naive,
        "inference_speed_flashsvd": speed_metrics_flash,
        # Legacy fields for backward compatibility
        "initial_results": results_naive,
        "initial_results_naive": results_naive,
        "initial_results_flashsvd": results_flashsvd,
        "final_results": results_naive,
        "final_results_naive": results_naive,
        "final_results_flashsvd": results_flashsvd,
        "best_metric": cfg["metric"],
        "best_value": metric_value,
        "best_value_flashsvd": metric_value_flash,
    }


def finetune_on_task(checkpoint_path: Path, task: str, args) -> Dict:
    """Fine-tune compressed model on a specific task.

    If the task has no train split (e.g. HANS), skip fine-tuning and
    delegate to evaluate_compressed_model instead.
    """
    cfg = ALL_TASKS[task]

    # Tasks without a train split (e.g. HANS) cannot be fine-tuned
    if cfg.get("train_split") is None:
        print(f"\n[info] Task '{task}' has no train split — skipping fine-tuning, eval only.")
        return evaluate_compressed_model(checkpoint_path, task, args)

    print(f"\n{'='*70}")
    print(f"STEP 2: Fine-tuning on {task.upper()}")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Record start time
    import time
    start_time = time.time()

    # Load compressed model
    model, tokenizer, comp_info = load_compressed_model(
        str(checkpoint_path),
        device=device,
        dtype=torch.float32,
    )

    # Prepare data
    print(f"\n[data] Loading {task} dataset...")
    train_loader, val_loader = prepare_data(task, tokenizer, args.seq_len, args.batch_size)
    print(f"[data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Get dataset sizes
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Get original model ID for label remapping
    original_model_id = comp_info.get('model_id', None)

    # Compute training label remap (inverse of eval remap).
    # Required when model class ordering differs from dataset label ordering
    # (e.g. howey/bert-base-uncased-boolq: model class 0=True, dataset 0=False).
    _eval_remap = _compute_label_remap(model, task, device, original_model_id)
    train_label_remap = None
    if _eval_remap is not None:
        train_label_remap = _invert_label_remap(_eval_remap)
        print(f"[train] eval label_remap:  {_eval_remap.tolist()}")
        print(f"[train] train label_remap: {train_label_remap.tolist()} (inverted for training alignment)")

    # Initial evaluation — naive backend
    _comp_method = comp_info.get('method', args.method)
    print(f"\n[eval] Initial evaluation (naive)...")
    initial_results, initial_loss = evaluate_task(model, val_loader, task, device, original_model_id=original_model_id)
    initial_results_naive = initial_results
    print(f"[eval] Before fine-tuning (naive): {initial_results}")

    # Initial evaluation — flashsvd backend (save/restore keeps naive blocks for training)
    initial_results_flashsvd = None
    if _comp_method != 'dense' and getattr(args, 'qkv_mode', 'per_head') == 'per_head':
        try:
            _saved_blocks = _enable_flashsvd_save(model)
            print(f"\n[eval] Initial evaluation (flashsvd)...")
            initial_results_flashsvd, _ = evaluate_task(model, val_loader, task, device, original_model_id=original_model_id)
            print(f"[eval] Before fine-tuning (flashsvd): {initial_results_flashsvd}")
            _restore_naive(_saved_blocks)   # restore NaiveSVDBlock for gradient-based training
        except RuntimeError as e:
            print(f"[eval] FlashSVD unavailable for initial eval: {e}")

    # Safety check: ensure no FlashSVDBlock is present before training.
    # Triton kernels don't support autograd; training through them would silently
    # produce zero gradients or crash.
    from eval_encoder.flashsvd_backend import FlashSVDBlock as _FSB
    _flash_found = [type(m).__name__ for m in model.modules()
                    if isinstance(m, _FSB) or type(m).__name__ == 'FlashSVDBlock']
    assert not _flash_found, (
        f"FlashSVDBlock found in model before training: {_flash_found}. "
        "_restore_naive() must be called before the training loop."
    )

    # Training loop (ensure grad is enabled — previous task cleanup may have disabled it globally)
    torch.set_grad_enabled(True)
    print(f"\n[train] Training for {args.num_epochs} epochs...")
    best_metric_value = initial_results.get(cfg["metric"], 0)
    best_model_state = None

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            if train_label_remap is not None and not cfg["is_regression"]:
                batch = dict(batch)
                batch["labels"] = train_label_remap[batch["labels"]]

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # End of epoch evaluation
        results, val_loss = evaluate_task(model, val_loader, task, device, original_model_id=original_model_id)
        metric_value = results.get(cfg["metric"], 0)

        print(f"\n[epoch {epoch+1}] Results: {results}")
        print(f"[epoch {epoch+1}] Loss: {val_loss:.4f}")

        # Save best model
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[epoch {epoch+1}] ✓ New best {cfg['metric']}: {best_metric_value:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Final evaluation
    final_results, final_loss = evaluate_task(model, val_loader, task, device, original_model_id=original_model_id)

    # Record end time
    end_time = time.time()
    training_time = end_time - start_time

    print(f"\n{'='*70}")
    print(f"Fine-tuning Complete: {task.upper()}")
    print("="*70)
    print(f"Initial: {initial_results}")
    print(f"Final:   {final_results}")
    print(f"Time:    {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print("="*70)

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL: Clean up training memory before inference benchmark
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("MEMORY CLEANUP: Removing training artifacts")
    print("="*70)

    # Ensure model is in eval mode
    model.eval()
    torch.set_grad_enabled(False)

    # Delete optimizer and scheduler (frees ~2x model parameters)
    print("[cleanup] Deleting optimizer and scheduler...")
    del optimizer, scheduler

    # Clear all gradients
    print("[cleanup] Clearing gradients...")
    model.zero_grad(set_to_none=True)

    # Force Python garbage collection
    print("[cleanup] Running garbage collection...")
    import gc
    gc.collect()

    # Clear CUDA cache and reset peak memory stats
    if device.type == "cuda" or str(device) == "cuda":
        print("[cleanup] Clearing CUDA cache and resetting memory stats...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Get current memory before reset (for logging)
        pre_cleanup_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
        post_cleanup_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[cleanup] Memory: {pre_cleanup_mem:.1f} MB → {post_cleanup_mem:.1f} MB")

    print("="*70)

    # Enable FlashSVD for final accuracy eval and throughput benchmark.
    # Training always runs naive (Triton kernels don't support autograd).
    # We switch here, after the optimizer is gone, so both accuracy and
    # throughput measurements reflect the real FlashSVD execution path.
    final_results_naive = final_results
    final_results_flashsvd = None
    if _comp_method != 'dense' and getattr(args, 'qkv_mode', 'per_head') == 'per_head':
        try:
            from eval_encoder.flashsvd_backend import enable_flashsvd
            enable_flashsvd(model)
            model.eval()                    # ensure dropout is off for accuracy measurement
            torch.set_grad_enabled(False)   # no gradients needed past this point
            print(f"\n[eval] Final evaluation (flashsvd)...")
            final_results_flashsvd, _ = evaluate_task(model, val_loader, task, device, original_model_id=original_model_id)
            print(f"[eval] After fine-tuning (flashsvd): {final_results_flashsvd}")
        except RuntimeError as e:
            print(f"[eval] FlashSVD unavailable for final eval: {e}")

    # Benchmark inference speed (FlashSVD already enabled if available)
    print(f"\n{'='*70}")
    print("STEP 3: Inference Speed Benchmark (Clean Inference State)")
    print("="*70)
    speed_metrics = benchmark_inference_speed(
        model, val_loader, device,
        warmup_steps=10, measure_steps=50
    )

    # Calculate improvement for primary metric (naive → naive)
    initial_primary = initial_results_naive.get(cfg["metric"], 0)
    final_primary = final_results_naive.get(cfg["metric"], 0)
    improvement = final_primary - initial_primary

    final_metric_flashsvd = (final_results_flashsvd.get(cfg["metric"], 0)
                              if final_results_flashsvd is not None else None)
    initial_metric_flashsvd = (initial_results_flashsvd.get(cfg["metric"], 0)
                                if initial_results_flashsvd is not None else None)

    return {
        "task": task,
        "dataset": {
            "train_size": train_size,
            "val_size": val_size,
            "num_labels": cfg["num_labels"],
            "is_regression": cfg["is_regression"],
        },
        "metrics": {
            "primary_metric": cfg["metric"],
            "initial": initial_results_naive,
            "initial_naive": initial_results_naive,
            "initial_flashsvd": initial_results_flashsvd,
            "final": final_results_naive,
            "final_naive": final_results_naive,
            "final_flashsvd": final_results_flashsvd,
            "best_value": best_metric_value,
            "best_value_flashsvd": final_metric_flashsvd,
            "improvement": improvement,
        },
        "training": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "total_steps": len(train_loader) * args.num_epochs,
            "time_seconds": training_time,
            "time_minutes": training_time / 60,
        },
        "inference_speed": speed_metrics,
        # Legacy fields for backward compatibility
        "initial_results": initial_results_naive,
        "initial_results_naive": initial_results_naive,
        "initial_results_flashsvd": initial_results_flashsvd,
        "final_results": final_results_naive,
        "final_results_naive": final_results_naive,
        "final_results_flashsvd": final_results_flashsvd,
        "best_metric": cfg["metric"],
        "best_value": best_metric_value,
        "best_value_flashsvd": final_metric_flashsvd,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Calculation
# ═══════════════════════════════════════════════════════════════════════════

def calculate_averages(all_results):
    """
    Calculate G-Avg (GLUE Average) and A-Avg (Accuracy Average).

    G-Avg: Average of all primary metrics across all tasks
           For MCC and Pearson (range -1~1), normalize to 0~1 first

    A-Avg: Average of accuracy-based metrics only
           Includes: SST-2, MNLI, QNLI, RTE (primary accuracy)
                     MRPC, QQP (secondary accuracy if available)
    """
    if not all_results:
        return 0.0, 0.0, 0.0, 0.0

    # G-Avg calculation
    g_scores_initial = []
    g_scores_final = []

    # A-Avg: all tasks whose primary metric is accuracy (auto-detected, covers GLUE + SuperGLUE + HANS + ANLI)
    a_scores_initial = []
    a_scores_final = []

    for result in all_results:
        task = result["task"]
        metric = result["best_metric"]

        # Get initial and final scores
        initial = result["initial_results"].get(metric, 0)
        final = result["best_value"]

        # Normalize MCC and Pearson correlation from [-1, 1] to [0, 1]
        if metric in ["matthews_correlation", "pearson"]:
            initial_normalized = (initial + 1) / 2
            final_normalized = (final + 1) / 2
        else:
            initial_normalized = initial
            final_normalized = final

        # Add to G-Avg
        g_scores_initial.append(initial_normalized)
        g_scores_final.append(final_normalized)

        # Add to A-Avg: primary accuracy metric + MRPC/QQP secondary accuracy
        if metric == "accuracy":
            a_scores_initial.append(initial)
            a_scores_final.append(final)
        elif task in ["mrpc", "qqp"] and "accuracy" in result["initial_results"]:
            # MRPC and QQP also report accuracy as secondary metric
            acc_initial = result["initial_results"].get("accuracy", 0)
            acc_final = result["final_results"].get("accuracy", 0)
            a_scores_initial.append(acc_initial)
            a_scores_final.append(acc_final)

    # Calculate averages
    g_avg_initial = sum(g_scores_initial) / len(g_scores_initial) if g_scores_initial else 0.0
    g_avg_final = sum(g_scores_final) / len(g_scores_final) if g_scores_final else 0.0

    a_avg_initial = sum(a_scores_initial) / len(a_scores_initial) if a_scores_initial else 0.0
    a_avg_final = sum(a_scores_final) / len(a_scores_final) if a_scores_final else 0.0

    return g_avg_initial, g_avg_final, a_avg_initial, a_avg_final


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    """Run complete GLUE evaluation pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    all_results = []
    checkpoint_path = None  # Initialize checkpoint path
    checkpoint_paths = []  # Track all checkpoint paths

    print("\n" + "="*70)
    print("GLUE Benchmark Pipeline")
    print("="*70)
    print(f"Model:       {args.model_id}")
    print(f"Method:      {args.method}")
    print(f"Rank:        {args.rank if args.method != 'adasvd' else 'N/A'}")
    print(f"Budget:      {args.budget if args.method == 'adasvd' else 'N/A'}")
    print(f"Backend:     {args.backend}")
    print(f"Tasks:       {', '.join(args.tasks)}")
    print(f"Fine-tune:   {'No (evaluate only)' if args.skip_finetuning else 'Yes'}")
    if not args.skip_finetuning:
        print(f"Epochs:      {args.num_epochs}")
        print(f"Batch size:  {args.batch_size}")
        print(f"LR:          {args.learning_rate}")
    print("="*70)
    if args.pretrain_before_compress:
        print(f"[mode] Pretrain-before-compress: base → finetune → compress → finetune")

    # Step 0/1/2: For each task
    for task in args.tasks:
        try:
            # Step 0 (optional): pre-train base model on task
            model_id_override = None
            pretrain_metric_value = None
            if args.pretrain_before_compress:
                _task_cfg = ALL_TASKS[task]
                if _task_cfg.get("train_split") is None:
                    print(f"[warn] Task '{task}' has no train split — skipping pretrain_before_compress step.")
                    print(f"[warn] Pipeline continues: compress → eval (no pretrained base for this task).")
                    print(f"[warn] (HANS has no supervised training set; ANLI is intentionally eval-only.)")
                else:
                    pretrained_ckpt, pretrain_metric_value = pretrain_base_model(args, task)
                    model_id_override = str(pretrained_ckpt)

            # Step 1: Compress model
            if args.pretrain_before_compress:
                # Always per-task when pretrain_before_compress (each task has its own pre-trained base)
                print(f"\n[task] Compressing pre-trained model for: {task.upper()}")
                checkpoint_path = compress_model(args, task=task, model_id_override=model_id_override)
                checkpoint_paths.append(str(checkpoint_path))
            elif args.use_task_models or getattr(args, 'local_pretrained_dir', None):
                print(f"\n[task] Processing task-specific model for: {task.upper()}")
                checkpoint_path = compress_model(args, task=task)
                checkpoint_paths.append(str(checkpoint_path))
            else:
                # Use shared compressed model for all tasks
                if checkpoint_path is None:
                    print(f"\n[shared] Compressing shared model (will be used for all tasks)")
                    checkpoint_path = compress_model(args, task=None)
                    checkpoint_paths.append(str(checkpoint_path))
                else:
                    print(f"\n[shared] Reusing shared compressed model: {checkpoint_path}")

            # Fine-tune on task OR just evaluate (if skip_finetuning or dense method)
            # Dense model is already task-fine-tuned; re-training is redundant compute.
            if args.skip_finetuning or args.method == 'dense':
                if args.method == 'dense' and not args.skip_finetuning:
                    print(f"\n[info] Dense method: skipping fine-tuning (model already task-fine-tuned).")
                else:
                    print(f"\n[info] Skipping fine-tuning, evaluating compressed model directly...")
                results = evaluate_compressed_model(checkpoint_path, task, args)
            else:
                results = finetune_on_task(checkpoint_path, task, args)

            # Inject pre-training metric if available
            if pretrain_metric_value is not None:
                results["metrics"]["pretrain_value"] = pretrain_metric_value
                results["pretrain_value"] = pretrain_metric_value  # legacy field

            # Write post-compression fine-tune result to CSV (if fine-tuning was done)
            if not args.skip_finetuning and args.method != 'dense':
                _write_finetune_csv_row(checkpoint_path, task, results, args)

            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error on task {task}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # 清理GPU内存，避免累积导致CUDA错误
            import torch
            import gc
            if torch.cuda.is_available():
                # 清空缓存
                torch.cuda.empty_cache()
                # 同步所有操作
                torch.cuda.synchronize()
                # 重置内存统计（可选，帮助监控）
                torch.cuda.reset_peak_memory_stats()
                # 显示当前内存使用
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"[cleanup] GPU memory after task {task}: {mem_alloc:.1f}MB allocated, {mem_reserved:.1f}MB reserved")
            # Python垃圾回收
            gc.collect()
            print(f"[cleanup] Cleanup complete for task {task}")

    # Calculate G-Avg and A-Avg
    g_avg_initial, g_avg_final, a_avg_initial, a_avg_final = calculate_averages(all_results)

    # Save results (include backend to avoid conflicts)
    results_file = output_dir / f"glue_results_{args.method}_{args.backend}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "config": vars(args),
            "checkpoint": checkpoint_paths[0] if len(checkpoint_paths) == 1 else checkpoint_paths,
            "results": all_results,
            "summary": {
                "G-Avg": {
                    "initial": g_avg_initial,
                    "final": g_avg_final,
                    "improvement": g_avg_final - g_avg_initial
                },
                "A-Avg": {
                    "initial": a_avg_initial,
                    "final": a_avg_final,
                    "improvement": a_avg_final - a_avg_initial
                }
            }
        }, f, indent=2)

    # Print detailed summary
    has_pretrain = any("pretrain_value" in r for r in all_results)

    if has_pretrain:
        # pretrain_before_compress mode:
        # Pretrain | Compressed | Finetuned | Δ(ft-pre)
        # Δ = Finetuned - Pretrain (net effect vs original baseline)
        W = 125
        print("\n" + "="*W)
        print("GLUE Benchmark Detailed Results  [mode: pretrain → compress → finetune]")
        print("="*W)
        print(f"{'Task':<8} | {'Metric':<18} | {'Pretrain':>8} | {'Compressed':>10} | {'Finetuned':>9} | {'Δ(ft-pre)':>9} | {'Time':<8} | {'Throughput':<12} | {'Latency':<10}")
        print("-"*W)
        for result in all_results:
            task = result["task"]
            metric = result["best_metric"]
            pretrain  = result.get("pretrain_value")
            compressed = result["initial_results"].get(metric, 0)   # after compress, before ft
            finetuned  = result["best_value"]                        # after post-compress ft
            delta = (finetuned - pretrain) if pretrain is not None else (finetuned - compressed)
            train_time = result.get("training", {}).get("time_minutes", 0)
            speed = result.get("inference_speed", {})
            throughput = speed.get("throughput_samples_per_sec", 0)
            latency = speed.get("latency_ms_per_batch", 0)
            pre_str = f"{pretrain:8.4f}" if pretrain is not None else "       -"
            print(f"{task.upper():<8} | {metric:<18} | {pre_str} | {compressed:10.4f} | {finetuned:9.4f} | {delta:+9.4f} | {train_time:6.1f}m | {throughput:8.1f} s/s | {latency:7.2f} ms")
        print("="*W)
    else:
        # Normal mode: Before FT | After FT | Δ
        W = 138
        print("\n" + "="*W)
        print("GLUE Benchmark Detailed Results")
        print("="*W)
        print(f"{'Task':<8} | {'Metric':<12} | {'Pre-N':>7} | {'Pre-F':>7} | {'Post-N':>7} | {'Post-F':>7} | {'Δ(N)':>7} | {'Δ(F)':>7} | {'Time':<7} | {'Throughput':<12} | {'Latency':<10}")
        print("-"*W)
        for result in all_results:
            task = result["task"]
            metric = result["best_metric"]
            pre_n = (result.get("initial_results_naive") or result["initial_results"]).get(metric, 0)
            _pfd = result.get("initial_results_flashsvd")
            pre_f = _pfd.get(metric, 0) if _pfd is not None else None
            post_n = (result.get("final_results_naive") or result["final_results"]).get(metric, 0)
            _qfd = result.get("final_results_flashsvd")
            post_f = _qfd.get(metric, 0) if _qfd is not None else None
            delta_n = post_n - pre_n
            delta_f = (post_f - pre_f) if (pre_f is not None and post_f is not None) else None
            train_time = result.get("training", {}).get("time_minutes", 0)
            speed = result.get("inference_speed", {})
            throughput = speed.get("throughput_samples_per_sec", 0)
            latency = speed.get("latency_ms_per_batch", 0)
            pre_f_s  = f"{pre_f:7.4f}"  if pre_f  is not None else "    N/A"
            post_f_s = f"{post_f:7.4f}" if post_f is not None else "    N/A"
            delta_f_s = f"{delta_f:+7.4f}" if delta_f is not None else "    N/A"
            print(f"{task.upper():<8} | {metric:<12} | {pre_n:7.4f} | {pre_f_s} | {post_n:7.4f} | {post_f_s} | {delta_n:+7.4f} | {delta_f_s} | {train_time:5.1f}m | {throughput:8.1f} s/s | {latency:7.2f} ms")
        print("="*W)
    print(f"\n{'Final Evaluation Scores':^100}")
    print("="*100)
    print(f"G-Avg (GLUE Average):")
    print(f"  Initial:     {g_avg_initial:.4f}")
    print(f"  Final:       {g_avg_final:.4f}")
    print(f"  Improvement: {g_avg_final - g_avg_initial:+.4f}")
    print(f"\nA-Avg (Accuracy Average):")
    print(f"  Initial:     {a_avg_initial:.4f}")
    print(f"  Final:       {a_avg_final:.4f}")
    print(f"  Improvement: {a_avg_final - a_avg_initial:+.4f}")
    print("="*100)
    print(f"\n✅ Results saved to: {results_file}")

    return all_results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    try:
        results = run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
