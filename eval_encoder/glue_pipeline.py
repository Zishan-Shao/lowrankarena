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
    parser.add_argument("--tasks", nargs="+",
                        choices=list(GLUE_TASKS.keys()),
                        default=["sst2"],
                        help="GLUE tasks to evaluate")

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
            # Default: rank=300
            args.rank = 300
            print(f"[info] Using default rank=300")
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

    # Reuse if already exists
    if pretrain_dir.exists() and args.reuse_checkpoint:
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
    cfg = GLUE_TASKS[task]
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

    # Training loop
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

    return pretrain_dir, best_metric_value


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Compression
# ═══════════════════════════════════════════════════════════════════════════

def get_task_model_id(task: str, args) -> str:
    """Get model ID for a specific task."""
    if not args.use_task_models:
        return args.model_id

    # Get task-specific pretrained models
    task_config = GLUE_TASKS.get(task, {})
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

    # Calculate rank from retention rate if specified
    if args.retention is not None and args.method not in ["dense", "adasvd"]:
        args.rank = calculate_rank_from_retention(args.retention, model_id)

    # Determine model name based on compression method
    # NOTE: Must match run_encoder_benchmark.py's naming convention exactly
    # No task suffix, no retention suffix (retention is just a way to calculate rank)
    if args.method == "adasvd":
        model_name = f"{args.method}_b{args.budget}_{args.backend}"
    elif args.method == "dense":
        model_name = "dense_naive"
    else:
        # For SVD-based methods, include rank info in name
        # Use component-specific naming if specified
        if args.rank_attn is not None or args.rank_ffn is not None or args.rank_wo is not None:
            ra = args.rank_attn if args.rank_attn is not None else args.rank
            rf = args.rank_ffn if args.rank_ffn is not None else args.rank
            rw = args.rank_wo if args.rank_wo is not None else args.rank
            model_name = f"{args.method}_ra{ra}_rf{rf}_rw{rw}_{args.qkv_mode}_{args.backend}"
        elif args.rank is not None:
            model_name = f"{args.method}_r{args.rank}_{args.qkv_mode}_{args.backend}"
        else:
            model_name = f"{args.method}_rNone_{args.qkv_mode}_{args.backend}"

    # Use subdirectory structure when using task-specific models to prevent conflicts
    # Structure: eval_encoder/models/{task}/{model_name}
    # This allows each task to have its own checkpoint without name collisions
    if task and args.use_task_models:
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

    cmd = [
        "python", "eval_encoder/run_encoder_benchmark.py",
        "--model_id", model_id,
        "--method", args.method,
        "--backend", args.backend,
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

        # Add calibration batches for methods that need calibration
        if args.method in ["fwsvd", "drone", "adasvd"]:
            cmd.extend(["--calib_batches", str(args.calib_batches)])

    print(f"\n[cmd] {' '.join(cmd)}\n")

    # Run compression
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Compression failed with exit code {result.returncode}")

    print(f"\n[✓] Compression complete: {checkpoint_path}")
    return checkpoint_path


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Fine-tuning on specific task
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data(task, tokenizer, seq_len, batch_size):
    """Prepare train and validation dataloaders for a GLUE task."""
    cfg = GLUE_TASKS[task]

    train_dataset = load_dataset("glue", task, split=cfg["train_split"])
    val_dataset = load_dataset("glue", task, split=cfg["val_split"])

    def tokenize(examples):
        keys = cfg["sentence_keys"]
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

    # Remove all columns except label
    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in train_dataset.column_names if c != "label"]
    )
    val_dataset = val_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in val_dataset.column_names if c != "label"]
    )

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    def collate_fn(batch):
        result = {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["label"] for x in batch]) if cfg["is_regression"]
                     else torch.tensor([x["label"] for x in batch]),
        }
        # Add token_type_ids if present (for sentence pair tasks)
        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = torch.stack([x["token_type_ids"] for x in batch])
        return result

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def evaluate_task(model, val_loader, task, device, original_model_id=None):
    """Evaluate model on a GLUE task.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        task: GLUE task name
        device: Device to run on
        original_model_id: Optional original model ID (for models loaded from checkpoint)
    """
    cfg = GLUE_TASKS[task]
    metric = load_metric("glue", task)
    model.eval()

    # Check if model needs label remapping for MNLI
    # textattack/bert-base-uncased-MNLI uses non-standard mapping:
    # Model: 0→contradiction, 1→entailment, 2→neutral
    # GLUE:  0→entailment,    1→neutral,     2→contradiction
    # Remapping needed: {0→2, 1→0, 2→1}
    label_remap = None
    if task == "mnli":
        # Try original_model_id first (for models loaded from checkpoint),
        # then fall back to model.config._name_or_path
        model_name = original_model_id or getattr(model.config, '_name_or_path', '')
        if 'textattack' in model_name.lower():
            label_remap = torch.tensor([2, 0, 1], dtype=torch.long, device=device)
            print(f"[info] Applying label remapping for {model_name}: {{0→2, 1→0, 2→1}}")

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

                # Apply label remapping if needed
                if label_remap is not None:
                    preds = label_remap[preds]

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


def evaluate_compressed_model(checkpoint_path: Path, task: str, args) -> Dict:
    """Evaluate compressed model on a specific GLUE task WITHOUT fine-tuning."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Evaluating on {task.upper()} (No Fine-tuning)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GLUE_TASKS[task]

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

    # Evaluation (no fine-tuning)
    print(f"\n[eval] Evaluating compressed model...")
    original_model_id = comp_info.get('model_id', None)
    results, loss = evaluate_task(model, val_loader, task, device, original_model_id=original_model_id)
    print(f"[eval] Results: {results}")

    # Record end time
    end_time = time.time()
    eval_time = end_time - start_time

    print(f"\n{'='*70}")
    print(f"Evaluation Complete: {task.upper()}")
    print("="*70)
    print(f"Results: {results}")
    print(f"Time:    {eval_time:.1f} seconds")
    print("="*70)

    # Benchmark inference speed
    print(f"\n{'='*70}")
    print("STEP 3: Inference Speed Benchmark")
    print("="*70)
    speed_metrics = benchmark_inference_speed(
        model, val_loader, device,
        warmup_steps=10, measure_steps=50
    )

    # Since no fine-tuning, initial = final
    metric_value = results.get(cfg["metric"], 0)

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
            "initial": results,
            "final": results,  # Same as initial (no fine-tuning)
            "best_value": metric_value,
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
        "inference_speed": speed_metrics,
        # Legacy fields for backward compatibility
        "initial_results": results,
        "final_results": results,
        "best_metric": cfg["metric"],
        "best_value": metric_value,
    }


def finetune_on_task(checkpoint_path: Path, task: str, args) -> Dict:
    """Fine-tune compressed model on a specific GLUE task."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Fine-tuning on {task.upper()}")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GLUE_TASKS[task]

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

    # Initial evaluation
    print(f"\n[eval] Initial evaluation...")
    initial_results, initial_loss = evaluate_task(model, val_loader, task, device, original_model_id=original_model_id)
    print(f"[eval] Before fine-tuning: {initial_results}")

    # Training loop
    print(f"\n[train] Training for {args.num_epochs} epochs...")
    best_metric_value = initial_results.get(cfg["metric"], 0)
    best_model_state = None

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

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

    # Benchmark inference speed
    print(f"\n{'='*70}")
    print("STEP 3: Inference Speed Benchmark (Clean Inference State)")
    print("="*70)
    speed_metrics = benchmark_inference_speed(
        model, val_loader, device,
        warmup_steps=10, measure_steps=50
    )

    # Calculate improvement for primary metric
    initial_primary = initial_results.get(cfg["metric"], 0)
    final_primary = final_results.get(cfg["metric"], 0)
    improvement = final_primary - initial_primary

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
            "initial": initial_results,
            "final": final_results,
            "best_value": best_metric_value,
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
        "initial_results": initial_results,
        "final_results": final_results,
        "best_metric": cfg["metric"],
        "best_value": best_metric_value,
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

    # A-Avg calculation (tasks with accuracy as primary or secondary metric)
    accuracy_tasks = ["sst2", "mnli", "qnli", "rte", "mrpc", "qqp"]
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

        # Add to A-Avg if task has accuracy metric
        if task in accuracy_tasks:
            if metric == "accuracy":
                # Primary accuracy metric
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
                pretrained_ckpt, pretrain_metric_value = pretrain_base_model(args, task)
                model_id_override = str(pretrained_ckpt)

            # Step 1: Compress model
            if args.pretrain_before_compress:
                # Always per-task when pretrain_before_compress (each task has its own pre-trained base)
                print(f"\n[task] Compressing pre-trained model for: {task.upper()}")
                checkpoint_path = compress_model(args, task=task, model_id_override=model_id_override)
                checkpoint_paths.append(str(checkpoint_path))
            elif args.use_task_models:
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

            # Fine-tune on task OR just evaluate (if skip_finetuning)
            if args.skip_finetuning:
                print(f"\n[info] Skipping fine-tuning, evaluating compressed model directly...")
                results = evaluate_compressed_model(checkpoint_path, task, args)
            else:
                results = finetune_on_task(checkpoint_path, task, args)

            # Inject pre-training metric if available
            if pretrain_metric_value is not None:
                results["metrics"]["pretrain_value"] = pretrain_metric_value
                results["pretrain_value"] = pretrain_metric_value  # legacy field

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

    print("\n" + "="*115)
    print("GLUE Benchmark Detailed Results")
    print("="*115)
    if has_pretrain:
        print(f"{'Task':<8} | {'Metric':<18} | {'Pretrain':>8} | {'Initial':>7} | {'Final':>7} | {'Δ':>7} | {'Time':<8} | {'Throughput':<12} | {'Latency':<10}")
    else:
        print(f"{'Task':<8} | {'Metric':<18} | {'Initial':>7} | {'Final':>7} | {'Δ':>7} | {'Time':<8} | {'Throughput':<12} | {'Latency':<10}")
    print("-"*115)
    for result in all_results:
        task = result["task"]
        metric = result["best_metric"]
        initial = result["initial_results"].get(metric, 0)
        final = result["final_results"].get(metric, 0)
        improvement = final - initial
        train_time = result.get("training", {}).get("time_minutes", 0)

        # Get speed metrics
        speed = result.get("inference_speed", {})
        throughput = speed.get("throughput_samples_per_sec", 0)
        latency = speed.get("latency_ms_per_batch", 0)

        if has_pretrain:
            pretrain = result.get("pretrain_value")
            pretrain_str = f"{pretrain:8.4f}" if pretrain is not None else "       -"
            print(f"{task.upper():<8} | {metric:<18} | {pretrain_str} | {initial:7.4f} | {final:7.4f} | {improvement:+7.4f} | {train_time:6.1f}m | {throughput:8.1f} s/s | {latency:7.2f} ms")
        else:
            print(f"{task.upper():<8} | {metric:<18} | {initial:7.4f} | {final:7.4f} | {improvement:+7.4f} | {train_time:6.1f}m | {throughput:8.1f} s/s | {latency:7.2f} ms")
    print("="*115)
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
