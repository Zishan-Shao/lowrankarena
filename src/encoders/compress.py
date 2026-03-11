#!/usr/bin/env python3
"""
Encoder low-rank compression benchmark.

Three execution modes:
  1) Dense-Naive:      original model, default PyTorch execution
  2) LowRank-Naive:    compressed weights, standard GEMM execution
  3) LowRank-FlashSVD: compressed weights, Triton FlashSVD kernels

Example
-------
# Dense baseline
python eval_encoder/run_encoder_benchmark.py --method dense --backend naive --task sst2

# FWSVD r=128, naive
python eval_encoder/run_encoder_benchmark.py --method fwsvd --rank 128 --backend naive --task sst2

# FWSVD r=128, flash
python eval_encoder/run_encoder_benchmark.py --method fwsvd --rank 128 --backend flashsvd --task sst2
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.encoders.blocks_misc import (
    BertLayerShim, NaiveSVDBlock,
    ModernBertLayerShim, NaiveModernBertSVDBlock,
)

# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Encoder low-rank compression benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # model / task
    p.add_argument("--model_id", default="bert-base-uncased")
    p.add_argument("--task", choices=list(TASK_CFG.keys()), default="sst2")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    # compression
    p.add_argument("--method", choices=["dense", "svd", "fwsvd", "drone", "adasvd"], default="dense")
    p.add_argument("--rank", type=int, default=None,
                   help="Unified rank for all components (will be auto-clipped to component-specific max ranks)")
    p.add_argument("--rank_attn", type=int, default=None,
                   help="Rank for Q/K/V attention matrices (per-head). If specified, overrides --rank for attention.")
    p.add_argument("--rank_ffn", type=int, default=None,
                   help="Rank for FFN matrices (Wi, Wo). If specified, overrides --rank for FFN.")
    p.add_argument("--rank_wo", type=int, default=None,
                   help="Rank for attention output projection (Wo). If specified, overrides --rank for Wo.")
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--scope", choices=["qkv", "ffn", "qkv+ffn", "qkv+wo+ffn"], default="qkv+ffn")
    p.add_argument("--qkv_mode", choices=["per_head", "full"], default="per_head",
                   help="QKV factorization mode: 'per_head' (rank limited to dh=64) or 'full' (paper-style, rank can be 256+)")
    # backend
    p.add_argument("--backend", choices=["naive", "flashsvd", "flashsvd15"], default="naive")
    p.add_argument("--attn_mode", choices=["einsum", "sdpa"], default="einsum",
                   help="Attention implementation for the naive backend. "
                        "'einsum': standard O(n²) attention (paper-faithful, default). "
                        "'sdpa': PyTorch 2.0 SDPA (avoids [B,H,M,M] materialization; "
                        "ablation to isolate flash-attention benefit from FlashSVD kernel).")
    # shortcut: load pre-compressed model from disk, skip compression
    p.add_argument("--load_model_dir", default=None,
                   help="Load a previously saved compressed model from this directory "
                        "(produced by --save_model). Skips compression entirely. "
                        "Use with --skip_eval --attn_mode sdpa to only measure SDPA performance.")
    # logging / perf
    p.add_argument("--out_csv", default="experiments/encoder_runs.csv")
    p.add_argument("--notes", default="")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--measure_steps", type=int, default=50)
    p.add_argument("--num_runs", type=int, default=1,
                   help="Number of times to run performance measurement (median will be reported)")
    p.add_argument("--full_validation", action="store_true",
                   help="Measure performance over full validation set (like src/profile_*.py). "
                        "When enabled, ignores --measure_steps and --num_runs, traverses entire dataset once. "
                        "Provides more realistic end-to-end performance but slower to measure.")
    p.add_argument("--reload_before_perf", action="store_true", default=True,
                   help="Move compressed model to CPU and back to GPU before performance "
                        "measurement to ensure accurate inference-only peak memory. "
                        "Enabled by default to avoid measuring compression-phase memory "
                        "as inference memory (old BertLayer params may still be alive via "
                        "reference cycles when reset_peak_memory_stats() is called). "
                        "Use --no_reload_before_perf to disable.")
    p.add_argument("--no_reload_before_perf", dest="reload_before_perf", action="store_false",
                   help="Disable GPU reload before performance measurement (faster but "
                        "peak_mem_infer_mb may include compression-phase memory).")
    # calibration (for fwsvd / drone)
    p.add_argument("--calib_batches", type=int, default=4,
                   help="Number of batches used for Fisher / covariance calibration")
    p.add_argument("--calib_split", choices=["train"], default="train",
                   help="Which split to use for calibration (MUST be train, NOT validation)")
    p.add_argument("--calib_seed", type=int, default=None,
                   help="Random seed for calibration data sampling (default: same as --seed)")
    p.add_argument("--calib_task", default=None, choices=list(TASK_CFG.keys()),
                   help="Task to use for calibration data (default: same as --task). "
                        "Use this to calibrate on a different task than the one being evaluated. "
                        "E.g. --task hans --calib_task mnli: evaluate on HANS, calibrate on MNLI train.")
    # AdaSVD-specific (paper-compliant adasvd_origin)
    p.add_argument("--adasvd_calib_samples", type=int, default=4000,
                   help="Max calibration samples for AdaSVD ARS (paper: ~4000, batch-level shuffle)")
    p.add_argument("--adasvd_steps", type=int, default=800,
                   help="AdaSVD hypernetwork training steps (paper ARS default: 800)")
    p.add_argument("--adasvd_engineering_stable", action="store_true",
                   help="Use learned alpha_z gate in PaperHN (ablation only, not paper-strict)")
    # model saving
    p.add_argument("--save_model", action="store_true",
                   help="Save compressed model to disk for later fine-tuning")
    p.add_argument("--save_dir", default="compressed_models",
                   help="Directory to save compressed models")
    # skip flags
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip task metric evaluation (only measure performance). "
                        "Useful with --load_model_dir --attn_mode sdpa to purely benchmark speed/memory.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Task configuration
# ═══════════════════════════════════════════════════════════════════════════
TASK_CFG = {
    "cola": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("sentence",), metric_name="matthews_correlation",
    ),
    "sst2": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("sentence",), metric_name="accuracy",
    ),
    "mrpc": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("sentence1", "sentence2"), metric_name="f1",
    ),
    "qqp": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("question1", "question2"), metric_name="f1",
    ),
    "mnli": dict(
        num_labels=3, val_split="validation_matched", train_split="train",
        sentence_keys=("premise", "hypothesis"), metric_name="accuracy",
    ),
    "qnli": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("question", "sentence"), metric_name="accuracy",
    ),
    "rte": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("sentence1", "sentence2"), metric_name="accuracy",
    ),
    "stsb": dict(
        num_labels=1, val_split="validation", train_split="train",
        sentence_keys=("sentence1", "sentence2"), metric_name="pearson",
        is_regression=True,
    ),
}

TASK_CFG.update({
    # ── SuperGLUE ──────────────────────────────────────────────────────────
    "boolq": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("passage", "question"), metric_name="accuracy",
        dataset_name="super_glue", dataset_config="boolq",
        # super_glue/boolq ClassLabel: 0=False, 1=True
        canonical_label2id={"False": 0, "True": 1},
        # Per-model remaps for models with generic LABEL_0/LABEL_1 naming.
        # Probe confirmed: howey model has class 0=True, class 1=False → flip [1,0]
        model_remap_overrides={"howey/bert-base-uncased-boolq": [1, 0]},
    ),
    "cb": dict(
        num_labels=3, val_split="validation", train_split="train",
        sentence_keys=("premise", "hypothesis"), metric_name="accuracy",
        dataset_name="super_glue", dataset_config="cb",
        # super_glue/cb ClassLabel: 0=entailment, 1=contradiction, 2=neutral
        canonical_label2id={"entailment": 0, "contradiction": 1, "neutral": 2},
        # textattack MNLI has generic LABEL_X; ordering: 0=contradiction, 1=entailment, 2=neutral
        # CB dataset: 0=entailment, 1=contradiction, 2=neutral → remap [1,0,2]
        model_remap_overrides={"textattack/bert-base-uncased-MNLI": [1, 0, 2]},
    ),
    "rte_sg": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("premise", "hypothesis"), metric_name="accuracy",
        dataset_name="super_glue", dataset_config="rte",
        # super_glue/rte ClassLabel: 0=entailment, 1=not_entailment
        canonical_label2id={"entailment": 0, "not_entailment": 1},
    ),
    "wic": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("sentence1", "sentence2"), metric_name="accuracy",
        dataset_name="super_glue", dataset_config="wic",
        # super_glue/wic ClassLabel: 0=false, 1=true
        canonical_label2id={"false": 0, "true": 1},
    ),
    # ── HANS ──────────────────────────────────────────────────────────────
    "hans": dict(
        num_labels=2, val_split="validation", train_split=None,
        sentence_keys=("sentence1", "sentence2"), metric_name="accuracy",
        dataset_name="hans", dataset_config=None,
        label_map={"entailment": 0, "non_entailment": 1, "non-entailment": 1, "nonentailment": 1},
        requires_label_fold=True,   # MNLI 3-class → HANS 2-class (no canonical_label2id needed)
    ),
    # ── ANLI ──────────────────────────────────────────────────────────────
    # ANLI dataset ClassLabel: 0=e(ntailment), 1=n(eutral), 2=c(ontradiction)
    # textattack MNLI: class 0=contradiction, 1=entailment, 2=neutral (LABEL_X generic names)
    # model_remap_overrides covers the case where canonical name-matching can't auto-detect
    # (remap [2,0,1]: contradiction→2, entailment→0, neutral→1)
    "anli_r1": dict(
        num_labels=3, val_split="test_r1", train_split="train_r1",
        sentence_keys=("premise", "hypothesis"), metric_name="accuracy",
        dataset_name="anli", dataset_config=None,
        canonical_label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
        model_remap_overrides={"textattack/bert-base-uncased-MNLI": [2, 0, 1]},
    ),
    "anli_r2": dict(
        num_labels=3, val_split="test_r2", train_split="train_r2",
        sentence_keys=("premise", "hypothesis"), metric_name="accuracy",
        dataset_name="anli", dataset_config=None,
        canonical_label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
        model_remap_overrides={"textattack/bert-base-uncased-MNLI": [2, 0, 1]},
    ),
    "anli_r3": dict(
        num_labels=3, val_split="test_r3", train_split="train_r3",
        sentence_keys=("premise", "hypothesis"), metric_name="accuracy",
        dataset_name="anli", dataset_config=None,
        canonical_label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
        model_remap_overrides={"textattack/bert-base-uncased-MNLI": [2, 0, 1]},
    ),
    # ── COPA ─────────────────────────────────────────────────────────────
    # Two-choice causal reasoning.  Cannot be scored with standard argmax —
    # uses NLI-based two-choice scoring (see _prepare_copa_loader / _evaluate_copa).
    # Model: textattack/bert-base-uncased-MNLI (class 1 = entailment).
    "copa": dict(
        num_labels=3, val_split="validation", train_split="train",
        sentence_keys=("premise", "choice1"),   # unused when copa_style=True
        metric_name="accuracy",
        dataset_name="super_glue", dataset_config="copa",
        copa_style=True,   # activates two-choice NLI evaluation path
    ),
})

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# ═══════════════════════════════════════════════════════════════════════════
# 1) Model loading
# ═══════════════════════════════════════════════════════════════════════════
def load_model(model_id: str, task: str, dtype_str: str, device: str):
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

    cfg = TASK_CFG[task]

    # Load model without overriding config (preserves trained classification head)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Verify num_labels matches task requirements
    expected_labels = cfg["num_labels"]
    actual_labels = model.config.num_labels
    if actual_labels != expected_labels:
        print(f"[warning] Model has {actual_labels} labels, task expects {expected_labels}")
        print(f"[warning] Using model's original configuration to preserve trained weights")

    pt_dtype = DTYPE_MAP[dtype_str]
    model = model.to(device=device, dtype=pt_dtype).eval()
    return model, tokenizer


def _detect_arch(model):
    """Detect model architecture. Returns (arch_name, encoder_layers)."""
    model_type = getattr(model.config, "model_type", "").lower()
    if model_type == "modernbert":
        return "modernbert", model.model.layers
    if hasattr(model, "roberta"):
        return "roberta", model.roberta.encoder.layer
    if hasattr(model, "bert"):
        return "bert", model.bert.encoder.layer
    raise RuntimeError(
        f"Unsupported model architecture (model_type={model_type}). "
        "Expected bert, roberta, or modernbert."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2) Dataset
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


def _prepare_copa_loader(tokenizer, seq_len: int, batch_size: int, split_name: str):
    """
    COPA two-choice NLI loader.

    For each COPA example (premise, question, choice1, choice2, label), build two NLI pairs:
      question=="effect":  (premise, choice_i)
      question=="cause":   (choice_i, premise)

    Returns a DataLoader with 2*batch_size items per batch:
      [choice1_pair_0, choice2_pair_0, choice1_pair_1, choice2_pair_1, ...]
    Labels are the original COPA labels (0 or 1), repeated: [lbl_0, lbl_0, lbl_1, lbl_1, ...]
    In _evaluate_copa, even indices = choice1, odd indices = choice2.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset

    raw = load_dataset("super_glue", "copa", split=split_name)

    sents1, sents2, labels_interleaved = [], [], []
    for ex in raw:
        q   = ex["question"]
        p   = ex["premise"]
        c1, c2, lbl = ex["choice1"], ex["choice2"], ex["label"]
        if q == "effect":
            sents1 += [p, p];   sents2 += [c1, c2]
        else:                   # cause
            sents1 += [c1, c2]; sents2 += [p, p]
        labels_interleaved += [lbl, lbl]

    enc = tokenizer(
        sents1, sents2,
        padding="max_length", truncation=True, max_length=seq_len,
        return_tensors="pt",
    )
    labels_t = torch.tensor(labels_interleaved, dtype=torch.long)
    tti      = enc.get("token_type_ids", None)

    class _CopaDataset(Dataset):
        def __len__(self): return len(labels_t)
        def __getitem__(self, i):
            item = {"input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "labels": labels_t[i]}
            if tti is not None:
                item["token_type_ids"] = tti[i]
            return item

    return DataLoader(_CopaDataset(), batch_size=batch_size * 2, shuffle=False)


@torch.no_grad()
def _evaluate_copa(model, loader, device) -> float:
    """
    Two-choice NLI scoring for COPA.

    Entailment class is detected from model.config.id2label:
    - textattack MNLI (generic LABEL_X): defaults to class 1 (0=contra, 1=entail, 2=neutral)
    - ModernBERT MNLI (HuggingFace GLUE order): class 0 = entailment
    For each pair of consecutive items in the batch (choice1, choice2),
    compare entailment logits and predict the higher-scoring choice.
    """
    _id2label = getattr(model.config, 'id2label', {})
    ENTAIL_IDX = 1  # textattack MNLI default
    for _cid, _cname in _id2label.items():
        if str(_cname).lower() == "entailment":
            ENTAIL_IDX = int(_cid)
            break
    print(f"[copa] entailment class index = {ENTAIL_IDX}  (id2label={_id2label})")
    correct = total = 0
    model.eval()
    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        labels  = batch.pop("labels")                  # [2*N]
        logits  = model(**batch).logits                # [2*N, 3]
        entail  = logits[:, ENTAIL_IDX]                # [2*N]
        ec1     = entail[0::2]                         # [N] choice1 scores
        ec2     = entail[1::2]                         # [N] choice2 scores
        lbl     = labels[0::2]                         # [N] original labels
        preds   = (ec2 > ec1).long()                   # 0→choice1 wins, 1→choice2 wins
        correct += (preds == lbl).sum().item()
        total   += lbl.shape[0]
    return correct / total if total > 0 else 0.0


def prepare_loader(task: str, tokenizer, seq_len: int, batch_size: int, split: str = "validation"):
    """
    Prepare DataLoader for any supported task (GLUE / SuperGLUE / HANS / ANLI).

    Args:
        task: Task name (key in TASK_CFG)
        tokenizer: HuggingFace tokenizer
        seq_len: Maximum sequence length
        batch_size: Batch size
        split: "train" or "validation"
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    cfg = TASK_CFG[task]
    if split == "train":
        split_name = cfg["train_split"]
    else:
        split_name = cfg["val_split"]

    # COPA: two-choice NLI scoring — special interleaved loader
    if cfg.get("copa_style", False):
        return _prepare_copa_loader(tokenizer, seq_len, batch_size, split_name)

    # Generic dataset loading
    ds_name = cfg.get("dataset_name", "glue")
    ds_cfg = cfg.get("dataset_config", task if ds_name == "glue" else None)

    # HANS uses a loading script not supported by datasets >= 3.0; load directly from GitHub
    if task == "hans":
        raw = _load_hans_dataset()
    elif ds_cfg is not None:
        raw = load_dataset(ds_name, ds_cfg, split=split_name)
    else:
        raw = load_dataset(ds_name, split=split_name)

    keys = cfg["sentence_keys"]
    label_map = cfg.get("label_map", None)  # e.g. HANS string → int

    # For HANS: gold_label is a string column; add integer "label" column
    if label_map is not None and "gold_label" in raw.column_names:
        raw = raw.map(lambda ex: {"label": label_map[ex["gold_label"]]})

    def tok_fn(batch):
        if len(keys) == 1:
            return tokenizer(batch[keys[0]], padding="max_length",
                             truncation=True, max_length=seq_len)
        return tokenizer(batch[keys[0]], batch[keys[1]], padding="max_length",
                         truncation=True, max_length=seq_len)

    remove = [c for c in raw.column_names if c != "label"]
    ds = raw.map(tok_fn, batched=True, remove_columns=remove)
    ds.set_format("torch")

    is_regression = cfg.get("is_regression", False)

    def collate(batch):
        result = {
            "input_ids":      torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels":         torch.stack([x["label"] for x in batch]) if is_regression
                              else torch.tensor([x["label"] for x in batch]),
        }
        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = torch.stack([x["token_type_ids"] for x in batch])
        return result

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def _make_calib_loader(loader, max_batches: int):
    """Yield at most *max_batches* from an existing loader."""
    from torch.utils.data import DataLoader, Subset

    ds = loader.dataset
    n = min(max_batches * loader.batch_size, len(ds))
    subset = Subset(ds, list(range(n)))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        collate_fn=loader.collate_fn,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3) Compression methods
# ═══════════════════════════════════════════════════════════════════════════

# ---------- plain SVD helpers (fallback) ----------
def _build_plain_svd_helpers(model):
    """Mimics utils.encoder_utils.svd_helpers.build_plain_svd_helpers."""
    def svd_per_head(Wt: torch.Tensor, rank: int):
        d_model, _ = Wt.shape
        H = model.config.num_attention_heads
        dh = d_model // H
        Wt3 = Wt.view(d_model, H, dh)
        Us, Vs = [], []
        for h in range(H):
            Wh = Wt3[:, h, :].float()
            U, S, Vh = torch.linalg.svd(Wh, full_matrices=False)
            Us.append((U[:, :rank] * S[:rank]).to(Wt.dtype))
            Vs.append(Vh[:rank, :].to(Wt.dtype))
        return torch.stack(Us, 0), torch.stack(Vs, 0)

    def svd_low_rank(W: torch.Tensor, rank: int):
        Wf = W.float()
        U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
        return (U[:, :rank] * S[:rank]).to(W.dtype), Vh[:rank, :].to(W.dtype)

    return svd_per_head, svd_low_rank


# ---------- FWSVD ----------
def _build_fwsvd(model, calib_loader, device, arch):
    """Build Fisher-weighted SVD helpers using the repo's existing fwsvd module."""
    if arch == "modernbert":
        from src.encoders.blocks.fwsvd import build_fwsvd_helpers_modernbert
        result = build_fwsvd_helpers_modernbert(model, calib_loader, device)
        model.eval()
        return result
    try:
        from src.encoders.blocks.svd_helpers import build_fwsvd_helpers
    except ImportError:
        raise RuntimeError(
            "Cannot import build_fwsvd_helpers from utils.encoder_utils.svd_helpers. "
            "Make sure the repo root is on PYTHONPATH."
        )
    # For RoBERTa, temporarily alias model.bert so the BERT-specific helper works
    _cleanup = False
    if arch == "roberta" and not hasattr(model, "bert"):
        model.bert = model.roberta
        _cleanup = True
    try:
        fwsvd_per_head, fwsvd_low_rank = build_fwsvd_helpers(
            model, calib_loader, device=device
        )
    finally:
        if _cleanup:
            del model.bert
    model.eval()  # build_fwsvd_helpers sets model to .train()
    return fwsvd_per_head, fwsvd_low_rank


# ---------- DRONE (data-aware SVD) ----------
def _safe_cholesky(C: torch.Tensor, max_tries: int = 5, base_eps: float = 1e-6):
    D = C.shape[-1]
    eps = base_eps * float(C.diag().mean().item() + 1.0)
    I = torch.eye(D, dtype=C.dtype, device=C.device)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(C + eps * I)
        except RuntimeError:
            eps *= 10.0
    return torch.linalg.cholesky(
        C + (1e-2 * float(C.diag().mean().item() + 1.0)) * I
    )


def _data_aware_low_rank(W: torch.Tensor, rank: int, cov_in: torch.Tensor):
    d_in, d_out = W.shape
    Wf, Cf = W.float(), cov_in.float()
    S = _safe_cholesky(Cf)
    A = Wf.t().contiguous() @ S
    try:
        U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    except torch._C._LinAlgError:
        # Ill-conditioned matrix: fall back to Jacobi driver (slower but more stable)
        try:
            U, s, Vh = torch.linalg.svd(A, full_matrices=False, driver="gesvd")
        except (torch._C._LinAlgError, Exception):
            # Last resort: add small ridge to A^T A before SVD
            ridge = 1e-4 * float(A.norm()) + 1e-8
            AtA = A.t() @ A + ridge * torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
            s2, Vh2 = torch.linalg.eigh(AtA)
            s2 = torch.clamp(s2, min=0)
            s = torch.sqrt(s2.flip(0))
            Vh = Vh2.t().flip(0)
            U = A @ Vh.t() / (s.unsqueeze(0) + 1e-12)
    V = Vh.t()
    k = min(rank, s.numel())
    X = torch.linalg.solve_triangular(S.t(), V[:, :k], upper=True)
    sqrt_s = torch.sqrt(torch.clamp(s[:k], min=0))
    U_data = X * sqrt_s.unsqueeze(0)
    V_data = sqrt_s.unsqueeze(1) * U[:, :k].t()
    return U_data.to(W.dtype), V_data.to(W.dtype)


def _data_aware_per_head(Wt: torch.Tensor, rank: int, cov_in: torch.Tensor,
                         num_heads: int):
    d_model = Wt.shape[0]
    dh = Wt.shape[1] // num_heads
    Wt3 = Wt.view(d_model, num_heads, dh)
    Us, Vs = [], []
    for h in range(num_heads):
        Uh, Vh = _data_aware_low_rank(Wt3[:, h, :], rank, cov_in)
        Us.append(Uh)
        Vs.append(Vh)
    return torch.stack(Us, 0), torch.stack(Vs, 0)


@torch.no_grad()
def _calibrate_covariances(model, loader, device, encoder_layers, max_batches=4):
    """Calibrate input covariances for BERT/RoBERTa layers (same internal structure)."""
    model.eval()
    num_layers = len(encoder_layers)
    dm = model.config.hidden_size
    d_ff = model.config.intermediate_size

    cov_attn_in = [torch.zeros(dm, dm, dtype=torch.float32, device=device)
                   for _ in range(num_layers)]
    n_attn_in = [0] * num_layers
    cov_attn_out = [torch.zeros(dm, dm, dtype=torch.float32, device=device)
                    for _ in range(num_layers)]
    n_attn_out = [0] * num_layers
    cov_ffn_in = [torch.zeros(dm, dm, dtype=torch.float32, device=device)
                  for _ in range(num_layers)]
    n_ffn_in = [0] * num_layers
    cov_ffn_out = [torch.zeros(d_ff, d_ff, dtype=torch.float32, device=device)
                   for _ in range(num_layers)]
    n_ffn_out = [0] * num_layers

    handles = []

    def _upd(cov_list, n_list, idx, x):
        if x is None:
            return
        x = x.detach()
        N = x.shape[0] * x.shape[1]
        X2d = x.reshape(N, x.shape[-1]).to(device=device, dtype=torch.float32)
        cov_list[idx] += X2d.t() @ X2d
        n_list[idx] += N

    for i, layer in enumerate(encoder_layers):
        handles.append(layer.attention.self.query.register_forward_pre_hook(
            lambda mod, inp, idx=i: _upd(cov_attn_in, n_attn_in, idx, inp[0])))
        handles.append(layer.attention.output.dense.register_forward_pre_hook(
            lambda mod, inp, idx=i: _upd(cov_attn_out, n_attn_out, idx, inp[0])))
        handles.append(layer.intermediate.dense.register_forward_pre_hook(
            lambda mod, inp, idx=i: _upd(cov_ffn_in, n_ffn_in, idx, inp[0])))
        handles.append(layer.output.dense.register_forward_pre_hook(
            lambda mod, inp, idx=i: _upd(cov_ffn_out, n_ffn_out, idx, inp[0])))

    seen = 0
    for batch in loader:
        if seen >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        # Pass all inputs except labels
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        model(**model_inputs)
        seen += 1

    for h in handles:
        h.remove()

    def _fin(cov, n):
        out = []
        for C, cnt in zip(cov, n):
            if cnt == 0:
                out.append(torch.eye(C.shape[0], dtype=torch.float32, device=C.device))
            else:
                Cn = C / float(cnt)
                ridge = 1e-6 * float(Cn.diag().mean().item() + 1.0)
                Cn += ridge * torch.eye(Cn.shape[0], dtype=Cn.dtype, device=Cn.device)
                out.append(Cn)
        return out

    return {
        "cov_attn_in": _fin(cov_attn_in, n_attn_in),
        "cov_attn_out": _fin(cov_attn_out, n_attn_out),
        "cov_ffn_in": _fin(cov_ffn_in, n_ffn_in),
        "cov_ffn_out": _fin(cov_ffn_out, n_ffn_out),
    }


@torch.no_grad()
def _calibrate_covariances_modernbert(model, loader, device, encoder_layers, max_batches=4):
    """Calibrate input covariances for ModernBERT layers (pre-norm, RoPE, GeGLU).

    Hook points (from MODERNBERT_INTEGRATION_PLAN.md):
      attn_in  — output of layer.attn_norm  (pre-norm hidden before attention)
      attn_out — input  to layer.attn.Wo    (attention context before output proj)
      ffn_in   — output of layer.mlp_norm   (pre-norm hidden before FFN)
      ffn_out  — input  to layer.mlp.Wo     (FFN hidden before output proj)

    Returns same structure as _calibrate_covariances():
      {"cov_attn_in": [...], "cov_attn_out": [...], "cov_ffn_in": [...], "cov_ffn_out": [...]}
    """
    model.eval()
    num_layers = len(encoder_layers)
    dm    = model.config.hidden_size
    d_ffn = encoder_layers[0].mlp.Wo.in_features  # ffn intermediate dim

    cov_attn_in  = [torch.zeros(dm,    dm,    dtype=torch.float32, device=device)
                    for _ in range(num_layers)]
    n_attn_in    = [0] * num_layers
    cov_attn_out = [torch.zeros(dm,    dm,    dtype=torch.float32, device=device)
                    for _ in range(num_layers)]
    n_attn_out   = [0] * num_layers
    cov_ffn_in   = [torch.zeros(dm,    dm,    dtype=torch.float32, device=device)
                    for _ in range(num_layers)]
    n_ffn_in     = [0] * num_layers
    cov_ffn_out  = [torch.zeros(d_ffn, d_ffn, dtype=torch.float32, device=device)
                    for _ in range(num_layers)]
    n_ffn_out    = [0] * num_layers

    handles = []

    def _upd(cov_list, n_list, idx, x):
        if x is None:
            return
        x = x.detach()
        N = x.shape[0] * x.shape[1]
        X2d = x.reshape(N, x.shape[-1]).to(device=device, dtype=torch.float32)
        cov_list[idx] += X2d.t() @ X2d
        n_list[idx] += N

    for i, layer in enumerate(encoder_layers):
        # attn_in: output of attn_norm (pre-norm output before attention)
        handles.append(layer.attn_norm.register_forward_hook(
            lambda mod, inp, out, idx=i: _upd(cov_attn_in, n_attn_in, idx, out)))
        # attn_out: input to Wo (context vector before attention output projection)
        handles.append(layer.attn.Wo.register_forward_pre_hook(
            lambda mod, inp, idx=i: _upd(cov_attn_out, n_attn_out, idx, inp[0])))
        # ffn_in: output of mlp_norm (pre-norm output before FFN)
        handles.append(layer.mlp_norm.register_forward_hook(
            lambda mod, inp, out, idx=i: _upd(cov_ffn_in, n_ffn_in, idx, out)))
        # ffn_out: input to Wo_ffn (GeGLU gated hidden before output projection)
        handles.append(layer.mlp.Wo.register_forward_pre_hook(
            lambda mod, inp, idx=i: _upd(cov_ffn_out, n_ffn_out, idx, inp[0])))

    seen = 0
    for batch in loader:
        if seen >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        model(**model_inputs)
        seen += 1

    for h in handles:
        h.remove()

    def _fin(cov, n):
        out = []
        for C, cnt in zip(cov, n):
            if cnt == 0:
                out.append(torch.eye(C.shape[0], dtype=torch.float32, device=C.device))
            else:
                Cn = C / float(cnt)
                ridge = 1e-6 * float(Cn.diag().mean().item() + 1.0)
                Cn += ridge * torch.eye(Cn.shape[0], dtype=Cn.dtype, device=Cn.device)
                out.append(Cn)
        return out

    return {
        "cov_attn_in":  _fin(cov_attn_in,  n_attn_in),
        "cov_attn_out": _fin(cov_attn_out, n_attn_out),
        "cov_ffn_in":   _fin(cov_ffn_in,   n_ffn_in),
        "cov_ffn_out":  _fin(cov_ffn_out,  n_ffn_out),
    }


def _build_drone(model, calib_loader, device, arch, encoder_layers):
    """Return per_head / low_rank callables that use DRONE-style covariance."""
    if arch == "modernbert":
        print("[drone] Calibrating input covariances (ModernBERT) ...")
        covs = _calibrate_covariances_modernbert(model, calib_loader, device, encoder_layers)
    else:
        print("[drone] Calibrating input covariances ...")
        covs = _calibrate_covariances(model, calib_loader, device, encoder_layers)
    H = model.config.num_attention_heads

    # Build closures *per layer* so we can look up the right covariance
    _layer_covs: Dict[int, dict] = {}
    for i in range(len(encoder_layers)):
        _layer_covs[i] = {
            "attn_in": covs["cov_attn_in"][i],
            "attn_out": covs["cov_attn_out"][i],
            "ffn_in": covs["cov_ffn_in"][i],
            "ffn_out": covs["cov_ffn_out"][i],
        }

    return _layer_covs


# ---------- AdaSVD ----------
def _build_adasvd(model, calib_loader, device, budget, arch, backend="naive",
                  max_calib_samples=4000, steps=800, engineering_stable=False,
                  seed=0):
    """Run adaptive rank selection (paper-compliant PaperHN) and return a ranks dict.

    Args:
        calib_loader: Full training DataLoader — NOT truncated by --calib_batches.
                      train_adasvd_ranks() handles its own batch-level shuffle/slice.
        backend: "naive" or "flashsvd"
        max_calib_samples: Max calibration samples (paper: ~4000)
        steps: HN training steps (paper: 800)
        engineering_stable: If True, use alpha_z gate in PaperHN (ablation)
        seed: Random seed for reproducible calibration shuffle
    """
    # Use paper-compliant adasvd_origin implementation
    ada_origin_path = os.path.join(_REPO_ROOT, "src", "encoders", "compression", "adasvd", "adasvd_origin")
    if ada_origin_path not in sys.path:
        sys.path.insert(0, ada_origin_path)

    try:
        from adasvd_wrapper import train_adasvd_ranks
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import from adasvd_origin: {e}\n"
            f"Path: {ada_origin_path}"
        ) from e

    # Train and save ranks
    output_dir = "experiments/ars_out"
    ranks = train_adasvd_ranks(
        model=model,
        calib_loader=calib_loader,
        budget=budget,
        output_dir=output_dir,
        steps=steps,
        max_calib_samples=max_calib_samples,
        engineering_stable=engineering_stable,
        seed=seed,
        device=device,
    )

    # Save ranks_path for later use
    model._adasvd_ranks_path = os.path.join(output_dir, "ranks.json")
    model._adasvd_backend = backend

    model.eval()
    return ranks


# ═══════════════════════════════════════════════════════════════════════════
# 3b) Apply compression to model
# ═══════════════════════════════════════════════════════════════════════════
def compress_model(model, method, rank, budget, scope, loader, device, calib_batches, calib_loader=None, backend="naive",
                   rank_attn=None, rank_ffn=None, rank_wo=None, qkv_mode="per_head",
                   adasvd_full_loader=None, adasvd_max_calib_samples=4000,
                   adasvd_steps=800, adasvd_engineering_stable=False, adasvd_seed=0,
                   attn_mode="einsum"):
    """Replace encoder layers with SVD blocks in-place (BERT/RoBERTa/ModernBERT).

    Args:
        calib_loader: Optional separate DataLoader for calibration. If None, uses loader.
        backend: "naive" or "flashsvd" - for AdaSVD, determines which implementation to use

    Returns:
        model: The compressed model (same object for most methods, reloaded for AdaSVD)
    """
    arch, encoder_layers = _detect_arch(model)
    dm = model.config.hidden_size
    d_ff = getattr(model.config, "intermediate_size", dm)
    H = model.config.num_attention_heads
    dh = dm // H

    # Choose full rank for attention by mode
    full_rank_attn = dh if qkv_mode == "per_head" else dm
    full_rank_ff = dm
    full_rank_wo = dm

    # Pick ranks (fallback to legacy --rank)
    base = rank
    if method != "adasvd":
        if base is None and (rank_attn is None or rank_ffn is None or rank_wo is None):
            raise ValueError("Need --rank OR provide --rank_attn/--rank_ffn/--rank_wo")

        r_attn = rank_attn if rank_attn is not None else base
        r_ff = rank_ffn if rank_ffn is not None else base
        r_wo = rank_wo if rank_wo is not None else base

        # Apply scope + clamp
        rank_attn = min(r_attn, full_rank_attn) if "qkv" in scope else full_rank_attn
        rank_ff = min(r_ff, full_rank_ff) if "ffn" in scope else full_rank_ff
        rank_wo = min(r_wo, full_rank_wo) if "qkv" in scope else full_rank_wo

        print(f"[compress] qkv_mode={qkv_mode} ranks: attn={rank_attn}, ff={rank_ff}, wo={rank_wo}")
    else:
        # For adasvd, rank will be determined during adaptive rank selection
        rank_attn = None
        rank_ff = None
        rank_wo = None

    # Use provided calib_loader or create from evaluation loader
    if calib_loader is None:
        calib_loader = _make_calib_loader(loader, calib_batches)
    else:
        calib_loader = _make_calib_loader(calib_loader, calib_batches)

    def _make_block(layer, per_head_fn, low_rank_fn, r_attn, r_ff, r_wo):
        """Create the correct SVD block + shim for the detected architecture."""
        if arch == "modernbert":
            blk = NaiveModernBertSVDBlock(
                layer, model.config, r_attn, r_ff, r_wo, per_head_fn, low_rank_fn,
            )
            if blk.rotary_emb is None:
                blk.rotary_emb = model.model.rotary_emb
            return ModernBertLayerShim(blk, attention_type=getattr(layer, "attention_type", "global")).to(device).eval()
        else:
            blk = NaiveSVDBlock(layer, r_attn, r_ff, per_head_fn, low_rank_fn, r_wo,
                                qkv_mode=qkv_mode, attn_mode=attn_mode)
            return BertLayerShim(blk).to(device).eval()

    if method == "svd":
        if rank_attn is not None:
            print(f"[svd] Building plain SVD helpers (rank_attn={rank_attn}, "
                  f"rank_ff={rank_ff}, rank_wo={rank_wo}) ...")
        else:
            print(f"[svd] Building plain SVD helpers ...")
        per_head_fn, low_rank_fn = _build_plain_svd_helpers(model)
        for i, layer in enumerate(encoder_layers):
            encoder_layers[i] = _make_block(
                layer, per_head_fn, low_rank_fn, rank_attn, rank_ff, rank_wo,
            )
            del layer  # Explicitly free original layer

    elif method == "fwsvd":
        if rank_attn is not None:
            print(f"[fwsvd] Building Fisher-weighted helpers (rank_attn={rank_attn}, "
                  f"rank_ff={rank_ff}, rank_wo={rank_wo}) ...")
        else:
            print(f"[fwsvd] Building Fisher-weighted helpers ...")
        per_head_fn, low_rank_fn = _build_fwsvd(model, calib_loader, device, arch)
        for i, layer in enumerate(encoder_layers):
            encoder_layers[i] = _make_block(
                layer, per_head_fn, low_rank_fn, rank_attn, rank_ff, rank_wo,
            )
            del layer  # Explicitly free original layer

    elif method == "drone":
        if rank_attn is not None:
            print(f"[drone] Building data-aware helpers (rank_attn={rank_attn}, "
                  f"rank_ff={rank_ff}, rank_wo={rank_wo}) ...")
        else:
            print(f"[drone] Building data-aware helpers ...")
        layer_covs = _build_drone(model, calib_loader, device, arch, encoder_layers)
        H_heads = model.config.num_attention_heads

        for i, layer in enumerate(encoder_layers):
            lc = layer_covs[i]

            if qkv_mode == "per_head":
                # Per-head mode: use per-head covariance
                def _per_head(Wt, r, _cov=lc["attn_in"], _H=H_heads):
                    return _data_aware_per_head(Wt, r, _cov, _H)

                def _low_rank_ffn(W, r, _c_in=lc["ffn_in"], _c_out=lc["ffn_out"],
                                  _c_attn_out=lc["attn_out"], _dm=dm):
                    if W.shape[0] == _dm and W.shape[1] > _dm:
                        return _data_aware_low_rank(W, r, _c_in)
                    elif W.shape[0] > _dm:
                        return _data_aware_low_rank(W, r, _c_out)
                    else:
                        return _data_aware_low_rank(W, r, _c_attn_out)

            else:  # qkv_mode == "full"
                # Full mode: QKV use full-matrix covariance (low_rank_fn handles all)
                _per_head = None  # Won't be used (block's full branch only calls svd_low_rank_fn)

                def _low_rank_ffn(W, r, _cov_attn=lc["attn_in"], _cov_ao=lc["attn_out"],
                                  _cov_ff_in=lc["ffn_in"], _cov_ff_out=lc["ffn_out"], _dm=dm, _d_ff=d_ff):
                    # QKV weights are [dm, dm]
                    if W.shape == (_dm, _dm):
                        return _data_aware_low_rank(W, r, _cov_attn)
                    # Attention output Wo is also [dm, dm], use attn_out cov
                    # (Note: Can't distinguish QKV from Wo by shape alone, but block calls in order)
                    # FFN Wi: [dm, d_ff]
                    if W.shape[0] == _dm and W.shape[1] == _d_ff:
                        return _data_aware_low_rank(W, r, _cov_ff_in)
                    # FFN Wo: [d_ff, dm]
                    if W.shape[0] == _d_ff and W.shape[1] == _dm:
                        return _data_aware_low_rank(W, r, _cov_ff_out)
                    # Fallback (shouldn't happen)
                    return _data_aware_low_rank(W, r, _cov_attn)

            encoder_layers[i] = _make_block(
                layer, _per_head, _low_rank_ffn, rank_attn, rank_ff, rank_wo,
            )
            del layer  # Explicitly free original layer

    elif method == "adasvd":
        if budget is None:
            raise ValueError("--budget is required for method=adasvd")
        print(f"[adasvd] Running adaptive rank selection (budget={budget}) ...")

        # Step 1: Train hypernetwork and generate ranks.json
        # Use adasvd_full_loader if provided (full training split, untruncated).
        # adasvd_full_loader bypasses --calib_batches; train_adasvd_ranks() does
        # its own batch-level shuffle and slices to --adasvd_calib_samples.
        ars_loader = adasvd_full_loader if adasvd_full_loader is not None else calib_loader
        ranks = _build_adasvd(model, ars_loader, device, budget, arch, backend=backend,
                              max_calib_samples=adasvd_max_calib_samples,
                              steps=adasvd_steps,
                              engineering_stable=adasvd_engineering_stable,
                              seed=adasvd_seed)
        vals = [v for v in ranks.values() if v > 0]
        median_rank = sorted(vals)[len(vals) // 2] if vals else rank or 64
        print(f"[adasvd] Median rank from ARS: {median_rank}")

        # Step 2: Reload model to get original Linear layers back
        # (AdaSVD training replaced them with MaskedSVDLinear)
        print(f"[adasvd] Reloading original model for compression...")
        ranks_path = model._adasvd_ranks_path  # Save before reload

        # Get model config before reload
        model_id_or_path = model.config._name_or_path
        num_labels = model.config.num_labels
        problem_type = getattr(model.config, "problem_type", None)

        # Reload model
        from transformers import AutoConfig, AutoModelForSequenceClassification
        config = AutoConfig.from_pretrained(model_id_or_path, num_labels=num_labels, problem_type=problem_type)
        model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path, config=config)
        model = model.to(device).eval()

        # Step 3: Apply compression using paper-compliant adasvd_origin
        ada_origin_path = os.path.join(_REPO_ROOT, "src", "encoders", "compression", "adasvd", "adasvd_origin")
        if ada_origin_path not in sys.path:
            sys.path.insert(0, ada_origin_path)

        from adasvd_wrapper import compress_adasvd_naive, compress_adasvd_flashsvd

        # Block format is determined by qkv_mode AND arch, NOT by --backend.
        # --backend only selects the eval-time forward kernel (naive einsum vs
        # Triton), exactly like svd/fwsvd/drone.
        #
        # BERT/RoBERTa per_head: compress_adasvd_flashsvd → BertLayerShim(NaiveSVDBlock)
        # ModernBERT per_head:  compress_adasvd_modernbert → ModernBertLayerShim(NaiveModernBertSVDBlock)
        # full (any arch):      compress_adasvd_naive → _LowRankLinear (full-matrix)
        if arch == "modernbert":
            if qkv_mode == "per_head":
                from adasvd_wrapper import compress_adasvd_modernbert
                print(f"[adasvd] Compressing with NaiveModernBertSVDBlock/per-head"
                      f" using ranks from {ranks_path}")
                compress_adasvd_modernbert(model, ranks_path, device=device)
            else:  # full
                print(f"[adasvd] Compressing with _LowRankLinear/full-matrix (ModernBERT, naive eval only)"
                      f" using ranks from {ranks_path}")
                compress_adasvd_naive(model, ranks_path, device=device)
        elif qkv_mode == "per_head":
            print(f"[adasvd] Compressing with NaiveSVDBlock/per-head (naive+flashsvd eval compatible)"
                  f" using ranks from {ranks_path}")
            compress_adasvd_flashsvd(model, ranks_path, device=device)
        else:  # full
            print(f"[adasvd] Compressing with _LowRankLinear/full-matrix (naive eval only)"
                  f" using ranks from {ranks_path}")
            compress_adasvd_naive(model, ranks_path, device=device)

        print(f"[adasvd] Model compressed (qkv_mode={qkv_mode}, eval backend={backend})")

        # Return the reloaded and compressed model
        return model

    else:
        raise ValueError(f"Unknown method: {method}")

    # Force garbage collection and defragment GPU memory before return
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Try to reduce fragmentation by moving model to CPU and back
        print("[compress] Defragmenting GPU memory...")
        model_device = next(model.parameters()).device
        model = model.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        model = model.to(model_device)
        print(f"[compress] Memory after defrag: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 4) Evaluation
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_task(model, loader, task, device):
    """Return (metric_name, metric_value). Supports GLUE / SuperGLUE / HANS / ANLI / COPA."""
    cfg = TASK_CFG[task]

    # COPA: two-choice NLI scoring — dedicated evaluation path
    if cfg.get("copa_style", False):
        acc = _evaluate_copa(model, loader, device)
        print(f"[eval] COPA two-choice accuracy: {acc:.4f}")
        return "accuracy", acc

    metric_name = cfg["metric_name"]
    is_regression = cfg.get("is_regression", False)
    ds_name = cfg.get("dataset_name", "glue")
    requires_label_fold = cfg.get("requires_label_fold", False)

    # For regression tasks (STSB), compute metric manually
    if is_regression:
        all_preds = []
        all_labels = []
        model.eval()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Prepare model inputs (exclude labels)
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
            logits = model(**model_inputs).logits

            # For regression, use logits directly (not argmax)
            preds = logits.squeeze(-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

        # Compute pearson correlation
        import numpy as np
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(all_preds, all_labels)
        return metric_name, correlation

    # For classification tasks
    from evaluate import load as load_metric

    # Choose metric loader based on dataset source
    # MVP: use generic accuracy for all non-GLUE tasks (super_glue/hans/anli) for stability
    if ds_name == "glue":
        metric = load_metric("glue", task)
    else:
        # SuperGLUE, HANS, ANLI — generic accuracy (sufficient for compression trend analysis)
        metric = load_metric("accuracy")

    # ── Canonical label remap ────────────────────────────────────────────────
    # Automatically aligns model output class indices with the dataset's label ordering.
    # Uses "canonical_label2id" from task config (the dataset's ground-truth ordering).
    # Example: BoolQ model id2label={0:"True",1:"False"}, dataset 0=False 1=True
    #          → remap [1,0] — flips predictions so accuracy measures the right thing.
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
        _model_name = getattr(model.config, '_name_or_path', '')
        for model_key, remap_list in overrides.items():
            if model_key.lower() in _model_name.lower() or _model_name.lower() in model_key.lower():
                label_remap = torch.tensor(remap_list, dtype=torch.long, device=device)
                break

    # Legacy fallback: MNLI textattack hardcoded remap (in case canonical doesn't catch it)
    if task == "mnli" and label_remap is None:
        model_name = getattr(model.config, '_name_or_path', '')
        if 'textattack' in model_name.lower():
            label_remap = torch.tensor([2, 0, 1], dtype=torch.long, device=device)

    # ── Sanity logging ───────────────────────────────────────────────────────
    _id2label = getattr(model.config, 'id2label', {})
    _num_labels = getattr(model.config, 'num_labels', '?')
    print(f"[eval] id2label={_id2label}  num_labels={_num_labels}")
    if label_remap is not None:
        print(f"[eval] label_remap={label_remap.tolist()}  (canonical={canon})")
    else:
        print(f"[eval] label_remap=None  (labels already canonical or no canonical defined)")
    # Detect entailment class for HANS fold and COPA scoring.
    # textattack MNLI uses generic LABEL_X names → default to class 1 (textattack convention).
    # Models with named id2label (e.g. ModernBERT fine-tuned on GLUE) → look up "entailment".
    _id2label_map = getattr(model.config, 'id2label', {})
    _entail_cls = 1  # textattack MNLI default
    for _cid, _cname in _id2label_map.items():
        if str(_cname).lower() == "entailment":
            _entail_cls = int(_cid)
            break

    if requires_label_fold:
        print(f"[eval] fold_rule: mnli_3→{task}_2  (pred=={_entail_cls}→0=entailment, else→1=non_entailment)")

    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = model(**model_inputs).logits

        preds = torch.argmax(logits, dim=-1)

        if label_remap is not None:
            preds = label_remap[preds]

        # HANS: fold MNLI 3-class → 2-class
        # Entailment class detected from model.config.id2label (_entail_cls).
        # textattack MNLI: 0=contradiction, 1=entailment, 2=neutral → _entail_cls=1
        # ModernBERT MNLI (HuggingFace GLUE order): 0=entailment → _entail_cls=0
        # HANS target: 0=entailment, 1=non_entailment
        if requires_label_fold:
            preds = torch.where(
                preds == _entail_cls,
                torch.zeros_like(preds),
                torch.ones_like(preds)
            )

        metric.add_batch(
            predictions=preds.cpu(), references=batch["labels"].cpu()
        )

    results = metric.compute()
    return metric_name, results.get(metric_name, results.get(list(results.keys())[0]))


# ═══════════════════════════════════════════════════════════════════════════
# 5) Performance measurement
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def measure_performance(model, loader, device, warmup_steps, measure_steps, num_runs=1, full_validation=False):
    """
    Measure model performance with multiple runs.

    Args:
        num_runs: Number of times to run the measurement. If >1, returns median.
        full_validation: If True, traverse entire validation set (like src/profile_*.py).
                        Ignores measure_steps and num_runs, provides single-run full-dataset metrics.

    Returns:
        latency_ms, throughput_sps, peak_mem_mb (median if num_runs > 1, or full-dataset if full_validation)
    """
    model.eval()
    is_cuda = device != "cpu" and torch.cuda.is_available()

    if full_validation:
        # Full validation mode: traverse entire dataset once (like src implementation)
        print(f"[perf] Full validation mode: measuring over entire dataset ...")

        # Warmup with first batch
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        for _ in range(warmup_steps):
            model(**model_inputs)

        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Measure over full dataset
        start = time.perf_counter()
        total_samples = 0
        num_batches = 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            bs = batch["input_ids"].shape[0]
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
            model(**model_inputs)
            total_samples += bs
            num_batches += 1

        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latency_ms = elapsed * 1000.0 / max(num_batches, 1)
        throughput_sps = total_samples / elapsed
        peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if is_cuda else 0.0

        print(f"[perf] Full dataset: {num_batches} batches, {total_samples} samples")
        print(f"[perf] latency={latency_ms:.2f}ms/batch throughput={throughput_sps:.1f}sps mem={peak_mem_mb:.1f}MB")

        return latency_ms, throughput_sps, peak_mem_mb

    # Standard mode: fixed steps with multiple runs
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    model_inputs = {k: v for k, v in batch.items() if k != "labels"}
    bs = batch["input_ids"].shape[0]

    # Collect results from multiple runs
    latencies = []
    throughputs = []
    peak_mems = []

    for run_idx in range(num_runs):
        # warmup
        for _ in range(warmup_steps):
            model(**model_inputs)

        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        for _ in range(measure_steps):
            model(**model_inputs)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latency_ms = elapsed * 1000.0 / measure_steps
        throughput_sps = bs * measure_steps / elapsed
        peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if is_cuda else 0.0

        latencies.append(latency_ms)
        throughputs.append(throughput_sps)
        peak_mems.append(peak_mem_mb)

        if num_runs > 1:
            print(f"    Run {run_idx+1}/{num_runs}: latency={latency_ms:.2f}ms throughput={throughput_sps:.1f}sps mem={peak_mem_mb:.1f}MB")

    # Return median if multiple runs, otherwise single result
    if num_runs > 1:
        import statistics
        latency_ms = statistics.median(latencies)
        throughput_sps = statistics.median(throughputs)
        peak_mem_mb = statistics.median(peak_mems)
        print(f"    Median: latency={latency_ms:.2f}ms throughput={throughput_sps:.1f}sps mem={peak_mem_mb:.1f}MB")

    return latency_ms, throughput_sps, peak_mem_mb


# ═══════════════════════════════════════════════════════════════════════════
# 6) CSV output
# ═══════════════════════════════════════════════════════════════════════════
CSV_FIELDS = [
    "timestamp", "model_id", "task", "dataset_split", "dataset_size",
    "seq_len", "batch_size", "dtype",
    "method", "qkv_mode", "rank", "budget", "scope", "backend", "attn_mode", "seed",
    # Calibration info (for FWSVD/Whiten/Ada methods)
    "calib_dataset", "calib_split", "calib_samples", "calib_batches", "calib_seed", "calib_seq_len",
    # Cross-task display: "mnli:train → hans:validation" — convenience for debugging
    "calib_source",   # e.g. "mnli:train"
    "eval_target",    # e.g. "hans:validation"
    "metric_name", "metric_value",
    "latency_ms", "throughput_sps",
    "peak_mem_infer_mb",  # Peak memory during inference only
    "peak_mem_e2e_mb",    # Peak memory end-to-end (compression + inference)
    "peak_mem_mb",        # Legacy: same as peak_mem_e2e_mb for backward compatibility
    "param_ratio",        # Compression ratio: compressed_params / original_params (affected layers)
    "original_params",    # Original model parameters (affected layers only)
    "compressed_params",  # Compressed model parameters (affected layers only)
    "total_param_ratio",  # Compression ratio for entire model
    "total_original_params",   # Original total model parameters
    "total_compressed_params", # Compressed total model parameters
    "notes", "git_commit",
]


def _count_model_params(model):
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def _calculate_param_ratio(model, method, original_total_params, rank=None, budget=None, scope="qkv+ffn", arch="bert"):  # scope: qkv+ffn or qkv+wo+ffn
    """
    Calculate compression ratio using ACTUAL parameter counts from the model.
    Returns a tuple: (ratio, original_params, compressed_params, total_original, total_compressed)

    Args:
        model: The model (potentially compressed)
        method: Compression method
        original_total_params: Total parameter count of original dense model (before compression)
        rank, budget, scope, arch: Compression configuration

    Returns:
        - ratio: compression ratio for affected layers
        - original_params: affected layers' original size
        - compressed_params: affected layers' compressed size
        - total_original: entire model's original size
        - total_compressed: entire model's actual size

    This function counts real parameters in compressed layers (with U/V matrices)
    and compares to what those layers would have been if kept dense.
    """
    # Use the provided original total parameter count (before compression)
    total_model_params = original_total_params

    if method == "dense":
        return 1.0, total_model_params, total_model_params, total_model_params, total_model_params

    if method == "adasvd":
        # AdaSVD saves budget report with achieved_ratio
        budget_report_path = "experiments/ars_out/budget_report.json"
        if os.path.exists(budget_report_path):
            import json
            with open(budget_report_path) as f:
                report = json.load(f)
                affected_ratio = report["achieved_ratio"]
                affected_original = report["original_model_params"]
                affected_compressed = report["total_params"]
                # Calculate total model size (affected + unaffected)
                unaffected_params = total_model_params - affected_original
                total_compressed = affected_compressed + unaffected_params
                return affected_ratio, affected_original, affected_compressed, total_model_params, total_compressed

    # Get encoder layers
    _, encoder_layers = _detect_arch(model)

    total_original = 0  # What it would be if dense
    total_compressed = 0  # What it actually is now

    for layer in encoder_layers:
        # Check if this layer has SVD blocks (has a 'block' attribute)
        if hasattr(layer, 'block'):
            block = layer.block

            # Count actual compressed parameters in this block
            for name, param in block.named_parameters():
                if param.requires_grad:
                    total_compressed += param.numel()

            # Estimate original dense size from model config
            hidden = model.config.hidden_size
            intermediate = getattr(model.config, "intermediate_size", hidden * 4)

            # QKV + Wo (attention)
            if "qkv" in scope:
                total_original += 4 * hidden * hidden  # Q, K, V, Wo
            else:
                # If QKV not compressed, add its actual size
                total_original += 4 * hidden * hidden

            # FFN
            if "ffn" in scope:
                total_original += hidden * intermediate + intermediate * hidden
            else:
                total_original += hidden * intermediate + intermediate * hidden
        else:
            # Dense layer - count actual parameters
            layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            total_original += layer_params
            total_compressed += layer_params

    # Calculate affected layers ratio
    ratio = total_compressed / total_original if total_original > 0 else 1.0

    # Calculate total model parameters
    # total_original = affected layers' original size
    # total_compressed = affected layers' compressed size
    # unaffected_params = total_model_params - total_original
    unaffected_params = total_model_params - total_original
    total_model_original = total_model_params  # Entire model if dense
    total_model_compressed = total_compressed + unaffected_params  # Compressed + unaffected

    return ratio, total_original, total_compressed, total_model_original, total_model_compressed


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _log_sdpa_backends(dtype_str: str = "bf16", seq_len: int = 512) -> str:
    """
    Log which PyTorch SDPA backends are enabled and which one will likely be chosen.

    This is CRITICAL for reproducibility: on different machines or PyTorch versions,
    F.scaled_dot_product_attention may dispatch to:
      - FlashAttention  (fastest; requires GPU + fp16/bf16 + specific head_dim)
      - Memory-Efficient (moderate; requires GPU + fp16/bf16/fp32)
      - Math             (slowest; always available; fp32-safe)

    Without recording this, a "sdpa" run on a V100 (no FlashAttention) vs A100
    may show very different throughput numbers while reporting the same backend.

    Returns:
        str: Detected path — "flash", "mem_efficient", "math", or "unknown".
    """
    if not torch.cuda.is_available():
        print("[sdpa] CUDA not available — SDPA will use CPU math backend")
        return "unknown"

    has_api = (
        hasattr(torch.backends, "cuda") and
        hasattr(torch.backends.cuda, "flash_sdp_enabled")
    )
    if not has_api:
        print("[sdpa] torch.backends.cuda SDPA query API not available (PyTorch < 2.0?)")
        return "unknown"

    flash   = torch.backends.cuda.flash_sdp_enabled()
    mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
    math    = torch.backends.cuda.math_sdp_enabled()

    # Heuristic: flash requires fp16/bf16 and head_dim in {16,32,64,128}
    is_half = dtype_str in ("fp16", "bf16")
    gpu_name = torch.cuda.get_device_name(0).lower()
    # Flash attention not supported on Pascal (GTX 10xx, P100) and older
    no_flash_gpus = ("v100", "p100", "gtx 10", "rtx 20", "2060", "2070", "2080")
    gpu_likely_flash = flash and is_half and not any(k in gpu_name for k in no_flash_gpus)

    if gpu_likely_flash:
        likely = "flash"
    elif mem_eff and is_half:
        likely = "mem_efficient"
    else:
        likely = "math"

    print(f"[sdpa] Enabled backends: flash={flash}, mem_efficient={mem_eff}, math={math}")
    print(f"[sdpa] dtype={dtype_str}  GPU={torch.cuda.get_device_name(0)}")
    print(f"[sdpa] Likely path: {likely.upper()}"
          f"  ← performance varies across machines; record this for reproducibility")
    return likely


def write_csv_row(args, metric_name, metric_value,
                  latency_ms, throughput_sps, peak_mem_infer_mb, peak_mem_e2e_mb,
                  param_ratio, original_params, compressed_params,
                  total_ratio, total_original_params, total_compressed_params,
                  dataset_info, calib_info):
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    csv_exists = os.path.isfile(args.out_csv)
    write_header = not csv_exists

    # Check if existing CSV has correct header (with total_* fields)
    if csv_exists:
        try:
            with open(args.out_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames
                # Check if new fields are present
                missing_fields = [f for f in ["total_param_ratio", "total_original_params", "total_compressed_params", "attn_mode"]
                                 if f not in existing_fields]
                if missing_fields:
                    # Backup old CSV and rewrite with new header
                    backup_path = args.out_csv.replace('.csv', '_backup_pre_v2.csv')
                    import shutil
                    shutil.copy2(args.out_csv, backup_path)
                    print(f"[csv] ⚠️  Old CSV header detected (missing: {missing_fields})")
                    print(f"[csv] 📦 Backed up to: {backup_path}")
                    print(f"[csv] 🔧 Adding new columns to CSV...")

                    # Read all existing rows
                    with open(args.out_csv, 'r', encoding='utf-8') as old_f:
                        reader = csv.DictReader(old_f)
                        old_rows = list(reader)

                    # Write with new header
                    with open(args.out_csv, 'w', encoding='utf-8', newline='') as new_f:
                        writer = csv.DictWriter(new_f, fieldnames=CSV_FIELDS)
                        writer.writeheader()
                        # Write old rows (missing fields will be empty)
                        for old_row in old_rows:
                            old_row.pop(None, None)  # trailing-comma rows get None key
                            # Fill missing fields with empty values
                            for field in CSV_FIELDS:
                                if field not in old_row:
                                    old_row[field] = ""
                            writer.writerow(old_row)
                    print(f"[csv] ✅ CSV updated with new header")
        except Exception as e:
            print(f"[csv] ⚠️  Could not check CSV header: {e}")

    # Determine if method needs calibration
    needs_calib = args.method in ["fwsvd", "drone", "adasvd", "adawhiten", "adafwsvd"]

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "task": args.task,
        "dataset_split": dataset_info.get("split", ""),
        "dataset_size": dataset_info.get("size", ""),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "method": args.method,
        "qkv_mode": getattr(args, 'qkv_mode', 'per_head'),
        "rank": (
            f"ra{args.rank_attn if args.rank_attn is not None else args.rank}"
            f"_rf{args.rank_ffn if args.rank_ffn is not None else args.rank}"
            f"_rw{args.rank_wo if args.rank_wo is not None else args.rank}"
            if (args.rank_attn is not None or args.rank_ffn is not None or args.rank_wo is not None)
            else (args.rank if args.rank is not None else "")
        ),
        "budget": args.budget if args.budget is not None else "",
        "scope": args.scope,
        "backend": args.backend,
        "attn_mode": getattr(args, 'attn_mode', 'einsum'),
        "seed": args.seed,
        # Calibration info (only for methods that need it)
        "calib_dataset": calib_info.get("dataset", "") if needs_calib else "",
        "calib_split": calib_info.get("split", "") if needs_calib else "",
        "calib_samples": calib_info.get("samples", "") if needs_calib else "",
        "calib_batches": calib_info.get("batches", "") if needs_calib else "",
        "calib_seed": calib_info.get("seed", "") if needs_calib else "",
        "calib_seq_len": calib_info.get("seq_len", "") if needs_calib else "",
        # Cross-task display fields (e.g. "mnli:train → hans:validation")
        "calib_source": (f"{calib_info.get('dataset','')}:{calib_info.get('split','')}"
                         if needs_calib and calib_info else ""),
        "eval_target": f"{args.task}:{dataset_info.get('split', '')}",
        "metric_name": metric_name,
        "metric_value": f"{metric_value:.6f}",
        "latency_ms": f"{latency_ms:.2f}",
        "throughput_sps": f"{throughput_sps:.1f}",
        "peak_mem_infer_mb": f"{peak_mem_infer_mb:.1f}",
        "peak_mem_e2e_mb": f"{peak_mem_e2e_mb:.1f}",
        "peak_mem_mb": f"{peak_mem_e2e_mb:.1f}",  # Legacy field for backward compatibility
        # Affected layers parameters
        "param_ratio": f"{param_ratio:.4f}",
        "original_params": original_params,
        "compressed_params": compressed_params,
        # Total model parameters
        "total_param_ratio": f"{total_ratio:.4f}",
        "total_original_params": total_original_params,
        "total_compressed_params": total_compressed_params,
        "notes": args.notes,
        "git_commit": _git_commit(),
    }

    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)

    if not csv_exists:
        print(f"\n[csv] ✅ Created new CSV: {args.out_csv}")
        print(f"[csv] Header written with {len(CSV_FIELDS)} columns")
    else:
        print(f"\n[csv] ✅ Appended row to existing CSV: {args.out_csv}")

    # Show total rows in CSV
    try:
        with open(args.out_csv, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # Exclude header
        print(f"[csv] Total data rows: {total_rows}")
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"=== Encoder Benchmark ===")
    print(f"  model_id : {args.model_id}")
    print(f"  task     : {args.task}")

    # Show rank information (either unified or component-specific)
    rank_info = f"rank={args.rank}"
    if args.rank_attn is not None or args.rank_ffn is not None or args.rank_wo is not None:
        rank_details = []
        if args.rank_attn is not None:
            rank_details.append(f"attn={args.rank_attn}")
        if args.rank_ffn is not None:
            rank_details.append(f"ffn={args.rank_ffn}")
        if args.rank_wo is not None:
            rank_details.append(f"wo={args.rank_wo}")
        rank_info = f"rank=[{', '.join(rank_details)}]" if rank_details else rank_info

    print(f"  method   : {args.method}  {rank_info}  budget={args.budget}  scope={args.scope}")
    print(f"  backend  : {args.backend}")
    if args.qkv_mode == "full":
        print(f"  qkv_mode : {args.qkv_mode} (paper-style full-matrix SVD)")
        if args.backend in ("flashsvd", "flashsvd15"):
            raise ValueError(
                f"--qkv_mode full is not compatible with --backend {args.backend}. "
                "FlashSVD kernels require per-head format (use --qkv_mode per_head)."
            )
    print(f"  dtype    : {args.dtype}  device={args.device}")
    if args.calib_task and args.calib_task != args.task:
        print(f"  calib_task: {args.calib_task}  (cross-task calibration: eval={args.task})")
    print()

    # 1) load model
    model, tokenizer = load_model(args.model_id, args.task, args.dtype, args.device)
    arch, _ = _detect_arch(model)
    # ModernBERT always compresses Wo_attn (qkv+wo+ffn); auto-correct default scope
    if arch == "modernbert" and args.scope == "qkv+ffn":
        args.scope = "qkv+wo+ffn"
    # Save original total parameter count BEFORE compression (for accurate total_model_params calculation)
    original_total_params = _count_model_params(model)
    print(f"[load] Model loaded: {original_total_params/1e6:.1f}M params  arch={arch}")

    # 2) data
    loader = prepare_loader(args.task, tokenizer, args.seq_len, args.batch_size, split="validation")
    cfg = TASK_CFG[args.task]
    dataset_info = {
        "split": cfg["val_split"],
        "size": len(loader.dataset),
    }
    print(f"[data] Validation: {len(loader.dataset)} samples, {len(loader)} batches (split={cfg['val_split']})")

    # 2b) calibration data (for methods that need it)
    # IMPORTANT: Calibration MUST use training data, NOT validation!
    # --calib_task allows calibrating on a different task than the one being evaluated
    # (e.g. --task hans --calib_task mnli: evaluate on HANS, calibrate on MNLI train)
    calib_loader = None
    calib_info = {}
    calib_seed = args.calib_seed if args.calib_seed is not None else args.seed

    # Resolve calibration task (defaults to eval task)
    calib_task = args.calib_task or args.task
    calib_cfg = TASK_CFG[calib_task]

    if getattr(args, 'load_model_dir', None):
        # When loading a locally pre-compressed model, the compression step is skipped and calibration data is not needed at all
        pass
    elif args.method in ["fwsvd", "drone", "adasvd", "adawhiten", "adafwsvd"]:
        if calib_cfg.get("train_split") is None:
            raise ValueError(
                f"Calibration task '{calib_task}' has no train split and cannot be used for "
                f"calibration-based methods (fwsvd/drone/adasvd). "
                + (f"Use --calib_task <task_with_train_split> (e.g. --calib_task mnli)."
                   if calib_task == args.task else "")
            )
        if args.calib_split != "train":
            raise ValueError(
                f"ERROR: Calibration must use training data! "
                f"Got calib_split={args.calib_split}, but only 'train' is allowed. "
                f"Using validation data for calibration causes information leakage."
            )

        # Set seed for calibration data sampling
        torch.manual_seed(calib_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(calib_seed)

        calib_loader = prepare_loader(calib_task, tokenizer, args.seq_len, args.batch_size, split=args.calib_split)
        calib_split_name = calib_cfg["train_split"]
        actual_calib_samples = min(args.calib_batches * args.batch_size, len(calib_loader.dataset))

        calib_info = {
            "dataset": calib_task,
            "split": calib_split_name,
            "samples": actual_calib_samples,
            "batches": args.calib_batches,
            "seed": calib_seed,
            "seq_len": args.seq_len,
        }

        if calib_task != args.task:
            print(f"[calib] Cross-task calibration: eval={args.task}, calib={calib_task}")
        print(f"[calib] Calibration data:")
        print(f"        dataset={calib_task}, split={calib_split_name}")
        print(f"        samples={actual_calib_samples} ({args.calib_batches} batches × {args.batch_size})")
        print(f"        seed={calib_seed}, seq_len={args.seq_len}")

    # 3) compress
    # For AdaSVD: build a separate full loader (untruncated by --calib_batches).
    # train_adasvd_ranks() does its own batch-level shuffle and slices to
    # --adasvd_calib_samples, so this loader must cover the full training split.
    adasvd_full_loader = None
    if args.method == "adasvd" and calib_loader is not None:
        # Guard: ARS uses task loss with calibration labels.  If calib_task has more
        # classes than the model, labels will be out of range → CUDA assert in nll_loss.
        # E.g. --calib_task mnli (3-class) + binary model → label=2 with n_classes=2.
        _calib_nl = TASK_CFG[calib_task].get("num_labels")
        _model_nl = getattr(model.config, "num_labels", None)
        _calib_is_reg = TASK_CFG[calib_task].get("is_regression", False)
        if (_calib_nl is not None and _model_nl is not None
                and not _calib_is_reg and _calib_nl > _model_nl):
            raise ValueError(
                f"[adasvd] Calibration task '{calib_task}' has {_calib_nl} label classes "
                f"but model only has {_model_nl}. ARS task loss would receive out-of-range "
                f"labels (e.g. label=2 when n_classes=2) → CUDA assert.\n"
                f"Fix: remove --calib_task (uses eval task's own data) or pick a calib task "
                f"with num_labels <= {_model_nl}."
            )
        adasvd_full_loader = prepare_loader(
            calib_task, tokenizer, args.seq_len, args.batch_size,
            split=args.calib_split,
        )

    # Reset peak stats before compression so compression_peak_mb only reflects
    # the compression phase, not the entire program history up to this point.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    compression_peak_mb = 0.0
    if getattr(args, 'load_model_dir', None):
        # --- Load pre-compressed model from disk, skip the compression step ---
        from src.encoders.io import load_compressed_model as _load_compressed
        print(f"\n[load_model_dir] Loading pre-compressed model from: {args.load_model_dir}")
        pt_dtype = DTYPE_MAP[args.dtype]
        model, tokenizer, _comp_info = _load_compressed(
            args.load_model_dir, device=args.device, dtype=pt_dtype
        )
        arch, encoder_layers = _detect_arch(model)
        # Override the default bert-base-uncased with model_id from comp_info to ensure correct CSV and display
        if _comp_info.get('model_id'):
            args.model_id = _comp_info['model_id']
        # Backfill calib_info so that CSV rows from the load path are consistent with those from compression time
        if _comp_info.get('calib_dataset'):
            calib_info = {
                "dataset":  _comp_info.get('calib_dataset', ''),
                "split":    _comp_info.get('calib_split', ''),
                "samples":  _comp_info.get('calib_samples', ''),
                "batches":  _comp_info.get('calib_batches', ''),
                "seed":     _comp_info.get('calib_seed', ''),
                "seq_len":  _comp_info.get('calib_seq_len', ''),
            }
        print(f"[load_model_dir] Loaded: arch={arch}, {len(encoder_layers)} layers")
        # If attn_mode=sdpa, patch all NaiveSVDBlocks
        if getattr(args, 'attn_mode', 'einsum') != 'einsum':
            patched = 0
            for layer in encoder_layers:
                blk = getattr(layer, 'block', None)
                if blk is not None and hasattr(blk, 'attn_mode'):
                    blk.attn_mode = args.attn_mode
                    patched += 1
            print(f"[load_model_dir] Patched attn_mode={args.attn_mode} on {patched} blocks")
    elif args.method != "dense":
        model = compress_model(model, args.method, args.rank, args.budget,
                               args.scope, loader, args.device, args.calib_batches,
                               calib_loader=calib_loader, backend=args.backend,
                               rank_attn=args.rank_attn, rank_ffn=args.rank_ffn,
                               rank_wo=args.rank_wo, qkv_mode=args.qkv_mode,
                               adasvd_full_loader=adasvd_full_loader,
                               adasvd_max_calib_samples=getattr(args, 'adasvd_calib_samples', 4000),
                               adasvd_steps=getattr(args, 'adasvd_steps', 800),
                               adasvd_engineering_stable=getattr(args, 'adasvd_engineering_stable', False),
                               adasvd_seed=calib_seed,
                               attn_mode=getattr(args, 'attn_mode', 'einsum'))

        # Capture peak memory during compression (includes calibration, SVD, etc.)
        if torch.cuda.is_available():
            compression_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"[compress] Peak memory during compression: {compression_peak_mb:.1f} MB")

            # Clean up memory before inference measurement
            print(f"[cleanup] Freeing calibration data and compressing cache...")
            del calib_loader
            calib_loader = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Check memory after cleanup
            after_cleanup_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"[cleanup] Memory allocated after cleanup: {after_cleanup_mb:.1f} MB")

    # 4) backend
    if args.backend in ("flashsvd", "flashsvd15"):
        if args.method == "dense":
            print(f"[backend] Warning: backend={args.backend} ignored for method=dense (no compression)")
        else:
            if args.backend == "flashsvd":
                from src.encoders.backend import enable_flashsvd
                enable_flashsvd(model)
            else:
                from src.encoders.backend import enable_flashsvd15
                enable_flashsvd15(model)

    # 4b) SDPA path detection & logging
    # Records which PyTorch SDPA backend (flash/mem_efficient/math) will be used.
    # The actual path depends on: enabled flags + dtype + seq_len + head_dim + GPU.
    # Without this log, SDPA runs on different machines may silently use different kernels.
    # The detected path is appended to args.notes so it appears in the CSV "notes" field.
    if getattr(args, 'attn_mode', 'einsum') == 'sdpa' and torch.cuda.is_available():
        sdpa_path = _log_sdpa_backends(args.dtype, args.seq_len)
        if sdpa_path:
            prefix = f"sdpa_path={sdpa_path}"
            args.notes = f"{prefix} {args.notes}".strip() if args.notes else prefix
        if sdpa_path == "unknown":
            print("[sdpa] ⚠️  WARNING: SDPA backend path could not be determined.")
            print("[sdpa]    Rows with sdpa_path=unknown should NOT be included in SDPA")
            print("[sdpa]    equivalence statistics — different machines may use different kernels.")

    # 5) evaluate task metric
    if getattr(args, 'skip_eval', False):
        print("\n[eval] Skipped (--skip_eval)")
        metric_name, metric_value = "skipped", 0.0
    else:
        print("\n[eval] Computing task metric ...")
        metric_name, metric_value = evaluate_task(model, loader, args.task, args.device)
        print(f"[eval] {metric_name} = {metric_value:.4f}")

    # 5.3) Save model if requested (including dense)
    if args.save_model:
        from pathlib import Path
        import json

        # Generate model name
        # NOTE: Must match glue_pipeline.py's naming convention exactly
        if args.method == "dense":
            model_name = "dense_naive"
        elif args.method == "adasvd":
            model_name = f"{args.method}_b{args.budget}_{args.qkv_mode}_{args.backend}"
        else:
            # For SVD-based methods, include rank and qkv_mode info in name
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

        save_path = Path(args.save_dir) / model_name
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[save] Saving model to {save_path}")

        # Save model and tokenizer
        # Use safe_serialization=False for FlashSVD blocks with shared tensors
        model.save_pretrained(save_path, safe_serialization=False)
        tokenizer.save_pretrained(save_path)

        # Save compression info
        info = {
            "method": args.method,
            "rank": args.rank if args.method != "adasvd" else None,
            "budget": args.budget if args.method == "adasvd" else None,
            "backend": args.backend,
            "task": args.task,
            "model_id": args.model_id,
            "accuracy_before_finetune": float(metric_value),
            "dtype": args.dtype,
            "seq_len": args.seq_len,
            # calib fields — needed to back-fill CSV when loading from checkpoint
            "calib_dataset": calib_info.get("dataset", ""),
            "calib_split":   calib_info.get("split", ""),
            "calib_samples": calib_info.get("samples", ""),
            "calib_batches": calib_info.get("batches", ""),
            "calib_seed":    calib_info.get("seed", ""),
            "calib_seq_len": calib_info.get("seq_len", ""),
        }

        with open(save_path / "compression_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"[save] Model saved successfully")
        print(f"[save] Accuracy before fine-tuning: {metric_value:.4f}")

    # 5.5) Reload model before performance measurement if requested
    # This ensures clean GPU state after calibration backward passes
    if args.reload_before_perf and args.method != "dense":
        print("\n[reload] Cleaning GPU state before performance measurement ...")

        # Move model to CPU to free GPU memory
        model = model.cpu()

        # Clear GPU cache thoroughly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        print(f"[reload] Cleared GPU cache and fragmentation")

        # Move model back to GPU with fresh allocation
        pt_dtype = DTYPE_MAP[args.dtype]
        model = model.to(device=args.device, dtype=pt_dtype).eval()

        # Re-apply backend if needed
        # Note: FlashSVD patches should already be in place after CPU/GPU transfer
        # Only re-enable if we can detect that patches were lost
        if args.backend == "flashsvd":
            # Check if FlashSVD is already enabled by looking for FlashSVDBlock
            arch, encoder_layers = _detect_arch(model)
            has_flashsvd = False
            for layer in encoder_layers:
                if hasattr(layer, 'block'):
                    block_type = type(layer.block).__name__
                    if 'Flash' in block_type:
                        has_flashsvd = True
                        break

            if not has_flashsvd:
                # FlashSVD was not applied yet or was lost, reapply
                from src.encoders.backend import enable_flashsvd
                enable_flashsvd(model)
                print(f"[reload] Re-enabled FlashSVD backend")
            else:
                print(f"[reload] FlashSVD backend already active (preserved through CPU/GPU transfer)")

        print(f"[reload] Model moved back to GPU with clean memory state")

    # 6) measure performance
    if args.full_validation:
        print(f"\n[perf] Full validation mode (Warmup={args.warmup_steps} steps, then full dataset) ...")
    elif args.num_runs > 1:
        print(f"\n[perf] Running {args.num_runs} times with Warmup={args.warmup_steps} Measure={args.measure_steps} steps ...")
    else:
        print(f"\n[perf] Warmup={args.warmup_steps}  Measure={args.measure_steps} steps ...")

    latency_ms, throughput_sps, peak_mem_mb = measure_performance(
        model, loader, args.device, args.warmup_steps, args.measure_steps, args.num_runs,
        full_validation=args.full_validation,
    )

    # Calculate overall peak (max of compression and inference)
    overall_peak_mb = max(compression_peak_mb, peak_mem_mb)

    # When compression dominates e2e peak, annotate the notes field so readers know
    # peak_mem_e2e_mb reflects compression (e.g. AdaSVD ARS training), not inference.
    # Threshold: compression > 1.5× inference peak.
    if compression_peak_mb > peak_mem_mb * 1.5:
        _note = (f"peak_e2e_includes_compression={compression_peak_mb:.0f}MB"
                 f";infer_only={peak_mem_mb:.0f}MB")
        args.notes = f"{args.notes} {_note}".strip() if args.notes else _note

    # Print detailed memory breakdown
    print(f"\n{'='*60}")
    print(f"Memory Usage Summary:")
    print(f"  Compression phase: {compression_peak_mb:>8.1f} MB")
    print(f"  Inference phase:   {peak_mem_mb:>8.1f} MB")
    print(f"  Overall peak:      {overall_peak_mb:>8.1f} MB")
    print(f"{'='*60}")

    if not args.full_validation:
        if args.num_runs > 1:
            print(f"[perf] MEDIAN: latency={latency_ms:.2f} ms/batch  "
                  f"throughput={throughput_sps:.1f} samples/s  "
                  f"peak_mem={overall_peak_mb:.1f} MB")
        else:
            print(f"[perf] latency={latency_ms:.2f} ms/batch  "
                  f"throughput={throughput_sps:.1f} samples/s  "
                  f"peak_mem={overall_peak_mb:.1f} MB")

    # 7) Calculate and display parameter compression ratio
    param_ratio, original_params, compressed_params, total_original, total_compressed = _calculate_param_ratio(
        model, args.method, original_total_params, args.rank, args.budget, args.scope, arch
    )
    total_ratio = total_compressed / total_original if total_original > 0 else 1.0

    print(f"\n{'='*70}")
    print(f"[param] Parameter Statistics:")
    print(f"  Affected Layers:")
    print(f"    Original:   {original_params:,} params")
    print(f"    Compressed: {compressed_params:,} params")
    print(f"    Ratio: {param_ratio:.4f} ({param_ratio*100:.2f}%)")
    print(f"    Reduction: {(1-param_ratio)*100:.2f}% fewer parameters")
    print(f"")
    print(f"  Entire Model:")
    print(f"    Original:   {total_original:,} params")
    print(f"    Compressed: {total_compressed:,} params")
    print(f"    Ratio: {total_ratio:.4f} ({total_ratio*100:.2f}%)")
    print(f"    Reduction: {(1-total_ratio)*100:.2f}% fewer parameters")
    print(f"{'='*70}\n")

    # 8) write CSV (with both inference and E2E peaks)
    write_csv_row(args, metric_name, metric_value,
                  latency_ms, throughput_sps, peak_mem_mb, overall_peak_mb,
                  param_ratio, original_params, compressed_params,
                  total_ratio, total_original, total_compressed,
                  dataset_info, calib_info)


if __name__ == "__main__":
    main()
