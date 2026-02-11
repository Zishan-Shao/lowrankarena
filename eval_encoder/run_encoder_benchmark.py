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
# Ensure repo root is on PYTHONPATH so we can import utils.encoder_utils.*
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval_encoder.blocks import (
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
    p.add_argument("--task", choices=["sst2", "mnli"], default="sst2")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    # compression
    p.add_argument("--method", choices=["dense", "svd", "fwsvd", "drone", "adasvd"], default="dense")
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--scope", choices=["qkv", "ffn", "qkv+ffn"], default="qkv+ffn")
    # backend
    p.add_argument("--backend", choices=["naive", "flashsvd"], default="naive")
    # logging / perf
    p.add_argument("--out_csv", default="eval_results/encoder_runs.csv")
    p.add_argument("--notes", default="")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--measure_steps", type=int, default=50)
    p.add_argument("--num_runs", type=int, default=1,
                   help="Number of times to run performance measurement (median will be reported)")
    p.add_argument("--full_validation", action="store_true",
                   help="Measure performance over full validation set (like src/profile_*.py). "
                        "When enabled, ignores --measure_steps and --num_runs, traverses entire dataset once. "
                        "Provides more realistic end-to-end performance but slower to measure.")
    p.add_argument("--reload_before_perf", action="store_true",
                   help="Save compressed model and reload before performance measurement. "
                        "Recommended for methods with calibration (fwsvd/drone/adasvd) to avoid "
                        "GPU memory fragmentation and cache pollution from backward passes.")
    # calibration (for fwsvd / drone)
    p.add_argument("--calib_batches", type=int, default=4,
                   help="Number of batches used for Fisher / covariance calibration")
    p.add_argument("--calib_split", choices=["train"], default="train",
                   help="Which split to use for calibration (MUST be train, NOT validation)")
    p.add_argument("--calib_seed", type=int, default=None,
                   help="Random seed for calibration data sampling (default: same as --seed)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Task configuration
# ═══════════════════════════════════════════════════════════════════════════
TASK_CFG = {
    "sst2": dict(
        num_labels=2, val_split="validation", train_split="train",
        sentence_keys=("sentence",), metric_name="accuracy",
    ),
    "mnli": dict(
        num_labels=3, val_split="validation_matched", train_split="train",
        sentence_keys=("premise", "hypothesis"), metric_name="accuracy",
    ),
}

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# ═══════════════════════════════════════════════════════════════════════════
# 1) Model loading
# ═══════════════════════════════════════════════════════════════════════════
def load_model(model_id: str, task: str, dtype_str: str, device: str):
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

    cfg = TASK_CFG[task]
    config = AutoConfig.from_pretrained(
        model_id, num_labels=cfg["num_labels"], trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, config=config, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

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
def prepare_loader(task: str, tokenizer, seq_len: int, batch_size: int, split: str = "validation"):
    """
    Prepare DataLoader for a specific task and split.

    Args:
        task: Task name (e.g., "sst2", "mnli")
        tokenizer: HuggingFace tokenizer
        seq_len: Maximum sequence length
        batch_size: Batch size
        split: Which split to load ("train" or "validation")
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    cfg = TASK_CFG[task]
    if split == "train":
        split_name = cfg["train_split"]
    else:
        split_name = cfg["val_split"]

    raw = load_dataset("glue", task, split=split_name)

    keys = cfg["sentence_keys"]

    def tok_fn(batch):
        if len(keys) == 1:
            return tokenizer(batch[keys[0]], padding="max_length",
                             truncation=True, max_length=seq_len)
        return tokenizer(batch[keys[0]], batch[keys[1]], padding="max_length",
                         truncation=True, max_length=seq_len)

    remove = [c for c in raw.column_names if c != "label"]
    ds = raw.map(tok_fn, batched=True, remove_columns=remove)
    ds.set_format("torch")

    def collate(batch):
        return {
            "input_ids":      torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels":         torch.tensor([x["label"] for x in batch]),
        }

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
        raise RuntimeError(
            "FWSVD is not yet supported for ModernBERT. "
            "Use --method dense, --method svd, or a BERT/RoBERTa model."
        )
    try:
        from utils.encoder_utils.svd_helpers import build_fwsvd_helpers
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
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
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
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
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


def _build_drone(model, calib_loader, device, arch, encoder_layers):
    """Return per_head / low_rank callables that use DRONE-style covariance."""
    if arch == "modernbert":
        raise RuntimeError(
            "DRONE is not yet supported for ModernBERT. "
            "Use --method dense, --method svd, or a BERT/RoBERTa model."
        )
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
def _build_adasvd(model, calib_loader, device, budget, arch, backend="naive"):
    """Run adaptive rank selection and return a ranks dict.

    Args:
        backend: "naive" or "flashsvd" - affects which implementation to use
    """
    if arch == "modernbert":
        raise RuntimeError(
            "AdaSVD is not yet supported for ModernBERT. "
            "Use --method dense, --method svd, or a BERT/RoBERTa model."
        )

    # Use refactored AdaSVD implementation
    ada_refactored_path = os.path.join(_REPO_ROOT, "src", "encoders", "adasvd_refactored")
    if ada_refactored_path not in sys.path:
        sys.path.insert(0, ada_refactored_path)

    try:
        from adasvd_wrapper import train_adasvd_ranks
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import from adasvd_refactored: {e}\n"
            f"Path: {ada_refactored_path}"
        ) from e

    # Train and save ranks
    output_dir = "ars_out"
    ranks = train_adasvd_ranks(
        model=model,
        calib_loader=calib_loader,
        budget=budget,
        output_dir=output_dir,
        steps=400,
        device=device
    )

    # Save ranks_path for later use
    model._adasvd_ranks_path = os.path.join(output_dir, "ranks.json")
    model._adasvd_backend = backend

    model.eval()
    return ranks


# ═══════════════════════════════════════════════════════════════════════════
# 3b) Apply compression to model
# ═══════════════════════════════════════════════════════════════════════════
def compress_model(model, method, rank, budget, scope, loader, device, calib_batches, calib_loader=None, backend="naive"):
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

    # Effective ranks based on scope
    full_rank_attn = dh       # per-head full rank = min(dm, dh) = dh
    full_rank_ff = dm         # full rank for FFN = min(dm, d_ff) = dm
    full_rank_wo = dm         # full rank for attn output

    if rank is None and method != "adasvd":
        raise ValueError(f"--rank is required for method={method}")

    # For adasvd, rank will be determined during adaptive rank selection
    if method != "adasvd":
        rank_attn = min(rank, full_rank_attn) if "qkv" in scope else full_rank_attn
        rank_ff = min(rank, full_rank_ff) if "ffn" in scope else full_rank_ff
        rank_wo = min(rank, full_rank_wo) if "qkv" in scope else full_rank_wo
    else:
        # Placeholder values for adasvd, will be overridden later
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
                layer, model.config, r_attn, r_ff, per_head_fn, low_rank_fn,
            )
            return ModernBertLayerShim(blk).to(device).eval()
        else:
            blk = NaiveSVDBlock(layer, r_attn, r_ff, per_head_fn, low_rank_fn, r_wo)
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

            def _per_head(Wt, r, _cov=lc["attn_in"], _H=H_heads):
                return _data_aware_per_head(Wt, r, _cov, _H)

            def _low_rank_attn_out(W, r, _cov=lc["attn_out"]):
                return _data_aware_low_rank(W, r, _cov)

            ffn_in_cov = lc["ffn_in"]
            ffn_out_cov = lc["ffn_out"]

            def _low_rank_ffn(W, r, _c_in=ffn_in_cov, _c_out=ffn_out_cov,
                              _c_attn_out=lc["attn_out"], _dm=dm):
                if W.shape[0] == _dm and W.shape[1] > _dm:
                    return _data_aware_low_rank(W, r, _c_in)
                elif W.shape[0] > _dm:
                    return _data_aware_low_rank(W, r, _c_out)
                else:
                    return _data_aware_low_rank(W, r, _c_attn_out)

            encoder_layers[i] = _make_block(
                layer, _per_head, _low_rank_ffn, rank_attn, rank_ff, rank_wo,
            )

    elif method == "adasvd":
        if budget is None:
            raise ValueError("--budget is required for method=adasvd")
        print(f"[adasvd] Running adaptive rank selection (budget={budget}) ...")

        # Step 1: Train hypernetwork and generate ranks.json
        # Note: This modifies the model in-place with MaskedSVDLinear
        ranks = _build_adasvd(model, calib_loader, device, budget, arch, backend=backend)
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

        # Step 3: Apply compression using refactored implementation
        ada_refactored_path = os.path.join(_REPO_ROOT, "eval_encoder", "adasvd_refactored")
        if ada_refactored_path not in sys.path:
            sys.path.insert(0, ada_refactored_path)

        from adasvd_wrapper import compress_adasvd_naive, compress_adasvd_flashsvd

        if backend == "naive":
            print(f"[adasvd] Compressing with naive backend (FWSVDBlock) using ranks from {ranks_path}")
            compress_adasvd_naive(model, ranks_path, device=device)
        elif backend == "flashsvd":
            print(f"[adasvd] Compressing with FlashSVD backend (Triton kernels) using ranks from {ranks_path}")
            compress_adasvd_flashsvd(model, ranks_path, ffn_kernel="v1", device=device)
        else:
            raise ValueError(f"Unsupported backend for AdaSVD: {backend}")

        print(f"[adasvd] Model compressed with AdaSVD {backend} backend")

        # Return the reloaded and compressed model
        return model

    else:
        raise ValueError(f"Unknown method: {method}")

    # Fallback return (should not reach here)
    return model

    print(f"[compress] Done. Replaced {len(encoder_layers)} layers ({arch}).")


# ═══════════════════════════════════════════════════════════════════════════
# 4) Evaluation
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_task(model, loader, task, device):
    """Return (metric_name, metric_value)."""
    from evaluate import load as load_metric

    cfg = TASK_CFG[task]
    metric_name = cfg["metric_name"]
    metric = load_metric("accuracy")

    total, steps = 0.0, 0
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        preds = torch.argmax(logits, dim=-1)
        total += metric.compute(
            predictions=preds.cpu(), references=batch["labels"].cpu()
        )["accuracy"]
        steps += 1

    return metric_name, total / max(steps, 1)


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
        for _ in range(warmup_steps):
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

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
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
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
    bs = batch["input_ids"].shape[0]

    # Collect results from multiple runs
    latencies = []
    throughputs = []
    peak_mems = []

    for run_idx in range(num_runs):
        # warmup
        for _ in range(warmup_steps):
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        for _ in range(measure_steps):
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
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
    "method", "rank", "budget", "scope", "backend", "seed",
    # Calibration info (for FWSVD/Whiten/Ada methods)
    "calib_dataset", "calib_split", "calib_samples", "calib_batches", "calib_seed", "calib_seq_len",
    "metric_name", "metric_value",
    "latency_ms", "throughput_sps",
    "peak_mem_infer_mb",  # Peak memory during inference only
    "peak_mem_e2e_mb",    # Peak memory end-to-end (compression + inference)
    "peak_mem_mb",        # Legacy: same as peak_mem_e2e_mb for backward compatibility
    "param_ratio",  # Compression ratio: compressed_params / original_params
    "notes", "git_commit",
]


def _count_model_params(model):
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _calculate_param_ratio(model, method, rank=None, budget=None, scope="qkv+ffn", arch="bert"):
    """
    Calculate compression ratio (compressed_params / original_params).
    Returns a float between 0 and 1.
    """
    if method == "dense":
        return 1.0

    if method == "adasvd":
        # AdaSVD saves budget report with achieved_ratio
        budget_report_path = "ars_out/budget_report.json"
        if os.path.exists(budget_report_path):
            import json
            with open(budget_report_path) as f:
                report = json.load(f)
                return report["achieved_ratio"]
        # Fallback: assume it matches the budget
        return budget if budget is not None else 0.5

    # For SVD/FWSVD/DRONE: calculate based on rank
    # Get model architecture info
    if arch == "bert":
        # BERT-base: 12 layers, hidden=768, num_heads=12, head_dim=64, intermediate=3072
        num_layers = len(list(model.bert.encoder.layer))
        hidden = model.config.hidden_size
        intermediate = model.config.intermediate_size
    elif arch == "roberta":
        num_layers = len(list(model.roberta.encoder.layer))
        hidden = model.config.hidden_size
        intermediate = model.config.intermediate_size
    elif arch == "modernbert":
        num_layers = len(list(model.model.layers))
        hidden = model.config.hidden_size
        intermediate = model.config.intermediate_size
    else:
        # Unknown architecture, return 1.0
        return 1.0

    if rank is None:
        return 1.0

    # Calculate compressed vs original params
    total_original = 0
    total_compressed = 0

    for layer_idx in range(num_layers):
        # QKV matrices
        if "qkv" in scope:
            # Q, K, V: each is [hidden, hidden]
            for _ in range(3):  # Q, K, V
                orig = hidden * hidden
                comp = (hidden + hidden) * rank
                total_original += orig
                total_compressed += comp

            # Wo (output projection): [hidden, hidden]
            orig = hidden * hidden
            comp = (hidden + hidden) * rank
            total_original += orig
            total_compressed += comp
        else:
            # Keep QKV dense
            for _ in range(3):
                total_original += hidden * hidden
                total_compressed += hidden * hidden
            total_original += hidden * hidden
            total_compressed += hidden * hidden

        # FFN matrices
        if "ffn" in scope:
            # FFN1: [hidden, intermediate]
            orig1 = hidden * intermediate
            comp1 = (hidden + intermediate) * rank
            # FFN2: [intermediate, hidden]
            orig2 = intermediate * hidden
            comp2 = (intermediate + hidden) * rank
            total_original += orig1 + orig2
            total_compressed += comp1 + comp2
        else:
            # Keep FFN dense
            total_original += hidden * intermediate + intermediate * hidden
            total_compressed += hidden * intermediate + intermediate * hidden

    return total_compressed / total_original if total_original > 0 else 1.0


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def write_csv_row(args, metric_name, metric_value,
                  latency_ms, throughput_sps, peak_mem_infer_mb, peak_mem_e2e_mb, param_ratio, dataset_info, calib_info):
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    write_header = not os.path.isfile(args.out_csv)

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
        "rank": args.rank if args.rank is not None else "",
        "budget": args.budget if args.budget is not None else "",
        "scope": args.scope,
        "backend": args.backend,
        "seed": args.seed,
        # Calibration info (only for methods that need it)
        "calib_dataset": calib_info.get("dataset", "") if needs_calib else "",
        "calib_split": calib_info.get("split", "") if needs_calib else "",
        "calib_samples": calib_info.get("samples", "") if needs_calib else "",
        "calib_batches": calib_info.get("batches", "") if needs_calib else "",
        "calib_seed": calib_info.get("seed", "") if needs_calib else "",
        "calib_seq_len": calib_info.get("seq_len", "") if needs_calib else "",
        "metric_name": metric_name,
        "metric_value": f"{metric_value:.6f}",
        "latency_ms": f"{latency_ms:.2f}",
        "throughput_sps": f"{throughput_sps:.1f}",
        "peak_mem_infer_mb": f"{peak_mem_infer_mb:.1f}",
        "peak_mem_e2e_mb": f"{peak_mem_e2e_mb:.1f}",
        "peak_mem_mb": f"{peak_mem_e2e_mb:.1f}",  # Legacy field for backward compatibility
        "param_ratio": f"{param_ratio:.4f}",
        "notes": args.notes,
        "git_commit": _git_commit(),
    }

    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"\n[csv] Appended row to {args.out_csv}")


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
    print(f"  method   : {args.method}  rank={args.rank}  budget={args.budget}  scope={args.scope}")
    print(f"  backend  : {args.backend}")
    print(f"  dtype    : {args.dtype}  device={args.device}")
    print()

    # 1) load model
    model, tokenizer = load_model(args.model_id, args.task, args.dtype, args.device)
    arch, _ = _detect_arch(model)
    print(f"[load] Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params  arch={arch}")

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
    calib_loader = None
    calib_info = {}
    calib_seed = args.calib_seed if args.calib_seed is not None else args.seed

    if args.method in ["fwsvd", "drone", "adasvd", "adawhiten", "adafwsvd"]:
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

        calib_loader = prepare_loader(args.task, tokenizer, args.seq_len, args.batch_size, split=args.calib_split)
        calib_split_name = cfg["train_split"]
        actual_calib_samples = min(args.calib_batches * args.batch_size, len(calib_loader.dataset))

        calib_info = {
            "dataset": args.task,
            "split": calib_split_name,
            "samples": actual_calib_samples,
            "batches": args.calib_batches,
            "seed": calib_seed,
            "seq_len": args.seq_len,
        }

        print(f"[calib] Calibration data:")
        print(f"        dataset={args.task}, split={calib_split_name}")
        print(f"        samples={actual_calib_samples} ({args.calib_batches} batches × {args.batch_size})")
        print(f"        seed={calib_seed}, seq_len={args.seq_len}")

    # 3) compress
    compression_peak_mb = 0.0
    if args.method != "dense":
        model = compress_model(model, args.method, args.rank, args.budget,
                               args.scope, loader, args.device, args.calib_batches, calib_loader=calib_loader, backend=args.backend)

        # Capture peak memory during compression (includes calibration, SVD, etc.)
        if torch.cuda.is_available():
            compression_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"[compress] Peak memory during compression: {compression_peak_mb:.1f} MB")

    # 4) backend
    if args.backend == "flashsvd":
        if args.method == "dense":
            raise ValueError("--backend flashsvd requires a compression method "
                             "(--method != dense).")
        # For AdaSVD, FlashSVD backend is already applied during compression
        # No need to call enable_flashsvd again
        if args.method == "adasvd":
            print("[backend] AdaSVD already using FlashSVD backend (applied during compression)")
        else:
            from eval_encoder.flashsvd_backend import enable_flashsvd
            enable_flashsvd(model)

    # 5) evaluate task metric
    print("\n[eval] Computing task metric ...")
    metric_name, metric_value = evaluate_task(model, loader, args.task, args.device)
    print(f"[eval] {metric_name} = {metric_value:.4f}")

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
                from eval_encoder.flashsvd_backend import enable_flashsvd
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

    # 7) write CSV (with both inference and E2E peaks)
    param_ratio = _calculate_param_ratio(model, args.method, args.rank, args.budget, args.scope, arch)
    write_csv_row(args, metric_name, metric_value,
                  latency_ms, throughput_sps, peak_mem_mb, overall_peak_mb, param_ratio, dataset_info, calib_info)


if __name__ == "__main__":
    main()
