#!/usr/bin/env python3
"""
E-3a: Per-Step Training Time Benchmark for SVD-compressed BERT encoders.

Measures the full training step cost (forward + backward + optimizer.step)
for compressed models.  Only `naive` and `sdpa` backends support autograd;
`flashsvd` / `flashsvd15` Triton kernels have no backward pass and will
cause this script to exit with code 2 (SKIP convention).

Usage
-----
python eval_encoder/scripts/run_train_timing.py \\
    --model_dir eval_encoder/models/mnli/svd_ra48_rf256_rw208_per_head_naive \\
    --task mnli --backend naive --dtype bf16 \\
    --warmup 50 --measure 100 \\
    --out_csv eval_encoder/eval_results/expE_train_timing.csv

Exit codes
----------
0  : Success — row appended to out_csv
2  : SKIP — unsupported backend (flashsvd / flashsvd15), no row written
1+ : Error
"""

import argparse
import csv
import datetime
import math
import os
import re
import sys
import time

import torch

# ── repo root on path ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOWRANK = os.path.abspath(os.path.join(_HERE, "..", ".."))   # lowrankarena/
_REPO    = os.path.abspath(os.path.join(_LOWRANK, ".."))       # parent
for _p in (_REPO, _LOWRANK):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Backends that have autograd support (full training step possible)
_TRAIN_BACKENDS = {"naive", "sdpa"}
# Backends with Triton kernels and no autograd
_NO_GRAD_BACKENDS = {"flashsvd", "flashsvd15"}


def _parse_rank_config(model_dir: str) -> str:
    """
    Build a rank_config string from the model_dir basename.

    Mirrors the fallback chain in analyze_compute.py main():
      1. Parse ra/rf/rw from dirname (e.g. svd_ra48_rf256_rw208_per_head_naive)
      2. Parse budget (e.g. adasvd_b0.527_per_head_naive)
      3. Fallback to "unknown"
    """
    name = os.path.basename(model_dir.rstrip("/"))
    # Try ra/rf/rw pattern
    m = re.search(r'ra(\d+)_rf(\d+)_rw(\d+)_([a-z_]+?)(?:_naive)?$', name)
    if m:
        ra, rf, rw, qkv = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"ra{ra}_rf{rf}_rw{rw}_{qkv}"
    # Try budget pattern
    m = re.search(r'b([\d.]+)_([a-z_]+?)(?:_naive)?$', name)
    if m:
        return f"b{m.group(1)}_{m.group(2)}"
    return "unknown"


def _parse_method(model_dir: str) -> str:
    """Extract method from dirname (svd / fwsvd / drone / adasvd)."""
    name = os.path.basename(model_dir.rstrip("/"))
    for method in ("fwsvd", "adasvd", "drone", "svd"):
        if name.startswith(method):
            return method
    return "unknown"


def _make_synthetic_train_loader(
    task: str, seq_len: int, batch_size: int, num_batches: int,
    num_labels: int, is_regression: bool,
    vocab_size: int = 30522, seed: int = 42,
):
    """
    Synthetic DataLoader for training timing — decouples from real data loading.

    Batch structure matches HuggingFace model forward:
        input_ids, attention_mask, token_type_ids, labels

    Labels:
        is_regression → float32 scalar in [0, 1]
        else          → long in [0, num_labels)
    """
    from torch.utils.data import DataLoader, TensorDataset

    N = num_batches * batch_size
    rng = torch.Generator()
    rng.manual_seed(seed)

    input_ids      = torch.randint(100, vocab_size - 100, (N, seq_len),
                                   dtype=torch.long, generator=rng)
    attention_mask = torch.ones(N, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(N, seq_len, dtype=torch.long)

    if is_regression:
        labels = torch.rand(N, generator=rng).float()
    else:
        labels = torch.randint(0, num_labels, (N,), generator=rng)

    ds = TensorDataset(input_ids, attention_mask, token_type_ids, labels)

    def _collate(batch):
        ii, am, tt, lb = zip(*batch)
        return {
            "input_ids":      torch.stack(ii),
            "attention_mask": torch.stack(am),
            "token_type_ids": torch.stack(tt),
            "labels":         torch.stack(lb),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)


# Task metadata (num_labels, is_regression) — subset of GLUE_TASKS
_TASK_META = {
    "cola":  {"num_labels": 2, "is_regression": False},
    "sst2":  {"num_labels": 2, "is_regression": False},
    "mrpc":  {"num_labels": 2, "is_regression": False},
    "qqp":   {"num_labels": 2, "is_regression": False},
    "mnli":  {"num_labels": 3, "is_regression": False},
    "qnli":  {"num_labels": 2, "is_regression": False},
    "rte":   {"num_labels": 2, "is_regression": False},
    "stsb":  {"num_labels": 1, "is_regression": True},
    # SuperGLUE
    "boolq": {"num_labels": 2, "is_regression": False},
    "cb":    {"num_labels": 3, "is_regression": False},
    "rte_sg":{"num_labels": 2, "is_regression": False},
    "wic":   {"num_labels": 2, "is_regression": False},
    "copa":  {"num_labels": 3, "is_regression": False},
    # Robustness
    "hans":   {"num_labels": 2, "is_regression": False},
    "anli_r1":{"num_labels": 3, "is_regression": False},
    "anli_r2":{"num_labels": 3, "is_regression": False},
    "anli_r3":{"num_labels": 3, "is_regression": False},
}


def _enable_sdpa(model):
    """Patch all MinimalSVDBlock / NaiveSVDBlock layers to use sdpa attention."""
    try:
        from eval_encoder.flashsvd_backend import enable_sdpa
        enable_sdpa(model)
    except Exception:
        # Fallback: patch attn_mode directly
        for m in model.modules():
            if hasattr(m, "attn_mode"):
                m.attn_mode = "sdpa"


def _restore_naive(model):
    """Restore model to naive (einsum) attention mode."""
    for m in model.modules():
        if hasattr(m, "attn_mode"):
            m.attn_mode = "einsum"


def parse_args():
    p = argparse.ArgumentParser(
        description="E-3a: per-step training time for compressed BERT encoders")
    p.add_argument("--model_dir",  required=True,
                   help="Path to compressed checkpoint directory")
    p.add_argument("--task",       required=True,
                   help="Task name (mnli, mrpc, ...)")
    p.add_argument("--backend",    default="naive",
                   choices=["naive", "sdpa", "flashsvd", "flashsvd15"],
                   help="Backend to use.  flashsvd/flashsvd15 → SKIP (no autograd)")
    p.add_argument("--seq_len",    type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dtype",      choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--warmup",     type=int, default=50,
                   help="Number of warmup steps (not timed)")
    p.add_argument("--measure",    type=int, default=100,
                   help="Number of measured steps")
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_csv",    default=None,
                   help="CSV path to append result row")
    return p.parse_args()


def main():
    args = parse_args()

    # ── SKIP unsupported backends ──────────────────────────────────────────────
    if args.backend in _NO_GRAD_BACKENDS:
        print(f"SKIP (no autograd): {args.backend}")
        print(f"  FlashSVD/flashsvd15 Triton kernels have no backward pass.")
        print(f"  Use --backend naive or --backend sdpa for training timing.")
        sys.exit(2)

    DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = DTYPE_MAP[args.dtype]
    device = args.device

    # ── task metadata ─────────────────────────────────────────────────────────
    task_meta = _TASK_META.get(args.task, {"num_labels": 2, "is_regression": False})
    num_labels   = task_meta["num_labels"]
    is_regression = task_meta["is_regression"]

    # ── load model ────────────────────────────────────────────────────────────
    print(f"[load] Loading compressed model: {args.model_dir}")
    from eval_encoder.load_compressed_model import load_compressed_model
    model, tokenizer, comp_info = load_compressed_model(
        args.model_dir, device=device, dtype=dtype)

    # Apply sdpa backend patch if requested
    if args.backend == "sdpa":
        print("[backend] Patching to sdpa attention mode ...")
        _enable_sdpa(model)
    # naive: no patch needed (default state after load)

    model.train()

    # ── rank_config and method ────────────────────────────────────────────────
    rank_config = _parse_rank_config(args.model_dir)
    method = comp_info.get("method") or _parse_method(args.model_dir)

    # ── synthetic DataLoader ──────────────────────────────────────────────────
    num_batches = args.warmup + args.measure + 8   # +8 headroom
    loader = _make_synthetic_train_loader(
        args.task, args.seq_len, args.batch_size, num_batches,
        num_labels, is_regression, seed=42)
    data_iter = iter(loader)

    def _next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── reset peak memory stats ───────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ── timing loop ───────────────────────────────────────────────────────────
    times = []

    print(f"[timing] Warmup={args.warmup} Measure={args.measure} "
          f"backend={args.backend} dtype={args.dtype} ...")

    for step in range(args.warmup + args.measure):
        batch = _next_batch()
        batch = {k: v.to(device) for k, v in batch.items()}

        model.train()

        t0 = time.perf_counter()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        if step >= args.warmup:
            times.append((t1 - t0) * 1000.0)   # ms

        if step < args.warmup and (step + 1) % max(1, args.warmup // 5) == 0:
            print(f"  [warmup] step={step+1}/{args.warmup}")

    # ── compute stats ─────────────────────────────────────────────────────────
    import statistics as _stats
    mean_ms = _stats.mean(times)
    std_ms  = _stats.stdev(times) if len(times) > 1 else 0.0
    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
               if torch.cuda.is_available() else 0.0)

    print(f"[result] train_step_ms = {mean_ms:.2f} ± {std_ms:.2f} ms  "
          f"peak_mem = {peak_mb:.1f} MB")

    # ── CSV output ────────────────────────────────────────────────────────────
    if args.out_csv:
        row = {
            "timestamp":           datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "task":                args.task,
            "method":              method,
            "backend":             args.backend,
            "seq_len":             str(args.seq_len),
            "batch_size":          str(args.batch_size),
            "dtype":               args.dtype,
            "rank_config":         rank_config,
            "train_step_ms_mean":  f"{mean_ms:.2f}",
            "train_step_ms_std":   f"{std_ms:.2f}",
            "peak_mem_train_mb":   f"{peak_mb:.1f}",
            "notes":               f"warmup={args.warmup} measure={args.measure}",
        }
        write_header = not os.path.exists(args.out_csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"[csv] Row appended → {args.out_csv}")


if __name__ == "__main__":
    main()
