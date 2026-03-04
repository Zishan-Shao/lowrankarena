#!/usr/bin/env python3
"""
E-3b: Fine-tune Recovery Curve for SVD-compressed BERT encoders.

Tracks accuracy vs. training step count to quantify how quickly a compressed
model recovers to dense-equivalent accuracy through fine-tuning.

Training is always done with `naive` (or `sdpa`) backend — the only autograd-
compatible paths.  At each eval checkpoint, additional eval-only backends
(e.g. flashsvd15) can be evaluated with torch.no_grad to capture the
"still fast after fine-tuning" story.

Usage
-----
python benchmark/run_recovery_curve.py \\
    --model_dir compressed_models/bert/mnli/svd_ra48_rf256_rw208_per_head_naive \\
    --task mnli \\
    --eval_steps 0 200 500 1000 \\
    --num_epochs 3 \\
    --train_backend naive \\
    --eval_backends naive,flashsvd15 \\
    --out_csv experiments/expE_recovery.csv

Exit codes
----------
0  : Success
1  : Error (bad train_backend, missing checkpoint, etc.)
"""

import argparse
import csv
import datetime
import os
import re
import sys

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ── repo root on path ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOWRANK = os.path.abspath(os.path.join(_HERE, "..", ".."))   # lowrankarena/
_REPO    = os.path.abspath(os.path.join(_LOWRANK, ".."))       # parent
for _p in (_REPO, _LOWRANK):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_NO_GRAD_BACKENDS = {"flashsvd", "flashsvd15"}
_TRAIN_BACKENDS   = {"naive", "sdpa"}


def _parse_rank_config(model_dir: str) -> str:
    """Parse rank_config string from model_dir basename (mirrors analyze_compute.py)."""
    name = os.path.basename(model_dir.rstrip("/"))
    m = re.search(r'ra(\d+)_rf(\d+)_rw(\d+)_([a-z_]+?)(?:_naive)?$', name)
    if m:
        return f"ra{m.group(1)}_rf{m.group(2)}_rw{m.group(3)}_{m.group(4)}"
    m = re.search(r'b([\d.]+)_([a-z_]+?)(?:_naive)?$', name)
    if m:
        return f"b{m.group(1)}_{m.group(2)}"
    return "unknown"


def _parse_method(model_dir: str, comp_info: dict) -> str:
    """Extract method from comp_info or dirname."""
    if comp_info.get("method"):
        return comp_info["method"]
    name = os.path.basename(model_dir.rstrip("/"))
    for method in ("fwsvd", "adasvd", "drone", "svd"):
        if name.startswith(method):
            return method
    return "unknown"


def _enable_backend(model, backend: str):
    """Patch model in-place for the given backend."""
    if backend == "naive":
        for m in model.modules():
            if hasattr(m, "attn_mode"):
                m.attn_mode = "einsum"
    elif backend == "sdpa":
        try:
            from src.encoders.backend import enable_sdpa
            enable_sdpa(model)
        except Exception:
            for m in model.modules():
                if hasattr(m, "attn_mode"):
                    m.attn_mode = "sdpa"
    elif backend == "flashsvd":
        from src.encoders.backend import enable_flashsvd
        enable_flashsvd(model)
    elif backend == "flashsvd15":
        from src.encoders.backend import enable_flashsvd15
        enable_flashsvd15(model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def _restore_train_backend(model, train_backend: str):
    """Restore model to training backend after an eval-backend evaluation."""
    _enable_backend(model, train_backend)


def _append_csv_row(out_csv: str, row: dict):
    """Atomically append a row to CSV; write header if file is new."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(
        description="E-3b: fine-tune recovery curve for compressed BERT encoders")
    p.add_argument("--model_dir",    required=True,
                   help="Path to compressed checkpoint directory")
    p.add_argument("--task",         required=True,
                   help="Task name (mnli, mrpc, stsb, ...)")
    p.add_argument("--eval_steps",   nargs="+", type=int,
                   default=[0, 200, 500, 1000],
                   help="Global step numbers at which to evaluate")
    p.add_argument("--num_epochs",   type=int, default=3)
    p.add_argument("--train_backend", default="naive",
                   choices=["naive", "sdpa"],
                   help="Backend for training. flashsvd* rejected (no autograd).")
    p.add_argument("--eval_backends", default="naive",
                   help="Comma-separated backends for eval at each checkpoint. "
                        "Extra backends (e.g. flashsvd15) run eval-only (no_grad).")
    p.add_argument("--seq_len",      type=int, default=512)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--dtype",        choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                   help="Fraction of total training steps used for LR warmup")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_csv",      default=None,
                   help="CSV path to append recovery curve rows")
    return p.parse_args()


def main():
    args = parse_args()

    # ── validate train_backend ────────────────────────────────────────────────
    if args.train_backend in _NO_GRAD_BACKENDS:
        print(f"[error] --train_backend={args.train_backend} has no autograd support.")
        print(f"  Use --train_backend naive or --train_backend sdpa.")
        sys.exit(1)

    DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = DTYPE_MAP[args.dtype]
    device = args.device

    eval_backends = [b.strip() for b in args.eval_backends.split(",") if b.strip()]
    eval_steps_set = set(args.eval_steps)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"[load] Loading compressed model: {args.model_dir}")
    from src.encoders.io import load_compressed_model
    model, tokenizer, comp_info = load_compressed_model(
        args.model_dir, device=device, dtype=dtype)

    rank_config = _parse_rank_config(args.model_dir)
    method = _parse_method(args.model_dir, comp_info)
    original_model_id = comp_info.get("model_id")

    # Apply training backend
    _enable_backend(model, args.train_backend)
    model.train()

    # ── prepare data loaders ──────────────────────────────────────────────────
    print(f"[data] Loading task={args.task} ...")
    from src.encoders.evaluate import prepare_data, evaluate_task

    train_loader, val_loader = prepare_data(
        args.task, tokenizer, args.seq_len, args.batch_size)

    if train_loader is None:
        print(f"[error] Task {args.task!r} has no training split.")
        sys.exit(1)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.num_epochs
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))

    print(f"[train] steps_per_epoch={steps_per_epoch}  total={total_steps}  "
          f"warmup={warmup_steps}  epochs={args.num_epochs}")
    print(f"[train] train_backend={args.train_backend}  "
          f"eval_backends={eval_backends}")
    print(f"[train] eval at steps: {sorted(eval_steps_set)}")

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── helper: build a CSV row ───────────────────────────────────────────────
    def _make_row(global_step, epoch_frac, eval_backend, metric_name, metric_value, notes=""):
        return {
            "timestamp":     datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "task":          args.task,
            "method":        method,
            "rank_config":   rank_config,
            "train_backend": args.train_backend,
            "eval_backend":  eval_backend,
            "seq_len":       str(args.seq_len),
            "batch_size":    str(args.batch_size),
            "dtype":         args.dtype,
            "global_step":   str(global_step),
            "epoch_frac":    f"{epoch_frac:.4f}",
            "metric_name":   metric_name,
            "metric_value":  f"{metric_value:.6f}",
            "notes":         notes,
        }

    # ── helper: evaluate at a checkpoint ─────────────────────────────────────
    def _maybe_eval(global_step: int):
        """Evaluate on each eval_backend and write CSV rows."""
        epoch_frac = global_step / steps_per_epoch

        for eb in eval_backends:
            print(f"\n[eval] step={global_step}  eval_backend={eb}  "
                  f"epoch_frac={epoch_frac:.3f}")

            try:
                # Patch to eval backend (may be different from train backend)
                _enable_backend(model, eb)

                results, avg_loss = evaluate_task(
                    model, val_loader, args.task, device,
                    original_model_id=original_model_id)

                for metric_name, metric_value in results.items():
                    if isinstance(metric_value, (int, float)):
                        row = _make_row(
                            global_step, epoch_frac, eb,
                            metric_name, float(metric_value),
                            notes=f"loss={avg_loss:.4f}")
                        if args.out_csv:
                            _append_csv_row(args.out_csv, row)
                        print(f"  {metric_name}={metric_value:.4f}  loss={avg_loss:.4f}")

            except Exception as exc:
                print(f"[eval] ERROR at step={global_step} eval_backend={eb}: {exc}")
                raise
            finally:
                # Always restore training backend before next training step
                _restore_train_backend(model, args.train_backend)

    # ── initial evaluation (step 0, before any training) ─────────────────────
    if 0 in eval_steps_set:
        _maybe_eval(0)

    # ── training loop ─────────────────────────────────────────────────────────
    global_step = 0
    model.train()

    for epoch in range(args.num_epochs):
        print(f"\n[epoch] {epoch+1}/{args.num_epochs}  global_step={global_step}")
        for batch in train_loader:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            model.train()
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step in eval_steps_set:
                _maybe_eval(global_step)
                model.train()

            if global_step % 100 == 0:
                print(f"  step={global_step}  loss={loss.item():.4f}")

    # ── final evaluation ──────────────────────────────────────────────────────
    if global_step not in eval_steps_set:
        print(f"\n[eval] Final evaluation at step={global_step}")
        _maybe_eval(global_step)

    print(f"\n[done] Recovery curve complete.  global_step={global_step}")
    if args.out_csv:
        print(f"[csv] Results appended → {args.out_csv}")


if __name__ == "__main__":
    main()
