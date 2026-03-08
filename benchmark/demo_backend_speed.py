#!/usr/bin/env python3
"""
demo_backend_speed.py — Compare FlashSVD backend inference speed interactively
===============================================================================
Load a single compressed checkpoint, patch it into multiple backends,
then enter an interactive loop: type a sentence, see predictions and
latency for each backend side-by-side.

Backends
--------
  naive     -- PyTorch einsum attention  (default NaiveSVDBlock)
  sdpa      -- PyTorch SDPA attention    (Flash-Attention-2 fused, if available)
  flashsvd  -- Triton v1 kernels         (fused rank-space attention + FFN)
  flashsvd15-- Triton v1.5 kernels       (native bf16/fp16, recommended)

Usage
-----
  python benchmark/demo_backend_speed.py \\
      --checkpoint compressed_models/bert/svd/sst2/svd_r256_naive

  # only compare naive vs flashsvd15, in bf16
  python benchmark/demo_backend_speed.py \\
      --checkpoint /path/to/checkpoint \\
      --backends naive flashsvd15 --dtype bf16

  # full sweep, large batch
  python benchmark/demo_backend_speed.py \\
      --checkpoint /path/to/checkpoint \\
      --backends naive sdpa flashsvd flashsvd15 \\
      --dtype bf16 --batch_size 32 --seq_len 512 --warmup 20 --steps 200
"""

import argparse
import copy
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.encoders.io import load_compressed_model
from src.encoders.backend import enable_sdpa, enable_flashsvd, enable_flashsvd15

DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

ALL_BACKENDS = {
    "naive":      ("naive (einsum)", None),
    "sdpa":       ("naive (sdpa)",   enable_sdpa),
    "flashsvd":   ("flashsvd v1",    enable_flashsvd),
    "flashsvd15": ("flashsvd v1.5",  enable_flashsvd15),
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to compressed model directory (must contain compression_info.json)")
    p.add_argument("--backends", nargs="+",
                   choices=list(ALL_BACKENDS.keys()),
                   default=list(ALL_BACKENDS.keys()),
                   help="Backends to benchmark (default: all four)")
    p.add_argument("--dtype",   choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--warmup",  type=int, default=20,
                   help="Warmup steps (not timed)")
    p.add_argument("--steps",   type=int, default=100,
                   help="Measurement steps")
    p.add_argument("--seq_len",    type=int, default=512,
                   help="Tokenizer max sequence length (512 recommended for meaningful GPU load)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Replicate the input sentence N times to form a batch (32 recommended)")
    return p.parse_args()


def _time_inference(model, inputs, device, warmup, steps):
    """Return (median_latency_ms, pred_label_idx). Uses per-step CUDA sync for accuracy."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs)

    torch.cuda.synchronize(device)
    times = []
    with torch.no_grad():
        for _ in range(steps):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = model(**inputs)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

    times.sort()
    median_ms = times[len(times) // 2] * 1000
    pred_idx  = out.logits[0].argmax().item()
    return median_ms, pred_idx


def _run_backends(text, models, id2label, tokenizer, device, seq_len, batch_size, warmup, steps):
    single = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    # replicate single sentence into a batch so GPU load is meaningful
    inputs = {k: v.expand(batch_size, -1).contiguous().to(device)
              for k, v in single.items()}

    print(f"\n  Running (bs={batch_size}, seq={seq_len}, "
          f"{warmup} warmup + {steps} measure steps)...\n")
    results = []
    for name, m in models:
        print(f"  [{name:<18s}] ", end="", flush=True)
        try:
            lat_ms, pred_idx = _time_inference(m, inputs, device, warmup, steps)
            pred_label = id2label.get(pred_idx, str(pred_idx))
            print(f"latency={lat_ms:7.3f} ms   pred={pred_label}")
            results.append((name, lat_ms, pred_label, None))
        except Exception as e:
            msg = str(e).split("\n")[0][:55]
            print(f"FAILED -- {msg}")
            results.append((name, None, None, msg))

    return results


def _print_table(results, comp_info, dtype, batch_size, seq_len):
    baseline = next((r[1] for r in results if r[1] is not None), None)
    print(f"\n{'='*66}")
    print(f"  {'Backend':<20}  {'Latency (ms)':>12}  {'Speedup':>9}  {'Prediction':>14}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*9}  {'-'*14}")
    for name, lat, pred, err in results:
        if err is not None:
            print(f"  {name:<20}  {'ERROR':>12}  {'---':>9}  {'---':>14}")
        else:
            spd = "baseline" if lat == baseline else f"x{baseline/lat:.2f}"
            print(f"  {name:<20}  {lat:>12.3f}  {spd:>9}  {pred:>14}")
    print(f"{'='*66}")
    print(f"  method={comp_info['method']}  rank={comp_info.get('rank','N/A')}  "
          f"dtype={dtype}  bs={batch_size}  seq={seq_len}")
    print(f"{'='*66}\n")


def main():
    args   = parse_args()
    dtype  = DTYPE_MAP[args.dtype]
    device = args.device

    print(f"\n{'='*66}")
    print(f"  FlashSVD Backend Speed Demo")
    print(f"{'='*66}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Backends   : {', '.join(args.backends)}")
    print(f"  dtype={args.dtype}  device={device}  bs={args.batch_size}  "
          f"seq={args.seq_len}  warmup={args.warmup}  steps={args.steps}")
    print(f"{'='*66}\n")

    # Load once; each backend gets a deepcopy
    print("Loading compressed model...")
    base_model, tokenizer, comp_info = load_compressed_model(
        args.checkpoint, device=device, dtype=dtype
    )
    base_model.eval()
    id2label = getattr(base_model.config, "id2label", {0: "LABEL_0", 1: "LABEL_1"})

    # Prepare one copy per selected backend
    print("\nPreparing backends...")
    models = []
    for key in args.backends:
        name, patch_fn = ALL_BACKENDS[key]
        print(f"  [{name:<18s}] ", end="", flush=True)
        try:
            m = copy.deepcopy(base_model)
            if patch_fn is not None:
                patch_fn(m)
            m.eval()
            models.append((name, m))
            print("ready")
        except Exception as e:
            msg = str(e).split("\n")[0][:55]
            print(f"skipped -- {msg}")

    print(f"\n{len(models)} backend(s) ready. Type a sentence to run inference.")
    print("Enter 'q' or Ctrl-C to quit.\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if text.lower() in ("q", "quit", "exit", ""):
            print("Bye.")
            break

        results = _run_backends(
            text, models, id2label, tokenizer, device,
            args.seq_len, args.batch_size, args.warmup, args.steps,
        )
        _print_table(results, comp_info, args.dtype, args.batch_size, args.seq_len)


if __name__ == "__main__":
    main()
