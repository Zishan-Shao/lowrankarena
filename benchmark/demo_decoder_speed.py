#!/usr/bin/env python3
"""
demo_decoder_speed.py — Compare FlashSVD decoder inference speed interactively
===============================================================================
Load a single SVD-LLM v1 checkpoint, toggle between backends via env vars,
then enter an interactive loop: type a prompt, see generated text and per-token
latency for each backend side-by-side.

Backends
--------
  svd      -- Low-rank matmul baseline  (SVDLLM_FLASH_FALLBACK=1)
  flashsvd -- FlashSVD fast path        (Triton RoPE-attn + SwiGLU kernels,
                                         FlashSVDDenseKVCache)

Special commands
----------------
  bench   -- latency sweep with random tokens (no text output)
  all     -- run all backends on the typed prompt

Usage
-----
  cd baselines/SVD-LLM
  python ../../benchmark/demo_decoder_speed.py \\
      --checkpoint checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt

  # only flashsvd backend, longer generation
  python ../../benchmark/demo_decoder_speed.py \\
      --checkpoint /path/to/checkpoint \\
      --backends flashsvd --new_tokens 64 --dtype bf16

  # bench sweep (random tokens, batch=1, prompt=512, decode=128)
  python ../../benchmark/demo_decoder_speed.py \\
      --checkpoint /path/to/checkpoint \\
      --backends svd flashsvd --warmup 5 --bench_tokens 128 --prompt_len 512
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # lowrankarena/
_SVDLLM = os.path.join(_ROOT, "baselines", "SVD-LLM")
for _p in [_ROOT, _SVDLLM]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.model_utils import get_model_from_local, get_model_from_huggingface

DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# ── Backend configuration (env vars read at forward-time in svd_llama.py) ─────

def _set_env(name: str, value: bool) -> None:
    if value:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)


def _configure_backend(key: str, *, ffn_backend: str = "auto",
                       enable_flash_dense_attn: bool = True) -> None:
    """Toggle SVD-LLM v1 backend via environment variables."""
    if key == "svd":
        _set_env("SVDLLM_FLASH_FALLBACK", True)
        _set_env("FLASH_SVD_DISABLE_FFN", True)
        _set_env("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE", False)
        _set_env("FLASH_SVD_BASELINE_DENSE_KVCACHE", False)
        _set_env("FLASH_SVD_REFERENCE_DENSE_ATTN", False)
        _set_env("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", False)
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH", False)
        os.environ["FLASH_SVD_FFN_BACKEND"] = "dual_split_cublas_legacy"
    else:  # flashsvd
        _set_env("SVDLLM_FLASH_FALLBACK", False)
        _set_env("FLASH_SVD_DISABLE_FFN", False)
        _set_env("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE", enable_flash_dense_attn)
        _set_env("FLASH_SVD_BASELINE_DENSE_KVCACHE", False)
        _set_env("FLASH_SVD_REFERENCE_DENSE_ATTN", False)
        _set_env("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", False)
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH", True)
        os.environ["FLASH_SVD_MLP_CUDA_GRAPH_SCOPE"] = "mlp"
        os.environ["FLASH_SVD_FFN_BACKEND"] = ffn_backend


BACKEND_LABELS = {
    "svd":      "svd (fallback)",
    "flashsvd": "flashsvd",
}


# ── Cache construction ─────────────────────────────────────────────────────────

def _make_cache(model, key: str, batch_size: int, max_cache_len: int, dtype, device):
    """Return the appropriate KV-cache object for the given backend."""
    if key == "flashsvd":
        try:
            from flashsvd_component.dense_cache import FlashSVDDenseKVCache
            return FlashSVDDenseKVCache(
                model.config,
                max_batch_size=batch_size,
                max_cache_len=max_cache_len,
                device=device,
                dtype=dtype,
            )
        except Exception:
            pass  # fall through to StaticCache
    from transformers.cache_utils import StaticCache
    return StaticCache(
        model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )


# ── Text generation (user prompt → actual output) ─────────────────────────────

@torch.no_grad()
def _generate(model, tokenizer, prompt: str, backend_key: str, *,
              new_tokens: int, device, dtype,
              ffn_backend: str, enable_flash_dense_attn: bool) -> tuple[str, float, float, float]:
    """
    Greedy-decode `new_tokens` tokens from `prompt`.

    Returns (generated_text, prefill_ms, decode_ms_per_token, decode_tok_s).
    Prefill and decode are timed separately.
    """
    model.eval()
    _configure_backend(backend_key, ffn_backend=ffn_backend,
                       enable_flash_dense_attn=enable_flash_dense_attn)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    max_cache_len = prompt_len + new_tokens + 2

    cache = _make_cache(model, backend_key, 1, max_cache_len, dtype, device)
    pos_pre = torch.arange(prompt_len, device=device).unsqueeze(0)

    # ── Prefill ────────────────────────────────────────────────────────────────
    cache_pos_pre = torch.arange(prompt_len, device=device, dtype=torch.long)
    if "cuda" in str(device):
        torch.cuda.synchronize(device)
    t_pre = time.perf_counter()
    out = model(input_ids=input_ids, past_key_values=cache,
                cache_position=cache_pos_pre, use_cache=True)
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    if "cuda" in str(device):
        torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - t_pre) * 1000.0

    # ── Decode loop (timed as whole loop, matching evaluater.py) ───────────────
    t0 = time.perf_counter()
    generated: list[int] = [int(next_tok.item())]
    for step in range(new_tokens - 1):
        cache_pos = torch.tensor([prompt_len + step + 1], device=device, dtype=torch.long)
        out = model(input_ids=next_tok, past_key_values=cache,
                    cache_position=cache_pos, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_tok.item()))

    if "cuda" in str(device):
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    ms_per_tok = dt * 1000.0 / max(1, new_tokens)
    tok_s = new_tokens / dt if dt > 0 else float("inf")
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, prefill_ms, ms_per_tok, tok_s


# ── Latency benchmark (random tokens, per-step median) ────────────────────────

@torch.no_grad()
def _bench(model, backend_key: str, *,
           prompt_len: int, new_tokens: int, batch_size: int,
           device, dtype, warmup: int,
           ffn_backend: str, enable_flash_dense_attn: bool) -> tuple[float, float]:
    """
    Pure decode-latency benchmark using random token inputs.
    Timing matches evaluater.py: one sync before loop, one sync after.

    Returns (ms_per_token, tok_s).
    """
    model.eval()
    vocab = int(getattr(model.config, "vocab_size", 32000))
    max_cache_len = prompt_len + new_tokens + warmup + 2

    def _one_run(timed: bool) -> float:
        _configure_backend(backend_key, ffn_backend=ffn_backend,
                           enable_flash_dense_attn=enable_flash_dense_attn)
        cache = _make_cache(model, backend_key, batch_size, max_cache_len, dtype, device)
        input_ids = torch.randint(0, vocab, (batch_size, prompt_len),
                                  device=device, dtype=torch.long)
        cache_pos_pre = torch.arange(prompt_len, device=device, dtype=torch.long)

        out = model(input_ids=input_ids, past_key_values=cache,
                    cache_position=cache_pos_pre, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if timed and "cuda" in str(device):
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for step in range(new_tokens - 1):
            cache_pos = torch.tensor([prompt_len + step + 1],
                                     device=device, dtype=torch.long)
            out = model(input_ids=next_tok, past_key_values=cache,
                        cache_position=cache_pos, use_cache=True)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if timed and "cuda" in str(device):
            torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) if timed else 0.0

    for _ in range(warmup):
        _one_run(timed=False)
    if "cuda" in str(device):
        torch.cuda.synchronize(device)

    dt = _one_run(timed=True)
    ms_per_tok = dt * 1000.0 / max(1, new_tokens - 1)
    tok_s = batch_size * (new_tokens - 1) / dt if dt > 0 else float("inf")
    return ms_per_tok, tok_s


# ── Table printers ─────────────────────────────────────────────────────────────

def _print_gen_table(results: list, args) -> None:
    baseline_ms = next((r["ms"] for r in results if r["ms"] is not None), None)
    print(f"\n{'='*78}")
    print(f"  {'Backend':<22}  {'prefill':>9}  {'ms/tok':>8}  {'tok/s':>8}  {'Speedup':>9}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*9}")
    for r in results:
        label = r["label"]
        if r["err"] is not None:
            print(f"  {label:<22}  {'ERROR':>9}  {'---':>8}  {'---':>8}  {'---':>9}")
        else:
            pre, ms, tok_s = r["prefill_ms"], r["ms"], r["tok_s"]
            spd = "baseline" if ms == baseline_ms else f"x{baseline_ms/ms:.2f}"
            print(f"  {label:<22}  {pre:>8.1f}ms  {ms:>8.3f}  {tok_s:>8.1f}  {spd:>9}")
    print(f"{'='*78}")
    print(f"  prompt_toks={results[0].get('prompt_len','?')}  "
          f"new_tokens={args.new_tokens}  dtype={args.dtype}  device={args.device}")
    print(f"{'='*78}\n")


def _print_bench_table(results: list, args) -> None:
    baseline_ms = next((r["ms"] for r in results if r["ms"] is not None), None)
    print(f"\n{'='*66}")
    print(f"  {'Backend':<22}  {'ms/tok':>8}  {'tok/s':>8}  {'Speedup':>9}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*9}")
    for r in results:
        label = r["label"]
        if r["err"] is not None:
            print(f"  {label:<22}  {'ERROR':>8}  {'---':>8}  {'---':>9}")
        else:
            ms, tok_s = r["ms"], r["tok_s"]
            spd = "baseline" if ms == baseline_ms else f"x{baseline_ms/ms:.2f}"
            print(f"  {label:<22}  {ms:>8.3f}  {tok_s:>8,.0f}  {spd:>9}")
    print(f"{'='*66}")
    print(f"  prompt={args.prompt_len}  decode={args.bench_tokens}  "
          f"bs={args.batch_size}  dtype={args.dtype}")
    print(f"{'='*66}\n")


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True,
                   help="SVD-LLM v1 checkpoint (.pt file or HF model id/dir)")
    p.add_argument("--backends", nargs="+", choices=["svd", "flashsvd"],
                   default=["svd", "flashsvd"],
                   help="Backends to include (default: both)")
    p.add_argument("--dtype",   choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--new_tokens", type=int, default=32,
                   help="Tokens to generate per interactive query")
    p.add_argument("--warmup",  type=int, default=3,
                   help="Warmup decode rounds before timing")
    # bench-mode params
    p.add_argument("--prompt_len",   type=int, default=512,
                   help="Prompt token length for bench mode")
    p.add_argument("--bench_tokens", type=int, default=128,
                   help="Decode steps for bench mode")
    p.add_argument("--batch_size",   type=int, default=1,
                   help="Batch size for bench mode")
    # FlashSVD options
    p.add_argument("--ffn_backend", default="auto",
                   choices=["auto", "dual_split_cublas", "dual_split_cublas_legacy",
                            "dual_split_kernel", "dual_split_kernel_v2",
                            "dual_split_kernel_v2_sm80", "dual_split_kernel_v3"],
                   help="FlashSVD FFN backend (default: auto)")
    p.add_argument("--no_flash_dense_attn", action="store_true",
                   help="Disable FlashSVDDenseKVCache for flashsvd backend")
    p.add_argument("--hf_token", default=None)
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────────

def _load(checkpoint: str, hf_token: str | None):
    path = Path(checkpoint)
    if path.exists():
        return get_model_from_local(str(path))
    return get_model_from_huggingface(checkpoint, hf_token=hf_token)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    dtype  = DTYPE_MAP[args.dtype]
    device = args.device
    enable_flash_dense = not args.no_flash_dense_attn

    print(f"\n{'='*66}")
    print(f"  FlashSVD Decoder Speed Demo  (SVD-LLM v1)")
    print(f"{'='*66}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Backends   : {', '.join(args.backends)}")
    print(f"  dtype={args.dtype}  device={device}  new_tokens={args.new_tokens}  "
          f"warmup={args.warmup}")
    print(f"  ffn_backend={args.ffn_backend}  "
          f"flash_dense_attn={enable_flash_dense}")
    print(f"{'='*66}\n")

    # Set a safe default env state before loading
    _configure_backend("svd")

    print("Loading checkpoint...")
    model, tokenizer = _load(args.checkpoint, args.hf_token)
    model = model.to(dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Model: {getattr(model.config, '_name_or_path', '?')}")
    print(f"  Params: {n_params:.2f}B  dtype={args.dtype}")

    # Warmup each backend once with a short random-token run
    print(f"\nWarming up ({args.warmup} rounds per backend)...")
    for key in args.backends:
        label = BACKEND_LABELS[key]
        print(f"  [{label:<20s}] ", end="", flush=True)
        try:
            _bench(model, key, prompt_len=64, new_tokens=16, batch_size=1,
                   device=device, dtype=dtype, warmup=args.warmup,
                   ffn_backend=args.ffn_backend,
                   enable_flash_dense_attn=enable_flash_dense)
            print("done")
        except Exception as e:
            print(f"FAILED -- {str(e).split(chr(10))[0][:55]}")

    backend_hint = "/".join(args.backends) + "/all/bench"
    print(f"\n{len(args.backends)} backend(s) ready. Warmup complete.")
    print("Commands: type a prompt, then choose a backend.")
    print("          'bench' runs a latency sweep with random tokens.")
    print("          'q' or Ctrl-C to quit.\n")

    current_prompt: str | None = None

    while True:
        # ── Step 1: get prompt ─────────────────────────────────────────────────
        try:
            prompt = input("prompt  >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if prompt.lower() in ("q", "quit", "exit"):
            print("Bye.")
            break
        if not prompt:
            continue
        current_prompt = prompt

        # ── Step 2: choose backend ─────────────────────────────────────────────
        try:
            choice = input(f"backend [{backend_hint}] >>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if choice in ("q", "quit", "exit"):
            print("Bye.")
            break

        print()

        # ── bench mode ────────────────────────────────────────────────────────
        if choice == "bench":
            results = []
            for key in args.backends:
                label = BACKEND_LABELS[key]
                print(f"  [{label:<20s}] ", end="", flush=True)
                try:
                    ms, tok_s = _bench(
                        model, key,
                        prompt_len=args.prompt_len,
                        new_tokens=args.bench_tokens,
                        batch_size=args.batch_size,
                        device=device, dtype=dtype, warmup=args.warmup,
                        ffn_backend=args.ffn_backend,
                        enable_flash_dense_attn=enable_flash_dense,
                    )
                    print(f"{ms:.3f} ms/tok  {tok_s:,.0f} tok/s")
                    results.append({"key": key, "label": label,
                                    "ms": ms, "tok_s": tok_s, "err": None})
                except Exception as e:
                    msg = str(e).split("\n")[0][:55]
                    print(f"FAILED -- {msg}")
                    results.append({"key": key, "label": label,
                                    "ms": None, "tok_s": None, "err": msg})
            _print_bench_table(results, args)
            continue

        # ── generate mode ──────────────────────────────────────────────────────
        keys = (args.backends if choice in ("all", "")
                else [choice] if choice in args.backends
                else None)

        if keys is None:
            print(f"  Unknown backend '{choice}'. Valid: {backend_hint}\n")
            continue

        results = []
        for key in keys:
            label = BACKEND_LABELS[key]
            print(f"  [{label:<20s}] generating {args.new_tokens} tokens...",
                  flush=True)
            try:
                text, prefill_ms, ms_tok, tok_s = _generate(
                    model, tokenizer, current_prompt, key,
                    new_tokens=args.new_tokens,
                    device=device, dtype=dtype,
                    ffn_backend=args.ffn_backend,
                    enable_flash_dense_attn=enable_flash_dense,
                )
                prompt_len = len(tokenizer(current_prompt).input_ids)
                preview = text[:100].replace("\n", " ")
                print(f"  [{label:<20s}] prefill={prefill_ms:.1f}ms  "
                      f"decode={ms_tok:.3f}ms/tok  {tok_s:.1f}tok/s")
                print(f"  output: {preview}")
                results.append({"key": key, "label": label, "prefill_ms": prefill_ms,
                                 "ms": ms_tok, "tok_s": tok_s, "prompt_len": prompt_len,
                                 "err": None})
            except Exception as e:
                msg = str(e).split("\n")[0][:55]
                print(f"  [{label:<20s}] FAILED -- {msg}")
                results.append({"key": key, "label": label, "prefill_ms": None,
                                 "ms": None, "tok_s": None, "prompt_len": 0,
                                 "err": msg})

        if len(results) > 1:
            _print_gen_table(results, args)
        else:
            print()


if __name__ == "__main__":
    main()
