#!/usr/bin/env python3
"""
demo_backend_speed.py — 对比不同 FlashSVD 后端的单句推理速度
=============================================================
加载同一个压缩 checkpoint，分别切换到 4 种后端。
启动后进入交互循环：每次输入一句话，显示各后端的预测结果与延迟对比。

用法：
  python benchmark/demo_backend_speed.py \\
      --checkpoint compressed_models/bert/svd/sst2/svd_r256_naive

  python benchmark/demo_backend_speed.py \\
      --checkpoint compressed_models/bert/svd/sst2/svd_r256_naive \\
      --dtype bf16 --warmup 20 --steps 200
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


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True,
                   help="压缩模型目录路径（含 compression_info.json）")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=20, help="预热步数（不计入计时）")
    p.add_argument("--steps",  type=int, default=100, help="计时步数")
    p.add_argument("--seq_len",    type=int, default=512, help="tokenizer 最大序列长度（建议 512）")
    p.add_argument("--batch_size", type=int, default=32,
                   help="把输入句子复制成 N 份构成 batch（建议 32，GPU 负载才足够）")
    return p.parse_args()


def _time_inference(model, inputs, device, warmup, steps):
    """返回 (median_latency_ms, pred_label_idx)。用逐次 CUDA synchronize 计时。"""
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
    """对给定句子跑所有后端，打印进度，返回 results 列表。"""
    single = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    # 把单句复制成 batch，让 GPU 有足够负载
    inputs = {k: v.expand(batch_size, -1).contiguous().to(device)
              for k, v in single.items()}

    print(f"\n  Running (bs={batch_size}, seq={seq_len}, {warmup} warmup + {steps} measure steps)...\n")
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
            print(f"FAILED — {msg}")
            results.append((name, None, None, msg))

    return results


def _print_table(results, comp_info, dtype, seq_len):
    baseline = next((r[1] for r in results if r[1] is not None), None)
    print(f"\n{'='*64}")
    print(f"  {'Backend':<20}  {'Latency (ms)':>12}  {'Speedup':>9}  {'Prediction':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*9}  {'-'*12}")
    for name, lat, pred, err in results:
        if err is not None:
            print(f"  {name:<20}  {'ERROR':>12}  {'---':>9}  {'---':>12}")
        else:
            spd = "baseline" if (lat == baseline) else f"×{baseline/lat:.2f}"
            print(f"  {name:<20}  {lat:>12.3f}  {spd:>9}  {pred:>12}")
    print(f"{'='*64}")
    print(f"  method={comp_info['method']}  rank={comp_info.get('rank','N/A')}  "
          f"dtype={dtype}  seq_len={seq_len}")
    print(f"{'='*64}\n")


def main():
    args   = parse_args()
    dtype  = DTYPE_MAP[args.dtype]
    device = args.device

    print(f"\n{'='*64}")
    print(f"  FlashSVD 后端速度对比 Demo")
    print(f"{'='*64}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  dtype={args.dtype}  device={device}  bs={args.batch_size}  "
          f"seq={args.seq_len}  warmup={args.warmup}  steps={args.steps}")
    print(f"{'='*64}\n")

    # ── 加载压缩模型（只加载一次）────────────────────────────────────
    print("Loading compressed model...")
    base_model, tokenizer, comp_info = load_compressed_model(
        args.checkpoint, device=device, dtype=dtype
    )
    base_model.eval()
    id2label = getattr(base_model.config, "id2label", {0: "LABEL_0", 1: "LABEL_1"})

    # ── 为每个后端准备独立副本（patch 后固定不变）─────────────────────
    print("\nPreparing backends...")
    backend_cfgs = [
        ("naive (einsum)", None),
        ("naive (sdpa)",   enable_sdpa),
        ("flashsvd v1",    enable_flashsvd),
        ("flashsvd v1.5",  enable_flashsvd15),
    ]
    models = []
    for name, patch_fn in backend_cfgs:
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
            print(f"skipped — {msg}")

    print(f"\n模型加载完毕，共 {len(models)} 个后端就绪。")
    print("输入句子后按 Enter 开始推理，输入 'q' 或 Ctrl-C 退出。\n")

    # ── 交互循环 ──────────────────────────────────────────────────────
    while True:
        try:
            text = input(">>> 输入句子: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if text.lower() in ("q", "quit", "exit", ""):
            print("退出。")
            break

        results = _run_backends(
            text, models, id2label, tokenizer, device,
            args.seq_len, args.batch_size, args.warmup, args.steps,
        )
        _print_table(results, comp_info, args.dtype, args.seq_len)


if __name__ == "__main__":
    main()
