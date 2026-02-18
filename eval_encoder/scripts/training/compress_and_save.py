#!/usr/bin/env python3
"""
Compress a model and save the checkpoint for later fine-tuning.

Usage:
    python eval_encoder/compress_and_save.py \
        --method fwsvd \
        --rank 300 \
        --backend flashsvd \
        --output_dir eval_encoder/compressed_models
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compress and save model")

    parser.add_argument("--model_id", default="textattack/bert-base-uncased-SST-2")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--method", required=True,
                        choices=["svd", "fwsvd", "drone", "adasvd"])
    parser.add_argument("--rank", type=int)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--backend", default="naive",
                        choices=["naive", "flashsvd"])
    parser.add_argument("--output_dir", default="eval_encoder/compressed_models")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate
    if args.method == "adasvd":
        if args.budget is None:
            raise ValueError("--budget required for adasvd")
        model_name = f"{args.method}_b{args.budget}_{args.backend}"
    else:
        if args.rank is None:
            raise ValueError("--rank required for svd/fwsvd/drone")
        model_name = f"{args.method}_r{args.rank}_{args.backend}"

    output_path = Path(args.output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"将压缩 {args.method} 模型并保存到: {output_path}")
    print(f"这将运行 benchmark 脚本进行压缩...")
    print()

    # Build command
    cmd = [
        sys.executable,
        "eval_encoder/run_encoder_benchmark.py",
        "--method", args.method,
        "--backend", args.backend,
        "--model_id", args.model_id,
        "--task", args.task,
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--dtype", "fp32",
    ]

    if args.method == "adasvd":
        cmd.extend(["--budget", str(args.budget)])
    else:
        cmd.extend(["--rank", str(args.rank)])

    cmd.extend(["--save_model", "--save_dir", str(output_path.parent)])

    # Run benchmark (compress and save)
    print("=" * 60)
    print("运行压缩并保存...")
    print("=" * 60)
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("\n❌ 压缩失败!")
        return 1

    print("\n" + "=" * 60)
    print("✅ 压缩完成!")
    print("=" * 60)
    print(f"\n模型已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
