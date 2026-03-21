#!/usr/bin/env python3
"""
convert_asvd_to_hf_dir.py
=========================
Load an ASVD .pt checkpoint ({"model": model, "tokenizer": tokenizer}),
merge every SVDLinear (ALinear @ BLinear → single nn.Linear),
and save the result as a standard HuggingFace model directory.

The merged weight is the rank-r approximation W ≈ A @ B, stored as a
full-rank nn.Linear.  This allows loading with AutoModelForCausalLM.from_pretrained.

Usage
-----
  python convert_asvd_to_hf_dir.py \
      --input  checkpoints/asvd/llama31_8b_instruct/Llama_3.1_8B_Instruct_asvd_raw_0.8.pt \
      --output checkpoints/asvd/llama31_8b_instruct/hf_0.8

  # batch convert all .pt files in a directory
  for pt in checkpoints/asvd/llama31_8b_instruct/*.pt; do
      keep=$(basename "$pt" .pt | grep -oP '[0-9.]+$')
      python convert_asvd_to_hf_dir.py --input "$pt" --output "checkpoints/asvd/llama31_8b_instruct/hf_${keep}"
  done
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


def _find_svdlinear_parent(root: nn.Module):
    """Yield (parent_module, attr_name, child_module) for every SVDLinear."""
    for name, module in root.named_modules():
        for attr, child in module.named_children():
            if type(child).__name__ == "SVDLinear":
                yield module, attr, child


def merge_svd_linears(model: nn.Module) -> int:
    """
    Replace every SVDLinear with a single nn.Linear whose weight = A @ B.
    Returns the number of layers merged.
    """
    replacements = list(_find_svdlinear_parent(model))
    for parent, attr, svd in replacements:
        A: nn.Linear = svd.ALinear   # weight: [out, r]
        B: nn.Linear = svd.BLinear   # weight: [r,  in]

        out_f = A.weight.shape[0]
        in_f  = B.weight.shape[1]
        bias  = A.bias

        merged = nn.Linear(in_f, out_f, bias=(bias is not None),
                           device=A.weight.device, dtype=A.weight.dtype)
        with torch.no_grad():
            merged.weight.copy_(A.weight @ B.weight)
            if bias is not None:
                merged.bias.copy_(bias)

        setattr(parent, attr, merged)

    return len(replacements)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  required=True, help="Path to ASVD .pt checkpoint")
    ap.add_argument("--output", required=True, help="Output HF model directory")
    ap.add_argument("--dtype",  default=None,
                    choices=["fp32", "fp16", "bf16"],
                    help="Cast weights before saving (default: keep original dtype)")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    obj = torch.load(args.input, map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict) and "model" in obj and "tokenizer" in obj):
        print("ERROR: expected {'model': ..., 'tokenizer': ...} dict", file=sys.stderr)
        sys.exit(1)

    model     = obj["model"]
    tokenizer = obj["tokenizer"]
    model.eval()

    print("Merging SVDLinear layers ...")
    n = merge_svd_linears(model)
    print(f"  merged {n} SVDLinear → nn.Linear")

    if args.dtype is not None:
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        model = model.to(dtype_map[args.dtype])
        print(f"  cast to {args.dtype}")

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving HF model to {args.output} ...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")
    print(f"  Load with: AutoModelForCausalLM.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
