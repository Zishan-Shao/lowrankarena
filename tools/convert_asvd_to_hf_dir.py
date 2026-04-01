#!/usr/bin/env python3
"""
convert_asvd_to_hf_dir.py
=========================
Load an ASVD .pt checkpoint ({"model": model, "tokenizer": tokenizer}),
and save it as a directory containing:

  model.pt            – the model object with SVDLinear structure intact
  tokenizer.*         – standard HuggingFace tokenizer files
  config.json         – model config
  lowrank_config.json – metadata for downstream loading (method, framework)

The low-rank structure (ALinear / BLinear) is preserved.
To load the model later, add baselines/ASVD to sys.path before torch.load.

Usage
-----
  python convert_asvd_to_hf_dir.py \\
      --input  checkpoints/asvd/llama31_8b/Llama_3.1_8B_asvd_raw_0.8.pt \\
      --output hf_ckpts/LowRankArena/llama31_8b/ASVD/hf_asvd_raw_0.8
"""

import argparse
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  required=True, help="Path to ASVD .pt checkpoint")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--dtype",  default=None,
                    choices=["fp32", "fp16", "bf16"],
                    help="Cast weights before saving (default: keep original dtype)")
    args = ap.parse_args()

    # ── add ASVD to sys.path so SVDLinear etc. can be unpickled ──────────────
    _here = os.path.dirname(os.path.abspath(__file__))
    _asvd_dir = os.path.normpath(os.path.join(_here, "..", "baselines", "ASVD"))
    if _asvd_dir not in sys.path:
        sys.path.insert(0, _asvd_dir)

    # Compatibility: older transformers used SiLUActivation; newer versions removed it.
    try:
        import transformers.activations as _act
        import torch.nn as _nn
        if not hasattr(_act, "SiLUActivation"):
            _act.SiLUActivation = _nn.SiLU
    except Exception:
        pass

    print(f"Loading {args.input} ...")
    obj = torch.load(args.input, map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict) and "model" in obj and "tokenizer" in obj):
        print("ERROR: expected {'model': ..., 'tokenizer': ...} dict", file=sys.stderr)
        sys.exit(1)

    model     = obj["model"]
    tokenizer = obj["tokenizer"]
    model.eval()

    if args.dtype is not None:
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        model = model.to(dtype_map[args.dtype])
        print(f"  cast to {args.dtype}")

    os.makedirs(args.output, exist_ok=True)

    # ── save model object (SVDLinear structure preserved) ─────────────────────
    model_pt_path = os.path.join(args.output, "model.pt")
    print(f"Saving model (with SVDLinear structure) to {model_pt_path} ...")
    torch.save(model, model_pt_path)

    # ── save config ───────────────────────────────────────────────────────────
    if hasattr(model, "config"):
        model.config.save_pretrained(args.output)
        print("  config saved")

    # ── save tokenizer ────────────────────────────────────────────────────────
    tokenizer.save_pretrained(args.output)
    print("  tokenizer saved")

    # ── save lowrank_config.json ──────────────────────────────────────────────
    lowrank_cfg = {
        "framework": "asvd",
        "source_checkpoint": os.path.abspath(args.input),
    }
    with open(os.path.join(args.output, "lowrank_config.json"), "w") as f:
        json.dump(lowrank_cfg, f, indent=2)
    print("  lowrank_config.json saved")

    print("Done.")
    print(f"  Load: add baselines/ASVD to sys.path, then torch.load('{model_pt_path}')")


if __name__ == "__main__":
    main()
