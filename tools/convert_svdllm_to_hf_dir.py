#!/usr/bin/env python3
"""
convert_svdllm_to_hf_dir.py
============================
Load a SVD-LLM .pt checkpoint ({"model": model, "tokenizer": tokenizer}),
and save it as a directory containing:

  model.pt            – the model object with SVD structure intact (U/V projections)
  tokenizer.*         – standard HuggingFace tokenizer files
  config.json         – model config
  lowrank_config.json – metadata for downstream loading (method, framework)

The low-rank structure (q_u_proj / q_v_proj etc.) is preserved.
To load the model later, add baselines/SVD-LLM and
baselines/SVD-LLM/flashsvd_component to sys.path before torch.load.

Supported methods: V1 (whitening_only), V2 (whitening_hetero), Basis Sharing,
                   V2+local_update.

Usage
-----
  python convert_svdllm_to_hf_dir.py \\
      --input  checkpoints/svdllm/llama31_8b/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt \\
      --output hf_ckpts/LowRankArena/llama31_8b/SVDLLMv1/hf_whitening_only_0.5
"""

import argparse
import json
import os
import sys

import torch


def _load_checkpoint(path: str, map_location: str = "cpu"):
    """Load SVD-LLM .pt with PyTorch>=2.6 compatibility."""
    # Compatibility: older transformers used SiLUActivation; newer versions removed it.
    try:
        import transformers.activations as _act
        import torch.nn as _nn
        if not hasattr(_act, "SiLUActivation"):
            _act.SiLUActivation = _nn.SiLU
    except Exception:
        pass

    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        obj = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("tokenizer")
    if hasattr(obj, "forward"):
        return obj, None
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",  required=True, help="Path to SVD-LLM .pt checkpoint")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--dtype",  default=None,
                    choices=["fp32", "fp16", "bf16"],
                    help="Cast weights before saving (default: keep original dtype)")
    args = ap.parse_args()

    # ── add SVD-LLM to sys.path so SVD_LlamaAttention etc. can be unpickled ──
    _here = os.path.dirname(os.path.abspath(__file__))
    _svdllm_dir = os.path.normpath(os.path.join(_here, "..", "baselines", "SVD-LLM"))
    for _p in [_svdllm_dir, os.path.join(_svdllm_dir, "flashsvd_component")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    print(f"Loading {args.input} ...")
    model, tokenizer = _load_checkpoint(args.input, map_location="cpu")
    model.eval()

    if args.dtype is not None:
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        model = model.to(dtype_map[args.dtype])
        print(f"  cast to {args.dtype}")

    os.makedirs(args.output, exist_ok=True)

    # ── save model object (SVD structure preserved) ───────────────────────────
    model_pt_path = os.path.join(args.output, "model.pt")
    print(f"Saving model (with SVD structure) to {model_pt_path} ...")
    torch.save(model, model_pt_path)

    # ── save config ───────────────────────────────────────────────────────────
    if hasattr(model, "config"):
        # Some old checkpoints store torch.dtype objects in config, which are
        # not JSON-serializable.  Convert them to strings before saving.
        cfg = model.config
        for attr in list(vars(cfg)):
            val = getattr(cfg, attr, None)
            if isinstance(val, torch.dtype):
                setattr(cfg, attr, str(val))
        cfg.save_pretrained(args.output)
        print("  config saved")

    # ── save tokenizer ────────────────────────────────────────────────────────
    if tokenizer is not None:
        tokenizer.save_pretrained(args.output)
        print("  tokenizer saved")
    else:
        model_id = getattr(getattr(model, "config", None), "_name_or_path", None)
        if model_id:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                tok.save_pretrained(args.output)
                print(f"  tokenizer reloaded from {model_id} and saved")
            except Exception as e:
                print(f"  [warn] could not save tokenizer: {e}")

    # ── save lowrank_config.json ──────────────────────────────────────────────
    lowrank_cfg = {
        "framework": "svdllm",
        "source_checkpoint": os.path.abspath(args.input),
    }
    with open(os.path.join(args.output, "lowrank_config.json"), "w") as f:
        json.dump(lowrank_cfg, f, indent=2)
    print("  lowrank_config.json saved")

    print("Done.")
    print(f"  Load: add baselines/SVD-LLM to sys.path, then torch.load('{model_pt_path}')")


if __name__ == "__main__":
    main()
