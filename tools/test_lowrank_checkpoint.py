#!/usr/bin/env python3
"""
Quick sanity check for a converted low-rank checkpoint directory.

Usage:
    python tools/test_lowrank_checkpoint.py \
        --ckpt hf_ckpts/LowRankArena/llama31_8b_instruct/SVDLLMv2/hf_v2_0.5

Run from: lowrankarena/
"""
import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to converted checkpoint directory")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    print(f"Checking: {ckpt}\n")

    # ── 1. directory structure ─────────────────────────────────────────────────
    required = ["model.pt", "lowrank_config.json", "tokenizer.json", "config.json"]
    print("=== Files ===")
    for f in required:
        exists = (ckpt / f).exists()
        print(f"  {'✓' if exists else '✗'} {f}")

    # ── 2. lowrank_config ──────────────────────────────────────────────────────
    cfg_path = ckpt / "lowrank_config.json"
    if cfg_path.exists():
        meta = json.loads(cfg_path.read_text())
        print(f"\n=== lowrank_config.json ===")
        for k, v in meta.items():
            print(f"  {k}: {v}")

    # ── 3. load model ──────────────────────────────────────────────────────────
    print("\n=== Loading model.pt ===")
    root = Path(__file__).resolve().parent.parent
    framework = meta.get("framework", "dobi") if cfg_path.exists() else "dobi"

    if framework == "svdllm":
        svdllm_dir = root / "baselines" / "SVD-LLM"
        for p in (str(svdllm_dir), str(svdllm_dir / "flashsvd_component")):
            if p not in sys.path:
                sys.path.insert(0, p)
    elif framework == "asvd":
        asvd_dir = root / "baselines" / "ASVD"
        if str(asvd_dir) not in sys.path:
            sys.path.insert(0, str(asvd_dir))

    model = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
    print(f"  model type : {type(model).__name__}")

    # ── 4. check layer types ───────────────────────────────────────────────────
    print("\n=== Layer 0 attention type ===")
    try:
        attn = model.model.layers[0].self_attn
        print(f"  {type(attn).__name__}")
        if hasattr(attn, "q_v_proj"):
            r = attn.q_v_proj.out_features
            d = attn.q_u_proj.out_features
            print(f"  q rank={r}, out_dim={d}  (SVD structure intact ✓)")
        else:
            print("  WARNING: no q_v_proj found — may be dense")
    except Exception as e:
        print(f"  could not inspect: {e}")

    # ── 5. quick forward pass ──────────────────────────────────────────────────
    print("\n=== Quick forward pass ===")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        ids = tok("Hello world", return_tensors="pt").input_ids
        model.eval()
        with torch.no_grad():
            out = model(ids)
        print(f"  input shape : {list(ids.shape)}")
        print(f"  logits shape: {list(out.logits.shape)}")
        print(f"  forward pass: ✓")
    except Exception as e:
        print(f"  forward pass FAILED: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
