#!/usr/bin/env python3
"""
convert_svdllm_to_hf_dir.py
============================
Load a SVD-LLM .pt checkpoint ({\"model\": model, \"tokenizer\": tokenizer}),
merge every pair of low-rank projections (X_u_proj @ X_v_proj → single nn.Linear),
and save the result as a standard HuggingFace model directory.

Supported methods: V1 (whitening_only), V2 (whitening_hetero), Basis Sharing,
                   V2+local_update, update_only.

The merged weight for each projection is:  W = u_proj.weight @ v_proj.weight
This gives the rank-r approximation of the original weight matrix, stored as a
full-rank nn.Linear — compatible with AutoModelForCausalLM.from_pretrained.

Usage
-----
  python convert_svdllm_to_hf_dir.py \\
      --input  checkpoints/svdllm/llama31_8b/meta_llama_Llama_3.1_8B_whitening_only_0.5.pt \\
      --output checkpoints/svdllm/llama31_8b/hf_v1_0.5

  # batch convert all whitening_only checkpoints
  for pt in checkpoints/svdllm/llama31_8b/*_whitening_only_*.pt; do
      tag=$(basename "$pt" .pt | grep -oP '(whitening_only|whitening_then_update|basis_sharing|v2)[^.]*')
      keep=$(basename "$pt" .pt | grep -oP '[0-9.]+$')
      python convert_svdllm_to_hf_dir.py --input "$pt" \\
          --output "checkpoints/svdllm/llama31_8b/hf_${tag}_${keep}"
  done
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


# ── helpers ────────────────────────────────────────────────────────────────────

def _getattr2(obj, *names):
    """Return the first attribute that exists on obj, or None."""
    for name in names:
        v = getattr(obj, name, None)
        if v is not None:
            return v
    return None


def _merge(u: nn.Linear, v: nn.Linear, device: str = "cpu") -> torch.Tensor:
    """Compute merged weight W = u.weight @ v.weight  (no grad).

    If device is a CUDA device, the matmul is performed on that device and the
    result is immediately moved back to CPU.  The input modules stay on CPU.
    """
    with torch.no_grad():
        uw = u.weight.float().to(device)
        vw = v.weight.float().to(device)
        result = uw @ vw
        return result.cpu()


# ── SVD-LLM projection map ─────────────────────────────────────────────────────
#
# Standard HF name   →  (u_attr, v_attr)  on the SVD module
# Attention (self_attn):
#   q_proj            →  (q_u_proj, q_v_proj)
#   k_proj            →  (k_u_proj, k_v_proj)
#   v_proj            →  (v_u_proj, v_v_proj)
#   o_proj            →  (o_u_proj / out_u_proj,  o_v_proj / out_v_proj)
# MLP:
#   gate_proj         →  (gate_u_proj, gate_v_proj)
#   up_proj           →  (up_u_proj,   up_v_proj)
#   down_proj         →  (down_u_proj, down_v_proj)

_ATTN_PROJ_MAP = {
    "q_proj":  ("q_u_proj",   "q_v_proj"),
    "k_proj":  ("k_u_proj",   "k_v_proj"),
    "v_proj":  ("v_u_proj",   "v_v_proj"),
    # o_proj: two possible naming conventions across SVD-LLM versions
    "o_proj":  (("o_u_proj", "out_u_proj"), ("o_v_proj", "out_v_proj")),
}

_MLP_PROJ_MAP = {
    "gate_proj": ("gate_u_proj", "gate_v_proj"),
    "up_proj":   ("up_u_proj",   "up_v_proj"),
    "down_proj": ("down_u_proj", "down_v_proj"),
}


def _build_merged_state_dict(svd_model: nn.Module, compute_device: str = "cpu") -> dict:
    """
    Walk the SVD-LLM model and produce a state dict with standard HF key names.

    Non-SVD parameters (embed_tokens, norms, lm_head, …) are copied unchanged.
    SVD projection pairs are merged: W = u @ v.
    """
    merged = {}
    sd = svd_model.state_dict()

    # Collect all SVD key prefixes so we can skip them during the passthrough loop
    svd_keys = set()

    layers = getattr(getattr(svd_model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Could not find model.layers — is this a LLaMA-style model?")

    n_layers = len(layers)

    for li, layer in enumerate(layers):
        prefix = f"model.layers.{li}"

        # ── Attention projections ──────────────────────────────────────────────
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise ValueError(f"Layer {li} has no self_attn")

        for hf_name, (u_names, v_names) in _ATTN_PROJ_MAP.items():
            u_names = (u_names,) if isinstance(u_names, str) else u_names
            v_names = (v_names,) if isinstance(v_names, str) else v_names
            u_mod = _getattr2(attn, *u_names)
            v_mod = _getattr2(attn, *v_names)

            if u_mod is None or v_mod is None:
                # Not factorized in this checkpoint (e.g. basis-sharing may skip some)
                # Try to copy the original weight directly if it exists
                orig = _getattr2(attn, hf_name)
                if orig is not None:
                    merged[f"{prefix}.self_attn.{hf_name}.weight"] = orig.weight.float()
                continue

            merged[f"{prefix}.self_attn.{hf_name}.weight"] = _merge(u_mod, v_mod, compute_device)
            # Mark SVD sub-keys as handled
            for uname in u_names:
                svd_keys.add(f"{prefix}.self_attn.{uname}.weight")
            for vname in v_names:
                svd_keys.add(f"{prefix}.self_attn.{vname}.weight")

        # ── MLP projections ────────────────────────────────────────────────────
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError(f"Layer {li} has no mlp")

        for hf_name, (u_name, v_name) in _MLP_PROJ_MAP.items():
            u_mod = getattr(mlp, u_name, None)
            v_mod = getattr(mlp, v_name, None)

            if u_mod is None or v_mod is None:
                orig = getattr(mlp, hf_name, None)
                if orig is not None:
                    merged[f"{prefix}.mlp.{hf_name}.weight"] = orig.weight.float()
                continue

            merged[f"{prefix}.mlp.{hf_name}.weight"] = _merge(u_mod, v_mod, compute_device)
            svd_keys.add(f"{prefix}.mlp.{u_name}.weight")
            svd_keys.add(f"{prefix}.mlp.{v_name}.weight")

    # ── Passthrough: all non-SVD parameters ───────────────────────────────────
    for key, val in sd.items():
        if key in svd_keys:
            continue
        if key in merged:
            continue
        # Skip cache/internal tensors
        if any(s in key for s in ("_flashsvd", "_decode_", "_dual_split",
                                   "_flashsvd_graph", "inv_freq")):
            continue
        merged[key] = val.float()

    return merged


def _load_checkpoint(path: str, map_location: str = "cpu"):
    """Load SVD-LLM .pt with PyTorch>=2.6 compatibility."""
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        obj = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("tokenizer")
    if hasattr(obj, "forward"):
        return obj, None
    raise ValueError(f"Unrecognized checkpoint format: {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",  required=True, help="Path to SVD-LLM .pt checkpoint")
    ap.add_argument("--output", required=True, help="Output HF model directory")
    ap.add_argument("--dtype",  default=None,
                    choices=["fp32", "fp16", "bf16"],
                    help="Cast weights before saving (default: keep original dtype)")
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index to use for merging (e.g. 0); default: CPU")
    args = ap.parse_args()

    compute_device = f"cuda:{args.gpu}" if (args.gpu is not None and torch.cuda.is_available()) else "cpu"

    # ── add SVD-LLM to sys.path so SVD_LlamaAttention etc. can be unpickled ──
    _here = os.path.dirname(os.path.abspath(__file__))
    for _p in [_here, os.path.join(_here, "flashsvd_component")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    print(f"Loading {args.input} (CPU, matmul on {compute_device}) ...")
    model, tokenizer = _load_checkpoint(args.input, map_location="cpu")
    model.eval()

    print("Merging low-rank projections ...")
    merged_sd = _build_merged_state_dict(model, compute_device=compute_device)
    n_merged = sum(
        1 for k in merged_sd
        if any(k.endswith(f".{p}.weight")
               for p in ("q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"))
    )
    print(f"  merged {n_merged} projection weights")

    # ── Build a fresh standard HF model and load the merged weights ───────────
    print("Building standard HF model from config ...")
    from transformers import LlamaForCausalLM, AutoConfig

    config = model.config
    # Remove SVD-specific config attributes that would confuse a fresh model
    for attr in ("flash_svd_use_lowrank_cache", "flash_svd_lowrank_rank"):
        if hasattr(config, attr):
            delattr(config, attr)

    # Create model on meta device to avoid double-allocating 16GB
    try:
        from accelerate import init_empty_weights
        with init_empty_weights():
            hf_model = LlamaForCausalLM(config)
        hf_model = hf_model.to_empty(device="cpu")
    except ImportError:
        # accelerate not installed — allocate normally (needs ~16 GB RAM for 8B)
        print("  [note] accelerate not found; allocating full model in RAM")
        hf_model = LlamaForCausalLM(config)

    missing, unexpected = hf_model.load_state_dict(merged_sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    if args.dtype is not None:
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        hf_model = hf_model.to(dtype_map[args.dtype])
        print(f"  cast to {args.dtype}")

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving HF model to {args.output} ...")
    hf_model.save_pretrained(args.output)

    if tokenizer is not None:
        tokenizer.save_pretrained(args.output)
        print("  tokenizer saved")
    else:
        # Reload tokenizer from model_id
        model_id = getattr(config, "_name_or_path", None)
        if model_id:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                tok.save_pretrained(args.output)
                print(f"  tokenizer reloaded from {model_id} and saved")
            except Exception as e:
                print(f"  [warn] could not save tokenizer: {e}")

    print("Done.")
    print(f"  Load with: AutoModelForCausalLM.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
