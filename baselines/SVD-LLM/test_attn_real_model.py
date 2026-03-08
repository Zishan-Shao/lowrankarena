#!/usr/bin/env python3
"""
Load the actual compressed checkpoint and compare kernel vs fallback
for one attention layer on real data.

Usage:
  python test_attn_real_model.py --model_path /path/to/checkpoint.pt

If kernel matches fallback (rel_fro < 10%), the kernel is correct for
these weights and the PPL problem lies elsewhere.
If kernel mismatches fallback, prints the first diverging layer index.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--max_layers", type=int, default=4,
                    help="only test first N layers (faster)")
args = parser.parse_args()

# ── load model ────────────────────────────────────────────────────────────────
from utils.model_utils import get_model_from_local
print(f"Loading {args.model_path} ...")
model, tokenizer = get_model_from_local(args.model_path)
model.eval()
dev = torch.device("cuda")
model = model.to(dev)
print(f"  dtype={next(model.parameters()).dtype}")

# ── find decoder layers and attention modules ─────────────────────────────────
from flashsvd_component.svd_llama import SVD_LlamaAttention

if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layers = model.model.layers
elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
    layers = model.model.decoder.layers
else:
    raise RuntimeError("Cannot find decoder layers")

print(f"  total layers: {len(layers)}, testing first {args.max_layers}")

from src.kernels.decoder.flashsvdropeattn_v16 import (
    PackedFactors, build_rope_tables,
    flashsvd_attn_packed, reference_packed_fp32,
)

def rel_fro(a, b):
    return (torch.linalg.norm((a - b).float()) /
            (torch.linalg.norm(b.float()) + 1e-12)).item()

# ── synthetic hidden_states (same shape as real inputs) ──────────────────────
# Use small S to keep test fast; the weight values are real.
B, S = 1, 64
torch.manual_seed(0)

for layer_idx in range(min(args.max_layers, len(layers))):
    layer = layers[layer_idx]
    attn = layer.self_attn

    if not isinstance(attn, SVD_LlamaAttention):
        print(f"  layer {layer_idx}: not SVD_LlamaAttention, skipping")
        continue

    hidden_size = attn.hidden_size
    H  = attn.num_heads
    dh = attn.head_dim
    R  = attn.q_v_proj.out_features

    x = torch.randn(B, S, hidden_size, device=dev,
                    dtype=next(attn.parameters()).dtype)

    # ── kernel path ───────────────────────────────────────────────────────────
    with torch.no_grad():
        Vq = attn._eff_weight(attn.q_u_proj).view(H, dh, R).permute(0,2,1).contiguous()
        Vk = attn._eff_weight(attn.k_u_proj).view(H, dh, R).permute(0,2,1).contiguous()
        Vv = attn._eff_weight(attn.v_u_proj).view(H, dh, R).permute(0,2,1).contiguous()

        Pq = attn.q_v_proj(x).unsqueeze(2).expand(B, S, H, R)
        Pk = attn.k_v_proj(x).unsqueeze(2).expand(B, S, H, R)
        Pv = attn.v_v_proj(x).unsqueeze(2).expand(B, S, H, R)

        cos, sin = attn._get_rope_tables(S, dev, x.dtype)

        f = PackedFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv)
        O_kernel = flashsvd_attn_packed(f, cos, sin, causal=True)   # [B,S,H,dh]

    # ── reference path (fp32 ground truth using same P/V) ────────────────────
    with torch.no_grad():
        O_ref = reference_packed_fp32(f, cos, sin, causal=True,
                                      window_left=-1, window_right=-1)

    rf = rel_fro(O_kernel, O_ref)
    fin = torch.isfinite(O_kernel).all().item()
    ma  = (O_kernel.float() - O_ref.float()).abs().max().item()

    status = "PASS" if (fin and rf < 1e-1) else "FAIL"
    print(f"  [{status}] layer {layer_idx:2d}  R={R:4d}  dtype={x.dtype}  "
          f"finite={fin}  max_abs={ma:.3e}  rel_fro={rf:.3e}")

    # Extra: check for scale anomaly in real weights
    Pq_mean = attn.q_v_proj(x).abs().float().mean().item()
    Vq_norm = Vq.abs().float().mean().item()
    print(f"           Pq abs_mean={Pq_mean:.3e}  Vq abs_mean={Vq_norm:.3e}  "
          f"O_kernel abs_mean={O_kernel.abs().float().mean():.3e}")

print("\nDone.")
