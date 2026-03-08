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

print("\n=== S=2048 test (PPL eval sequence length) on layer 0 ===")
layer = layers[0]
attn  = layer.self_attn
if isinstance(attn, SVD_LlamaAttention):
    S2   = 2048
    H2   = attn.num_heads
    dh2  = attn.head_dim
    R2   = attn.q_v_proj.out_features
    hid2 = attn.hidden_size
    x2   = torch.randn(1, S2, hid2, device=dev,
                       dtype=next(attn.parameters()).dtype)

    with torch.no_grad():
        Vq2 = attn._eff_weight(attn.q_u_proj).view(H2,dh2,R2).permute(0,2,1).contiguous()
        Vk2 = attn._eff_weight(attn.k_u_proj).view(H2,dh2,R2).permute(0,2,1).contiguous()
        Vv2 = attn._eff_weight(attn.v_u_proj).view(H2,dh2,R2).permute(0,2,1).contiguous()

        Pq2 = attn.q_v_proj(x2).unsqueeze(2).expand(1, S2, H2, R2)
        Pk2 = attn.k_v_proj(x2).unsqueeze(2).expand(1, S2, H2, R2)
        Pv2 = attn.v_v_proj(x2).unsqueeze(2).expand(1, S2, H2, R2)

        cos2, sin2 = attn._get_rope_tables(S2, dev, x2.dtype)

        f2k = PackedFactors(Pq=Pq2, Pk=Pk2, Pv=Pv2, Vq=Vq2, Vk=Vk2, Vv=Vv2)
        O_k2  = flashsvd_attn_packed(f2k, cos2, sin2, causal=True)

        # reference: use fp32 ground truth
        O_r2  = reference_packed_fp32(f2k, cos2, sin2, causal=True,
                                      window_left=-1, window_right=-1)

    rf2  = rel_fro(O_k2, O_r2)
    fin2 = torch.isfinite(O_k2).all().item()
    ma2  = (O_k2.float() - O_r2.float()).abs().max().item()
    status2 = "PASS" if (fin2 and rf2 < 1e-1) else "FAIL"
    print(f"  [{status2}] S=2048 layer 0  finite={fin2}  "
          f"max_abs={ma2:.3e}  rel_fro={rf2:.3e}")
    print(f"           O_kernel abs_mean={O_k2.abs().float().mean():.3e}  "
          f"max={O_k2.abs().float().max():.3e}")
    # check per-position output scale (first vs last token)
    print(f"           first-token output norm : {O_k2[0,0].float().norm():.3e}")
    print(f"           last-token  output norm : {O_k2[0,-1].float().norm():.3e}")
    print(f"           any nan/inf in output   : {(~torch.isfinite(O_k2)).any().item()}")

print("\nDone.")
