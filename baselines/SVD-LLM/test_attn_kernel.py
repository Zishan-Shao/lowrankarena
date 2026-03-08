#!/usr/bin/env python3
"""
Diagnostic: compare flashsvd_attn_packed vs reference for SVD-LLM-style inputs.

Three tests:
  T1  kernel's own self-check (contiguous P, random per-head) — baseline sanity
  T2  stride-0 P (expanded like SVD-LLM): kernel vs reference_packed_fp32
  T3  full SVD-LLM forward: kernel path vs fallback (q_u_proj o q_v_proj + SDPA)

Run on server:
  python test_attn_kernel.py

Expected output if kernel is correct:
  T1 ... rel_fro < 1e-2
  T2 ... rel_fro < 1e-2
  T3 ... rel_fro < 1e-2

If any T shows large error, that test identifies the bug.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))  # repo root

import math
import torch
import torch.nn.functional as F

# ── kernel imports ────────────────────────────────────────────────────────────
from src.kernels.decoder.flashsvdropeattn_v16 import (
    PackedFactors,
    build_rope_tables,
    flashsvd_attn_packed,
    reference_packed_fp32,
)

device = torch.device("cuda")
dtype  = torch.bfloat16   # model runs in bf16 on A100

torch.manual_seed(42)

# ── LLaMA-7b dims ────────────────────────────────────────────────────────────
B   = 1
S   = 64          # small enough for O(S^2) reference
H   = 32
Hk  = 32          # no GQA
Dh  = 128
# ratio=0.5 → low_rank = int(4096 * 0.5 / 2) = 1024
R   = 1024

def rel_fro(a, b):
    return (torch.linalg.norm((a - b).float()) /
            (torch.linalg.norm(b.float()) + 1e-12)).item()

def check(label, O, Oref):
    O   = O.float()
    Oref = Oref.float()
    fin  = torch.isfinite(O).all().item()
    ma   = (O - Oref).abs().max().item()
    rf   = rel_fro(O, Oref)
    status = "PASS" if (fin and rf < 5e-2) else "FAIL"
    print(f"  [{status}] {label:45s}  finite={fin}  max_abs={ma:.3e}  rel_fro={rf:.3e}")

cos, sin = build_rope_tables(S, Dh, base=10000.0, device=device, dtype=dtype)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T1: kernel self-check (contiguous P, per-head independent) ===")
# Exact same setup as the kernel's own --check mode
Pq_c = torch.randn(B, S, H,  R, device=device, dtype=dtype).contiguous()
Pk_c = torch.randn(B, S, Hk, R, device=device, dtype=dtype).contiguous()
Pv_c = torch.randn(B, S, Hk, R, device=device, dtype=dtype).contiguous()
Vq_c = torch.randn(H,  R, Dh, device=device, dtype=dtype).contiguous()
Vk_c = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
Vv_c = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()

f1 = PackedFactors(Pq=Pq_c, Pk=Pk_c, Pv=Pv_c, Vq=Vq_c, Vk=Vk_c, Vv=Vv_c)
O1     = flashsvd_attn_packed(f1, cos, sin, causal=True)
O1ref  = reference_packed_fp32(f1, cos, sin, causal=True, window_left=-1, window_right=-1)
check("contiguous P  vs reference_fp32", O1, O1ref)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T2: stride-0 P (expanded, like SVD-LLM) vs reference_fp32 ===")
# In SVD-LLM, q_v_proj / k_v_proj / v_v_proj each output [B,S,R] shared for all heads
pq_base = torch.randn(B, S, R, device=device, dtype=dtype)
pk_base = torch.randn(B, S, R, device=device, dtype=dtype)
pv_base = torch.randn(B, S, R, device=device, dtype=dtype)

# stride-0 expand — matches svd_llama.py exactly
Pq_e = pq_base.unsqueeze(2).expand(B, S, H,  R)   # stride(2)=0
Pk_e = pk_base.unsqueeze(2).expand(B, S, Hk, R)
Pv_e = pv_base.unsqueeze(2).expand(B, S, Hk, R)

# Vq same as T1 (independent per head)
f2 = PackedFactors(Pq=Pq_e, Pk=Pk_e, Pv=Pv_e, Vq=Vq_c, Vk=Vk_c, Vv=Vv_c)
O2     = flashsvd_attn_packed(f2, cos, sin, causal=True)
O2ref  = reference_packed_fp32(f2, cos, sin, causal=True, window_left=-1, window_right=-1)
check("stride-0 P    vs reference_fp32", O2, O2ref)

# Also compare T2 against T1 (they should differ because P data differs)
print(f"  [info ] T1 vs T2 rel_fro (should be non-zero): {rel_fro(O1, O2):.3e}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T3: full SVD-LLM forward — kernel vs fallback (matmul + SDPA) ===")
# Simulate SVD_LlamaAttention with weight matrices
# q_u_proj.weight: [H*dh, R] = [4096, 1024]
# q_v_proj.weight: [R, hidden] = [1024, 4096]

hidden = H * Dh   # 4096
Wu_q = torch.randn(H * Dh, R, device=device, dtype=dtype)  # q_u_proj.weight
Wu_k = torch.randn(H * Dh, R, device=device, dtype=dtype)
Wu_v = torch.randn(H * Dh, R, device=device, dtype=dtype)
Wv_q = torch.randn(R, hidden, device=device, dtype=dtype)   # q_v_proj.weight
Wv_k = torch.randn(R, hidden, device=device, dtype=dtype)
Wv_v = torch.randn(R, hidden, device=device, dtype=dtype)

x = torch.randn(B, S, hidden, device=device, dtype=dtype)

# ── Fallback path (identical to svd_llama.py else-branch) ────────────────────
def apply_rope_half(x_bshd, cos_s_half, sin_s_half):
    """Standard split-half RoPE, matching LlamaRotaryEmbedding."""
    half = x_bshd.shape[-1] // 2
    x0 = x_bshd[..., :half]
    x1 = x_bshd[..., half:]
    c  = cos_s_half.unsqueeze(0).unsqueeze(0)   # [1,1,S,half]
    s  = sin_s_half.unsqueeze(0).unsqueeze(0)
    return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)

# Fallback: reconstruct dense Q,K,V then SDPA
Pq_fb = F.linear(x, Wv_q)           # [B,S,R]
Pk_fb = F.linear(x, Wv_k)
Pv_fb = F.linear(x, Wv_v)

Q_fb = F.linear(Pq_fb, Wu_q).view(B, S, H, Dh)   # [B,S,H,Dh]
K_fb = F.linear(Pk_fb, Wu_k).view(B, S, H, Dh)
V_fb = F.linear(Pv_fb, Wu_v).view(B, S, H, Dh)

Q_fb = apply_rope_half(Q_fb, cos, sin)
K_fb = apply_rope_half(K_fb, cos, sin)

Q_t = Q_fb.transpose(1, 2)    # [B,H,S,Dh]
K_t = K_fb.transpose(1, 2)
V_t = V_fb.transpose(1, 2)
O_fallback = F.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=True)
O_fallback = O_fallback.transpose(1, 2)  # [B,S,H,Dh]

# ── Kernel path ───────────────────────────────────────────────────────────────
# Vq = Wu_q.view(H, Dh, R).permute(0,2,1).contiguous()  — matches svd_llama.py
Vq_k = Wu_q.view(H, Dh, R).permute(0, 2, 1).contiguous()   # [H,R,Dh]
Vk_k = Wu_k.view(H, Dh, R).permute(0, 2, 1).contiguous()
Vv_k = Wu_v.view(H, Dh, R).permute(0, 2, 1).contiguous()

Pq_k = F.linear(x, Wv_q).unsqueeze(2).expand(B, S, H, R)   # stride-0 on H
Pk_k = F.linear(x, Wv_k).unsqueeze(2).expand(B, S, H, R)
Pv_k = F.linear(x, Wv_v).unsqueeze(2).expand(B, S, H, R)

f3 = PackedFactors(Pq=Pq_k, Pk=Pk_k, Pv=Pv_k, Vq=Vq_k, Vk=Vk_k, Vv=Vv_k)
O_kernel = flashsvd_attn_packed(f3, cos, sin, causal=True)   # [B,S,H,Dh]

check("SVD-LLM kernel vs fallback (SDPA)", O_kernel, O_fallback)

# Extra debug: check if output is all-zero (l_i=0 bug)
print(f"  [info ] kernel output abs mean : {O_kernel.abs().float().mean():.3e}")
print(f"  [info ] fallback output abs mean: {O_fallback.abs().float().mean():.3e}")
print(f"  [info ] kernel finite           : {torch.isfinite(O_kernel).all().item()}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T3b: T3 but with contiguous P (test if stride-0 is the issue) ===")
# Make P fully contiguous (copy) — if this passes but T3 fails, stride-0 is the bug
Pq_cont = Pq_k.contiguous()
Pk_cont = Pk_k.contiguous()
Pv_cont = Pv_k.contiguous()
f3b = PackedFactors(Pq=Pq_cont, Pk=Pk_cont, Pv=Pv_cont, Vq=Vq_k, Vk=Vk_k, Vv=Vv_k)
O_kernel_cont = flashsvd_attn_packed(f3b, cos, sin, causal=True)
check("SVD-LLM kernel (contig P) vs fallback", O_kernel_cont, O_fallback)
check("kernel stride-0 vs kernel contig     ", O_kernel, O_kernel_cont)

print()
