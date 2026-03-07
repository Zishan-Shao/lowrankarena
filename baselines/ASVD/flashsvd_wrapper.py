"""
FlashSVD wrapper for ASVD-compressed LLaMA models.

After ASVD compresses a LlamaForCausalLM (replacing nn.Linear with SVDLinear),
call `apply_flashsvd_to_asvd_model(model)` to monkey-patch the attention and MLP
forward methods to use FlashSVD Triton kernels.

Compatibility conditions (checked at runtime):
  Attention : q/k/v projections are SVDLinear AND R_q == R_k == R_v
  MLP       : gate/up projections are SVDLinear AND R_gate == R_up

If either condition is not met for a layer, that layer falls back to the
original forward method unchanged.

Two execution paths per layer:
  - decode  (q_len == 1, use_cache=True)  → reconstruct_qkv_token_shared + SDPA
  - prefill (q_len > 1)                   → standard low-rank matmul (SVDLinear)

The MLP uses flashsvd_ffn_swiglu which auto-selects the decode kernel when
B <= 4 and L <= 4.
"""

from __future__ import annotations

import math
import os
import sys
import importlib.util
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── locate repo root and register it in sys.path ─────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))          # baselines/ASVD/
_REPO = os.path.dirname(os.path.dirname(_HERE))             # lowrankarena/
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ── import kernels from src/kernels ──────────────────────────────────────────
from src.kernels.decoder.flashsvdswiglu_v15 import flashsvd_ffn_swiglu                          # noqa: E402
from src.kernels.decoder.flashsvdropeattn_dense_decode import reconstruct_qkv_token_shared  # noqa: E402

try:
    from flash_attn import flash_attn_with_kvcache as _flash_attn_kvcache
    _HAS_FLASH_ATTN = True
except Exception:
    _HAS_FLASH_ATTN = False

# ── SVDLinear detection ───────────────────────────────────────────────────────
try:
    from modules.svd_linear import SVDLinear
except Exception:
    sys.path.insert(0, _HERE)
    from modules.svd_linear import SVDLinear


def _is_svd(m) -> bool:
    return isinstance(m, SVDLinear)


def _rank(m: SVDLinear) -> int:
    return int(m.BLinear.weight.shape[0])


# ── Attention decode helper ───────────────────────────────────────────────────

def _apply_rope_to_bhsd(x_bhsd, cos_bsd, sin_bsd):
    """Apply RoPE to x [B, H, S, Dh] using cos/sin [B, 1, S, Dh]."""
    x1 = x_bhsd[..., : x_bhsd.shape[-1] // 2]
    x2 = x_bhsd[..., x_bhsd.shape[-1] // 2 :]
    c = cos_bsd
    s = sin_bsd
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# ── MLP forward patcher ───────────────────────────────────────────────────────

def _make_mlp_forward(mlp_module):
    """Return a patched forward() that uses flashsvd_ffn_swiglu on CUDA."""
    original_forward = mlp_module.forward

    def new_forward(x):
        if not x.is_cuda:
            return original_forward(x)

        gate_proj = mlp_module.gate_proj
        up_proj   = mlp_module.up_proj
        down_proj = mlp_module.down_proj

        if not (_is_svd(gate_proj) and _is_svd(up_proj) and _is_svd(down_proj)):
            return original_forward(x)
        if _rank(gate_proj) != _rank(up_proj):
            return original_forward(x)

        R1   = _rank(up_proj)
        D    = up_proj.ALinear.weight.shape[0]    # intermediate_size

        # P = x @ BLinear_up^T  →  [B, L, R1]
        P = up_proj.BLinear(x)

        # V1 = [R1, 2*D]:  cols 0..D-1 = up factors, D..2D-1 = gate factors
        V1u = up_proj.ALinear.weight.T      # [R1, D]
        V1v = gate_proj.ALinear.weight.T    # [R1, D]  (gate)
        V1  = torch.cat([V1u, V1v], dim=1)  # [R1, 2D]

        # Down path
        R2   = _rank(down_proj)
        U2   = down_proj.BLinear.weight.T   # [D, R2]  (BLinear: [R2, D])
        V2   = down_proj.ALinear.weight.T   # [R2, H]  (ALinear: [H, R2])

        B, L = x.shape[0], x.shape[1]
        b1 = torch.zeros(2 * D, device=x.device, dtype=x.dtype)
        b2_bias = getattr(down_proj.ALinear, 'bias', None)
        b2 = b2_bias if b2_bias is not None else torch.zeros(V2.shape[1], device=x.device, dtype=x.dtype)

        return flashsvd_ffn_swiglu(P, V1, U2, V2, b1, b2)

    return new_forward


# ── Attention forward patcher ─────────────────────────────────────────────────

def _make_attn_forward(attn_module):
    """Return a patched forward() for LlamaAttention that uses FlashSVD decode."""
    original_forward = attn_module.forward

    def new_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return original_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs,
            )

        q_proj = attn_module.q_proj
        k_proj = attn_module.k_proj
        v_proj = attn_module.v_proj
        o_proj = attn_module.o_proj

        # Only use FlashSVD if Q/K/V are SVDLinear with equal rank
        if not (_is_svd(q_proj) and _is_svd(k_proj) and _is_svd(v_proj)):
            return original_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs,
            )
        R_q, R_k, R_v = _rank(q_proj), _rank(k_proj), _rank(v_proj)
        if not (R_q == R_k == R_v):
            return original_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs,
            )

        # Only accelerate decode (q_len == 1); fall back for prefill
        bsz, q_len, _ = hidden_states.size()
        if q_len != 1 or not (past_key_value is not None or use_cache):
            return original_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs,
            )

        R  = R_q
        H  = attn_module.num_heads
        dh = attn_module.head_dim
        # Hk may differ for GQA; ASVD typically uses same H for all
        Hk = getattr(attn_module, 'num_key_value_heads', H)

        # V matrices  [H, R, dh]
        Vq = q_proj.ALinear.weight.view(H,  dh, R).permute(0, 2, 1).contiguous()
        Vk = k_proj.ALinear.weight.view(Hk, dh, R).permute(0, 2, 1).contiguous()
        Vv = v_proj.ALinear.weight.view(Hk, dh, R).permute(0, 2, 1).contiguous()

        # P vectors for current token  [B, R]
        Pq = q_proj.BLinear(hidden_states).squeeze(1)  # [B, R]
        Pk = k_proj.BLinear(hidden_states).squeeze(1)
        Pv = v_proj.BLinear(hidden_states).squeeze(1)

        # Triton fused reconstruct  →  Q/K/V  [B, H, dh]
        Q_tok, K_tok, V_tok = reconstruct_qkv_token_shared(Pq, Pk, Pv, Vq, Vk, Vv)

        # [B, H, 1, dh]
        Q_4d = Q_tok.unsqueeze(2)
        K_4d = K_tok.unsqueeze(2)
        V_4d = V_tok.unsqueeze(2)

        # RoPE
        kv_offset = past_key_value[0].shape[-2] if past_key_value is not None else 0
        kv_seq_len = 1 + kv_offset
        if position_ids is None:
            position_ids = torch.tensor([[kv_offset]], device=hidden_states.device)
        cos, sin = attn_module.rotary_emb(V_4d, seq_len=kv_seq_len)
        # cos/sin: [1, 1, kv_seq_len, dh] → gather at position
        gather_idx = position_ids[:, None, :, None].expand(bsz, 1, 1, cos.shape[-1])
        cos_tok = torch.gather(cos.expand(bsz, -1, -1, -1), 2, gather_idx)
        sin_tok = torch.gather(sin.expand(bsz, -1, -1, -1), 2, gather_idx)
        Q_4d = _apply_rope_to_bhsd(Q_4d, cos_tok, sin_tok)
        K_4d = _apply_rope_to_bhsd(K_4d, cos_tok, sin_tok)

        # KV cache
        if past_key_value is not None:
            K_4d = torch.cat([past_key_value[0], K_4d], dim=2)
            V_4d = torch.cat([past_key_value[1], V_4d], dim=2)
        past_key_value = (K_4d, V_4d) if use_cache else None

        # Attention
        if _HAS_FLASH_ATTN:
            attn_out = _flash_attn_kvcache(
                Q_4d.transpose(1, 2),
                K_4d.transpose(1, 2),
                V_4d.transpose(1, 2),
                causal=True,
            )
            attn_output = attn_out.reshape(bsz, q_len, H * dh)
        else:
            scale = math.sqrt(dh)
            attn_w = torch.matmul(Q_4d, K_4d.transpose(-1, -2)) / scale
            attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(Q_4d.dtype)
            attn_out = torch.matmul(attn_w, V_4d)
            attn_output = attn_out.transpose(1, 2).reshape(bsz, q_len, H * dh)

        # Output projection (SVDLinear or plain Linear)
        attn_output = o_proj(attn_output)
        return attn_output, None, past_key_value

    return new_forward


# ── Public API ────────────────────────────────────────────────────────────────

def apply_flashsvd_to_asvd_model(model: nn.Module) -> nn.Module:
    """Patch an ASVD-compressed LlamaForCausalLM to use FlashSVD kernels.

    Returns the same model object (modified in-place).
    Layers that don't meet compatibility requirements are left unchanged.
    """
    patched_attn = 0
    patched_mlp  = 0

    # Detect LlamaDecoderLayer children
    try:
        layers = model.model.layers
    except AttributeError:
        raise ValueError("model.model.layers not found — is this a LlamaForCausalLM?")

    for i, layer in enumerate(layers):
        attn = getattr(layer, 'self_attn', None)
        mlp  = getattr(layer, 'mlp', None)

        if attn is not None:
            try:
                new_fwd = _make_attn_forward(attn)
                attn.forward = types.MethodType(new_fwd, attn)
                # Verify at least that attributes exist before counting
                _ = attn.num_heads
                patched_attn += 1
            except Exception as e:
                print(f"[flashsvd_wrapper] layer {i} attention skip: {e}")

        if mlp is not None:
            try:
                new_fwd = _make_mlp_forward(mlp)
                mlp.forward = types.MethodType(new_fwd, mlp)
                patched_mlp += 1
            except Exception as e:
                print(f"[flashsvd_wrapper] layer {i} MLP skip: {e}")

    print(f"[flashsvd_wrapper] patched {patched_attn} attention layers, {patched_mlp} MLP layers")
    return model


def remove_flashsvd_from_asvd_model(model: nn.Module) -> nn.Module:
    """Remove FlashSVD patches (restore original forward methods).

    Must be called before saving/serializing the model.
    """
    try:
        layers = model.model.layers
    except AttributeError:
        return model
    for layer in layers:
        for sub in [getattr(layer, 'self_attn', None), getattr(layer, 'mlp', None)]:
            if sub is not None and hasattr(sub, 'forward'):
                # Delete instance-level override to restore class method
                try:
                    del sub.__dict__['forward']
                except (KeyError, AttributeError):
                    pass
    return model
