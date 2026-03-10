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

from src.kernels.decoder.flashsvdswiglu_v15 import flashsvd_ffn_swiglu
from src.kernels.decoder.flashsvdropeattn_dense_decode import reconstruct_qkv_token_shared

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

    gate_proj = mlp_module.gate_proj
    up_proj   = mlp_module.up_proj
    down_proj = mlp_module.down_proj

    # Pre-check kernel compatibility; fall back to original if not met.
    if not (_is_svd(gate_proj) and _is_svd(up_proj) and _is_svd(down_proj)):
        return original_forward
    if _rank(gate_proj) != _rank(up_proj):
        return original_forward

    # Precompute static weight views/concatenations once (not per forward call).
    D  = up_proj.ALinear.weight.shape[0]    # intermediate_size
    V1 = torch.cat([up_proj.ALinear.weight.T, gate_proj.ALinear.weight.T], dim=1).contiguous()
    U2 = down_proj.BLinear.weight.T.contiguous()
    V2 = down_proj.ALinear.weight.T.contiguous()
    b1 = torch.zeros(2 * D, device=V1.device, dtype=V1.dtype)
    b2_bias = getattr(down_proj.ALinear, 'bias', None)
    b2 = b2_bias if b2_bias is not None else torch.zeros(V2.shape[1], device=V2.device, dtype=V2.dtype)

    def new_forward(x):
        if not x.is_cuda:
            return original_forward(x)
        # Prefill (L > 4): skip kernel to avoid Triton autotune on large L
        if x.shape[1] > 4:
            return original_forward(x)

        # P = x @ BLinear_up^T  →  [B, L, R1]
        P = up_proj.BLinear(x)
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


# ── Import ASVDFlashLlamaAttention ────────────────────────────────────────────
try:
    from modules.flashsvd_llama_attn import ASVDFlashLlamaAttention
    _HAS_FLASH_ATTN_MODULE = True
except Exception:
    _HAS_FLASH_ATTN_MODULE = False


# ── Public API ────────────────────────────────────────────────────────────────

def apply_flashsvd_to_asvd_model(model: nn.Module) -> nn.Module:
    """Patch an ASVD-compressed LlamaForCausalLM to use FlashSVD kernels.

    Attention: replaced with ASVDFlashLlamaAttention (proper module, not monkey-patch).
    MLP: forward method patched with flashsvd_ffn_swiglu.

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

        if attn is not None and _HAS_FLASH_ATTN_MODULE:
            try:
                flash_attn = ASVDFlashLlamaAttention(attn)
                # Store original for restore
                layer._orig_self_attn = attn
                layer.self_attn = flash_attn
                patched_attn += 1
            except Exception as e:
                print(f"[flashsvd_wrapper] layer {i} attention skip: {e}")

        if mlp is not None:
            try:
                new_fwd = _make_mlp_forward(mlp)
                mlp.forward = new_fwd
                patched_mlp += 1
            except Exception as e:
                print(f"[flashsvd_wrapper] layer {i} MLP skip: {e}")

    print(f"[flashsvd_wrapper] patched {patched_attn} attention layers, {patched_mlp} MLP layers")

    # ── one-time diagnostics ──────────────────────────────────────────────────
    try:
        from modules.flashsvd_llama_attn import _HAS_FA2, _HAS_KERNEL, _DenseKVCache, _is_svd, _rank
        print(f"[flashsvd_wrapper] _HAS_FA2={_HAS_FA2}  _HAS_KERNEL={_HAS_KERNEL}  _DenseKVCache={_DenseKVCache}")
        attn0 = layers[0].self_attn
        q, k, v = attn0.q_proj, attn0.k_proj, attn0.v_proj
        print(f"[flashsvd_wrapper] layer0: q_is_svd={_is_svd(q)} k_is_svd={_is_svd(k)} v_is_svd={_is_svd(v)}")
        if _is_svd(q) and _is_svd(k) and _is_svd(v):
            print(f"[flashsvd_wrapper] layer0 ranks: Rq={_rank(q)} Rk={_rank(k)} Rv={_rank(v)}")
    except Exception as _e:
        print(f"[flashsvd_wrapper] diagnostics error: {_e}")

    return model


def remove_flashsvd_from_asvd_model(model: nn.Module) -> nn.Module:
    """Remove FlashSVD patches (restore original modules/forward methods).

    Must be called before saving/serializing the model.
    """
    try:
        layers = model.model.layers
    except AttributeError:
        return model
    for layer in layers:
        # Restore original attention module if replaced
        if hasattr(layer, '_orig_self_attn'):
            layer.self_attn = layer._orig_self_attn
            del layer._orig_self_attn
        # Restore MLP forward (delete instance-level override)
        mlp = getattr(layer, 'mlp', None)
        if mlp is not None and 'forward' in mlp.__dict__:
            try:
                del mlp.__dict__['forward']
            except (KeyError, AttributeError):
                pass
    return model
