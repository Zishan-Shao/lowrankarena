"""
ASVDFlashLlamaAttention: drop-in replacement for LlamaAttention in ASVD-compressed models.

Mirrors SVD-LLM's SVD_LlamaAttention decode path:
  - Prefill (q_len > 1): standard SVDLinear forward (unchanged)
  - Decode  (q_len == 1, FlashSVDDenseKVCache): kernel reconstruct + FA2 with internal RoPE

Weight layout (SVDLinear with sigma_fuse="UV"):
  BLinear.weight: [R, in_features]   -- down-projection (input → rank)
  ALinear.weight: [out_features, R]  -- up-projection   (rank → output)

Kernel input layout:
  Pq: [B, R]       -- BLinear(x).squeeze(1)
  Vq: [H, R, dh]  -- ALinear.weight.view(H, dh, R).permute(0,2,1)
"""
from __future__ import annotations

import inspect
import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Locate repo root
_HERE = os.path.dirname(os.path.abspath(__file__))          # baselines/ASVD/modules/
_ASVD = os.path.dirname(_HERE)                              # baselines/ASVD/
_REPO = os.path.dirname(os.path.dirname(_ASVD))             # lowrankarena/
_SVDLLM = os.path.join(os.path.dirname(_ASVD), "SVD-LLM")  # baselines/SVD-LLM/

for _p in [_REPO, _ASVD, _SVDLLM]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modules.svd_linear import SVDLinear

# FlashSVD dense cache (from SVD-LLM)
try:
    from flashsvd_component.dense_cache import FlashSVDDenseKVCache
    _DenseKVCache = FlashSVDDenseKVCache
except Exception:
    _DenseKVCache = None

# FA2
try:
    from flash_attn import flash_attn_with_kvcache as _fa2_kvcache
    _HAS_FA2 = True
except Exception:
    _fa2_kvcache = None
    _HAS_FA2 = False

# FlashSVD reconstruct kernel
try:
    from src.kernels.decoder.flashsvdropeattn_dense_decode import reconstruct_qkv_token_shared
    _HAS_KERNEL = True
except Exception:
    _HAS_KERNEL = False


def _is_svd(m) -> bool:
    return isinstance(m, SVDLinear)


def _rank(m: SVDLinear) -> int:
    return int(m.BLinear.weight.shape[0])


def _vmat(proj: SVDLinear, num_heads: int, head_dim: int) -> torch.Tensor:
    """Return V matrix [H, R, dh] for kernel."""
    R = _rank(proj)
    return proj.ALinear.weight.view(num_heads, head_dim, R).permute(0, 2, 1).contiguous()


class ASVDFlashLlamaAttention(nn.Module):
    """
    Drop-in replacement for LlamaAttention in ASVD models.
    Reconstructed from an existing LlamaAttention module whose q/k/v projections
    are SVDLinear layers.
    """

    def __init__(self, orig_attn: nn.Module):
        super().__init__()
        # Copy all sub-modules and parameters from original attention
        self.__dict__.update(orig_attn.__dict__)
        # Store reference for fallback
        self._orig_forward = orig_attn.__class__.forward
        # Precompute V matrices [H, R, dh] once — avoids 96 contiguous() allocs per token
        self._Vq = self._Vk = self._Vv = None
        self._precompute_vmats()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _precompute_vmats(self) -> None:
        q, k, v = self.q_proj, self.k_proj, self.v_proj
        if not (_is_svd(q) and _is_svd(k) and _is_svd(v)):
            return
        if not (_rank(q) == _rank(k) == _rank(v)):
            return
        H  = self.num_heads
        Hk = getattr(self, "num_key_value_heads", H)
        dh = self.head_dim
        self._Vq = _vmat(q, H,  dh)
        self._Vk = _vmat(k, Hk, dh)
        self._Vv = _vmat(v, Hk, dh)

    def _can_use_flashsvd_decode(self, hidden_states: torch.Tensor, past_key_value) -> bool:
        if not _HAS_KERNEL or not _HAS_FA2:
            return False
        if _DenseKVCache is None or not isinstance(past_key_value, _DenseKVCache):
            return False
        if not hidden_states.is_cuda or self.training:
            return False
        if int(hidden_states.shape[1]) != 1:
            return False
        q, k, v = self.q_proj, self.k_proj, self.v_proj
        if not (_is_svd(q) and _is_svd(k) and _is_svd(v)):
            return False
        if not (_rank(q) == _rank(k) == _rank(v)):
            return False
        return True

    def _flashsvd_decode(
        self,
        hidden_states: torch.Tensor,
        past_key_value: "FlashSVDDenseKVCache",
        cache_position: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        B = hidden_states.shape[0]
        dh = self.head_dim

        # Rank-space projections: [B, R]
        Pq = self.q_proj.BLinear(hidden_states).squeeze(1)
        Pk = self.k_proj.BLinear(hidden_states).squeeze(1)
        Pv = self.v_proj.BLinear(hidden_states).squeeze(1)

        # V matrices: precomputed in __init__, no allocation here
        Vq, Vk, Vv = self._Vq, self._Vk, self._Vv

        # Reconstruct Q/K/V for current token: [B, H, dh]
        Q_bhd, K_bhd, V_bhd = reconstruct_qkv_token_shared(Pq, Pk, Pv, Vq, Vk, Vv)

        # [B, 1, H, dh] for FA2
        q_bmhd = Q_bhd.unsqueeze(1).contiguous()
        k_bmhd = K_bhd.unsqueeze(1).contiguous()
        v_bmhd = V_bhd.unsqueeze(1).contiguous()

        # Prepare FA2 cache
        seqlen_k = int(past_key_value.get_seq_length())
        smax = int(past_key_value.get_max_cache_shape() or max(seqlen_k + 1, 1))
        k_cache_bmhd, v_cache_bmhd, cache_seqlens = past_key_value.prepare_fa2_step(
            int(self.layer_idx),
            batch_size=B,
            cache_position=cache_position,
        )
        rotary_cos, rotary_sin = past_key_value.get_rope_tables(
            seqlen=smax,
            head_dim=dh,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # FA2 with internal RoPE
        out = _fa2_kvcache(
            q_bmhd,
            k_cache_bmhd,
            v_cache_bmhd,
            k_bmhd,
            v_bmhd,
            cache_seqlens=cache_seqlens,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            causal=True,
        )
        past_key_value.advance_after_fa2(
            int(self.layer_idx),
            q_len=1,
            cache_position=cache_position,
        )

        # [B, 1, H*dh] → output projection
        attn_output = out.reshape(B, 1, self.num_heads * dh)
        attn_output = self.o_proj(attn_output)
        return attn_output

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if self._can_use_flashsvd_decode(hidden_states, past_key_value):
            attn_output = self._flashsvd_decode(hidden_states, past_key_value, cache_position)
            return attn_output, None, past_key_value

        # Fallback: original LlamaAttention forward
        # Filter kwargs to only those accepted by the original (handles older transformers)
        call_kwargs = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        sig = inspect.signature(self._orig_forward)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_var_kw:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}
        return self._orig_forward(self, hidden_states, **call_kwargs)
