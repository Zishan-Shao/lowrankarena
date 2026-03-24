import math
import os
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Match HF RMSNorm behavior: compute in fp32, return in input dtype.
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        hs = hs * self.weight.to(torch.float32)
        return hs.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids: Optional[torch.LongTensor] = None):
    """
    Apply RoPE to q and k using provided cos/sin tables.

    - If position_ids is provided, gather per-token embeddings (old behavior).
    - If not, assume cos/sin already correspond to the positions of q/k.
    """
    if position_ids is not None:
        gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
        gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
        cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
        sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    else:
        # HF >=4.44 path: cos/sin may be [B, M, Hd]; unsqueeze to [B, 1, M, Hd]
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SVD_LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ratio=1,
        compat_ranks: Optional[bool] = None,
    ):
        super().__init__()
        self.ratio = ratio
        if compat_ranks is None:
            compat_ranks = os.getenv('SVDLLM_COMPAT_RANKS', '0') != '0'
        self.compat_ranks = compat_ranks
        if self.compat_ranks:
            # Official SVD-LLM truncated rank: int(m*n*ratio/(m+n))
            low_rank = max(1, int(intermediate_size * hidden_size * ratio / (intermediate_size + hidden_size)))
        else:
            low_rank = max(1, int(min(intermediate_size, hidden_size) * ratio))
        self.gate_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        
        self.down_u_proj = nn.Linear(low_rank, hidden_size, bias=False)
        self.down_v_proj = nn.Linear(intermediate_size, low_rank, bias=False)
        
        self.up_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        # Ensure projection weights follow input dtype to avoid matmul dtype mismatch
        dtype = x.dtype
        casted = getattr(self, "_weights_casted", None)
        if casted != dtype:
            # Cast all linear submodules' parameters to the input dtype in-place
            self.gate_u_proj.to(dtype=dtype)
            self.gate_v_proj.to(dtype=dtype)
            self.down_u_proj.to(dtype=dtype)
            self.down_v_proj.to(dtype=dtype)
            self.up_u_proj.to(dtype=dtype)
            self.up_v_proj.to(dtype=dtype)
            self._weights_casted = dtype
        # TODO: we should change this part by doing inference with FlashSVDSwiGLU kernel
        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))


class SVD_LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, ratio=1, compat_ranks: Optional[bool] = None, compat_attention: Optional[bool] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Compatibility flags (default from environment variables)
        if compat_ranks is None:
            compat_ranks = os.getenv('SVDLLM_COMPAT_RANKS', '0') != '0'
        if compat_attention is None:
            compat_attention = os.getenv('SVDLLM_COMPAT_ATTENTION', '0') != '0'
        self.compat_ranks = compat_ranks
        self.compat_attention = compat_attention
        # Support grouped-query attention (GQA): K/V heads can be fewer than Q heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = max(1, self.num_heads // self.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.ratio = ratio # 1 means no truncate, just keep normal attn
        # Default layer index (HF sets this on construction); keep a safe default
        self.layer_idx = 0

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        if self.compat_ranks:
            # Official attention rank: int(hidden_size * ratio / 2)
            low_rank = max(1, int(self.hidden_size * ratio / 2.0))
        else:
            low_rank = max(1, int(self.hidden_size * ratio))
        self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        # For GQA, K/V project to num_key_value_heads * head_dim
        self.k_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.v_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        # No Triton loader: we use fused SDPA path exclusively.

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # HF >=4.40 passes `past_key_values`; accept it for compatibility
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        # HF >=4.44 cache and pos embeddings
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Backward compatibility for older checkpoints without compat flags
        compat_attention = getattr(self, 'compat_attention', False)
        # Ensure projection weights follow hidden_states dtype to avoid matmul dtype mismatch
        dtype = hidden_states.dtype
        casted = getattr(self, "_weights_casted", None)
        if casted != dtype:
            self.q_u_proj.to(dtype=dtype)
            self.q_v_proj.to(dtype=dtype)
            self.k_u_proj.to(dtype=dtype)
            self.k_v_proj.to(dtype=dtype)
            self.v_u_proj.to(dtype=dtype)
            self.v_v_proj.to(dtype=dtype)
            self.o_u_proj.to(dtype=dtype)
            self.o_v_proj.to(dtype=dtype)
            self._weights_casted = dtype
        # Backward/forward compatibility between HF versions
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values
        bsz, q_len, _ = hidden_states.size()
        # Backward compat for older pickled checkpoints lacking GQA attributes
        if not hasattr(self, 'num_key_value_heads'):
            try:
                self.num_key_value_heads = int(getattr(self.config, 'num_key_value_heads', self.num_heads))
            except Exception:
                # Derive from weight shape if config is missing
                try:
                    self.num_key_value_heads = max(1, int(self.k_u_proj.out_features // self.head_dim))
                except Exception:
                    self.num_key_value_heads = self.num_heads
        if not hasattr(self, 'num_key_value_groups'):
            try:
                self.num_key_value_groups = max(1, int(self.num_heads // self.num_key_value_heads))
            except Exception:
                self.num_key_value_groups = 1
        # Detect whether caller uses new HF Cache API (provides cache object / cache_position / position_embeddings)
        new_cache_api = False
        try:
            from transformers.cache_utils import Cache
            if isinstance(past_key_value, Cache):
                new_cache_api = True
        except Exception:
            pass
        if (locals().get('cache_position', None) is not None) or (locals().get('position_embeddings', None) is not None):
            new_cache_api = True

        q = self.q_u_proj(self.q_v_proj(hidden_states))
        q_bias = getattr(self, "q_bias", None)
        if q_bias is not None:
            try:
                if int(q_bias.numel()) == int(q.shape[-1]):
                    q = q + q_bias.to(device=q.device, dtype=q.dtype)
            except Exception:
                pass
        query_states = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # For robustness with older checkpoints: infer kv head count from projection shapes if needed
        kv_heads_from_w = None
        try:
            kv_heads_from_w = int(self.k_u_proj.out_features // self.head_dim)
        except Exception:
            kv_heads_from_w = None
        kv_heads = kv_heads_from_w if kv_heads_from_w and kv_heads_from_w > 0 else self.num_key_value_heads
        k = self.k_u_proj(self.k_v_proj(hidden_states))
        k_bias = getattr(self, "k_bias", None)
        if k_bias is not None:
            try:
                if int(k_bias.numel()) == int(k.shape[-1]):
                    k = k + k_bias.to(device=k.device, dtype=k.dtype)
            except Exception:
                pass
        key_states = k.view(bsz, q_len, kv_heads, self.head_dim).transpose(1, 2)

        v = self.v_u_proj(self.v_v_proj(hidden_states))
        v_bias = getattr(self, "v_bias", None)
        if v_bias is not None:
            try:
                if int(v_bias.numel()) == int(v.shape[-1]):
                    v = v + v_bias.to(device=v.device, dtype=v.dtype)
            except Exception:
                pass
        value_states = v.view(bsz, q_len, kv_heads, self.head_dim).transpose(1, 2)

        # Optional QK normalization (Qwen-style) if present on the module.
        q_norm = getattr(self, "q_norm", None)
        if q_norm is not None:
            try:
                query_states = q_norm(query_states)
            except Exception:
                pass
        k_norm = getattr(self, "k_norm", None)
        if k_norm is not None:
            try:
                key_states = k_norm(key_states)
            except Exception:
                pass

        # Rotary embeddings: prefer shared `position_embeddings` from HF model if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)
        else:
            # Fallback to local rotary; ensure we cover max position idx
            if position_ids is not None:
                # Clamp to avoid negative indices from left-padding masks
                try:
                    position_ids = position_ids.clamp_min(0)
                except Exception:
                    pass
                max_pos = int(position_ids.max().item()) + 1
            else:
                max_pos = key_states.shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=max_pos)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        # KV cache support for both new `Cache` API and legacy tuples
        present = None
        if past_key_value is not None:
            try:
                from transformers.cache_utils import Cache
                if isinstance(past_key_value, Cache):
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, getattr(self, "layer_idx", 0), cache_kwargs
                    )
                    if use_cache:
                        present = past_key_value
                else:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)
                    if use_cache:
                        present = (key_states, value_states)
            except Exception:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                if use_cache:
                    present = (key_states, value_states)
        else:
            if use_cache:
                present = None

        # If using grouped key/value heads, expand them to match query heads for attention
        # Recompute groups if we derived kv_heads from weight (older checkpoints)
        if kv_heads_from_w and kv_heads_from_w > 0:
            try:
                self.num_key_value_groups = max(1, int(self.num_heads // kv_heads_from_w))
            except Exception:
                self.num_key_value_groups = 1
        if self.num_key_value_groups > 1:
            # [B, kv_heads, T, Hd] -> [B, heads, T, Hd]
            def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
                b, kv, t, hd = x.shape
                if n_rep == 1:
                    return x
                return (
                    x.unsqueeze(2)
                     .expand(b, kv, n_rep, t, hd)
                     .reshape(b, kv * n_rep, t, hd)
                )
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)

        # In compat mode, force explicit attention math and fixed 3-returns
        if compat_attention:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
        elif not output_attentions:
            # Use additive mask provided by HF model when it matches q/k sizes; otherwise rely on causal masking.
            q = query_states
            k = key_states
            v = value_states
            attn_mask_sdpa = None
            use_causal = True
            if attention_mask is not None:
                if torch.is_tensor(attention_mask):
                    if attention_mask.dim() == 4:
                        # Expected mask shape: [B, 1 or H, Q, K]
                        Bm, Hm, Qm, Km = attention_mask.shape
                        Bq, Hq, Qq, Kk = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
                        if Bm == Bq and Qm in (1, Qq) and Km == Kk and Hm in (1, Hq):
                            attn_mask_sdpa = attention_mask
                            use_causal = False
                    elif attention_mask.dim() == 2:
                        # Key padding mask from lm-eval-harness (bool, broadcast to [B,H,Q,K])
                        am = attention_mask.to(torch.bool)
                        attn_mask_sdpa = am[:, None, None, :]
                        use_causal = True
            # Call SDPA without forcing specific kernel flags to avoid version mismatches
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask_sdpa, dropout_p=0.0, is_causal=use_causal
            )
            if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
                attn_output = attn_output.masked_fill(~attention_mask[:, None, :, None].to(torch.bool), 0.0)
            # Numeric stability fallback: if SDPA returns non-finite, recompute in float32 math kernel
            if not torch.isfinite(attn_output).all():
                q32 = q.float()
                k32 = k.float()
                v32 = v.float()
                am32 = attn_mask_sdpa.float() if (attn_mask_sdpa is not None) else None
                try:
                    attn_output = F.scaled_dot_product_attention(
                        q32, k32, v32, attn_mask=am32, dropout_p=0.0, is_causal=use_causal
                    ).to(q.dtype)
                except Exception:
                    # Final safety: explicit softmax path
                    attn_weights = torch.matmul(q32, k32.transpose(2, 3)) / math.sqrt(self.head_dim)
                    if am32 is not None:
                        attn_weights = attn_weights + am32
                    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
                    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
                    attn_output = torch.matmul(attn_weights, v32).to(q.dtype)
                if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
                    attn_output = attn_output.masked_fill(~attention_mask[:, None, :, None].to(torch.bool), 0.0)
        else:
            # Fallback to explicit weights path only when attentions are requested
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        # up until here we should change to our code so it can works with our kernel easily
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_u_proj(self.o_v_proj(attn_output))
        o_bias = getattr(self, "o_bias", None)
        if o_bias is not None:
            try:
                if int(o_bias.numel()) == int(attn_output.shape[-1]):
                    attn_output = attn_output + o_bias.to(device=attn_output.device, dtype=attn_output.dtype)
            except Exception:
                pass

        if not output_attentions:
            attn_weights = None

        # Return policy
        if compat_attention:
            return attn_output, attn_weights, present
        if os.getenv('SVDLLM_ATTN_RETURN_THREE', '0') == '1' and use_cache:
            return attn_output, attn_weights, present
        return attn_output, attn_weights
    
