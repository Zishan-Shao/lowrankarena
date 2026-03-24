"""Qwen SVD modules for baselines/SVD-LLM.

Qwen2/Qwen2.5/Qwen3 are LLaMA-like but add per-head Q/K RMSNorm (qk_norm)
and optionally sliding-window attention. This module subclasses the FlashSVD
LLaMA attention so isinstance checks and no_split_module_classes continue to work.
"""

from __future__ import annotations

from typing import Optional, Any

import torch

from component.svd_llama import (
    SVD_LlamaAttention as _SVD_LlamaAttention,
    SVD_LlamaMLP as _SVD_LlamaMLP,
)
from flashsvd_component.svd_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


def _first_attr(obj: Any, names: tuple) -> Optional[Any]:
    for n in names:
        val = getattr(obj, n, None)
        if val is not None:
            return val
    return None


def _maybe_register_linear_bias(module, buf_name: str, src_linear: Any, expected_size: int) -> None:
    b = getattr(src_linear, "bias", None)
    if b is None:
        return
    try:
        b = b.detach()
    except Exception:
        pass
    try:
        exp = int(expected_size)
        if int(b.numel()) != exp:
            if int(b.numel()) > exp:
                b = b[:exp]
            else:
                nb = b.new_zeros(exp)
                nb[: int(b.numel())] = b
                b = nb
    except Exception:
        return
    try:
        if hasattr(module, buf_name):
            setattr(module, buf_name, b.clone())
        else:
            module.register_buffer(buf_name, b.clone())
    except Exception:
        pass


class SVD_LlamaAttention(_SVD_LlamaAttention):
    """Qwen variant: copies q_norm/k_norm and attention biases from original layer."""

    def __init__(
        self,
        config,
        ratio=1,
        compat_ranks: Optional[bool] = None,
        compat_attention: Optional[bool] = None,
        base_attn: Optional[Any] = None,
    ):
        kwargs = {}
        if compat_ranks is not None:
            kwargs["compat_ranks"] = compat_ranks
        if compat_attention is not None:
            kwargs["compat_attention"] = compat_attention
        super().__init__(config=config, ratio=ratio, **kwargs)

        if base_attn is not None:
            # Copy attention projection biases (Qwen2/Qwen2.5; Qwen3 has none).
            q_proj = _first_attr(base_attn, ("q_proj", "query_proj", "wq"))
            k_proj = _first_attr(base_attn, ("k_proj", "key_proj", "wk"))
            v_proj = _first_attr(base_attn, ("v_proj", "value_proj", "wv"))
            o_proj = _first_attr(base_attn, ("o_proj", "out_proj", "wo"))
            if q_proj is not None:
                _maybe_register_linear_bias(self, "q_bias", q_proj, self.num_heads * self.head_dim)
            if k_proj is not None:
                _maybe_register_linear_bias(self, "k_bias", k_proj, self.num_key_value_heads * self.head_dim)
            if v_proj is not None:
                _maybe_register_linear_bias(self, "v_bias", v_proj, self.num_key_value_heads * self.head_dim)
            if o_proj is not None:
                _maybe_register_linear_bias(self, "o_bias", o_proj, self.hidden_size)

            # Preserve sliding-window metadata.
            try:
                self.sliding_window = getattr(base_attn, "sliding_window", None)
            except Exception:
                pass

            # Copy per-head Q/K RMSNorm (Qwen3).
            q_src, k_src = None, None
            for attr in ("q_norm", "q_layernorm", "q_ln"):
                q_src = getattr(base_attn, attr, None)
                if q_src is not None:
                    break
            for attr in ("k_norm", "k_layernorm", "k_ln"):
                k_src = getattr(base_attn, attr, None)
                if k_src is not None:
                    break

            if q_src is not None and hasattr(q_src, "weight"):
                eps = getattr(q_src, "variance_epsilon", getattr(q_src, "eps", getattr(config, "rms_norm_eps", 1e-6)))
                self.q_norm = LlamaRMSNorm(self.head_dim, eps=float(eps))
                try:
                    self.q_norm.weight.data.copy_(q_src.weight.data)
                except Exception:
                    pass
            if k_src is not None and hasattr(k_src, "weight"):
                eps = getattr(k_src, "variance_epsilon", getattr(k_src, "eps", getattr(config, "rms_norm_eps", 1e-6)))
                self.k_norm = LlamaRMSNorm(self.head_dim, eps=float(eps))
                try:
                    self.k_norm.weight.data.copy_(k_src.weight.data)
                except Exception:
                    pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_values=None,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        # For Qwen3 sliding-window layers: rebuild an SDPA-compatible 4D mask.
        attn_mask_2d = attention_mask if (
            attention_mask is not None
            and torch.is_tensor(attention_mask)
            and attention_mask.dim() == 2
        ) else None
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None and not output_attentions:
            try:
                from transformers import masking_utils as _mu
                try:
                    from transformers.cache_utils import Cache as _Cache
                    pkv = past_key_value if isinstance(past_key_value, _Cache) else None
                except Exception:
                    pkv = None
                cp = cache_position
                if cp is None:
                    cp = position_ids[0] if position_ids is not None else torch.arange(
                        hidden_states.shape[1], device=hidden_states.device
                    )
                try:
                    _out = _mu._preprocess_mask_arguments(
                        config=self.config, input_embeds=hidden_states,
                        attention_mask=attn_mask_2d, cache_position=cp,
                        past_key_values=pkv, position_ids=position_ids,
                        layer_idx=getattr(self, "layer_idx", 0),
                    )
                except TypeError:
                    _out = _mu._preprocess_mask_arguments(
                        config=self.config, input_embeds=hidden_states,
                        attention_mask=attn_mask_2d, cache_position=cp,
                        past_key_values=pkv, layer_idx=getattr(self, "layer_idx", 0),
                    )
                if isinstance(_out, tuple):
                    if len(_out) == 5:
                        early_exit, am_prep, packed_seq_mask, kv_len, kv_off = _out
                    elif len(_out) == 4:
                        early_exit, am_prep, kv_len, kv_off = _out
                        packed_seq_mask = None
                    else:
                        raise ValueError
                    kv_len = int(hidden_states.shape[1]) if kv_len is None else int(kv_len)
                    kv_off = 0 if kv_off is None else int(kv_off)
                    if early_exit and am_prep is not None and torch.is_tensor(am_prep):
                        attention_mask = am_prep
                    else:
                        mask_fn = _mu.sliding_window_causal_mask_function(int(sliding_window))
                        if packed_seq_mask is not None:
                            try:
                                mask_fn = _mu.and_masks(mask_fn, _mu.packed_sequence_mask_function(packed_seq_mask))
                            except Exception:
                                pass
                        mask_4d = _mu.sdpa_mask(
                            batch_size=int(hidden_states.shape[0]),
                            cache_position=cp, kv_length=kv_len, kv_offset=kv_off,
                            mask_function=mask_fn, attention_mask=am_prep,
                            local_size=int(sliding_window), allow_is_causal_skip=True,
                        )
                        attention_mask = mask_4d if mask_4d is not None else attn_mask_2d
            except Exception:
                pass

        outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if not isinstance(outputs, tuple):
            return outputs
        if len(outputs) == 3:
            attn_output, attn_weights, present = outputs
        elif len(outputs) == 2:
            attn_output, attn_weights = outputs
            present = None
        else:
            return outputs

        if attn_mask_2d is not None:
            try:
                am = attn_mask_2d.to(torch.bool)
                if am.dim() == 2 and am.shape[1] == attn_output.shape[1]:
                    attn_output = attn_output.masked_fill(~am[:, :, None], 0.0)
            except Exception:
                pass

        if present is not None:
            return attn_output, attn_weights, present
        return attn_output, attn_weights


class SVD_LlamaMLP(_SVD_LlamaMLP):
    """Qwen MLP is compatible with the LLaMA SVD MLP."""
    pass


# Explicit aliases.
SVD_QwenAttention = SVD_LlamaAttention
SVD_QwenMLP = SVD_LlamaMLP
