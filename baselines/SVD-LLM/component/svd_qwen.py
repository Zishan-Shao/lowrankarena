"""Qwen SVD modules.

Qwen2/Qwen2.5/Qwen3 are largely LLaMA-like, but RoPE base (theta) can differ
from LLaMA's default 10000 (e.g. 1e6). We keep `component/svd_llama.py` as the
LLaMA baseline and provide a Qwen-specific variant that reuses the same SVD
implementation while fixing the rotary embedding base.

Design notes:
  - We subclass the LLaMA SVD modules so `isinstance(x, SVD_LlamaAttention)`
    checks continue to work (e.g. activation-space LoRA wrappers).
  - The subclasses are intentionally named `SVD_LlamaAttention/MLP` to keep
    `no_split_module_classes=["SVD_LlamaAttention", ...]` working without
    touching every script.
"""

from __future__ import annotations

from typing import Optional, Any

import torch

from component.svd_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    SVD_LlamaAttention as _SVD_LlamaAttention,
    SVD_LlamaMLP as _SVD_LlamaMLP,
)

def _first_attr(obj: Any, names: tuple[str, ...]) -> Optional[Any]:
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
        # If shape logic fails, skip bias copy (better than crashing).
        return
    try:
        if hasattr(module, buf_name):
            setattr(module, buf_name, b.clone())
        else:
            module.register_buffer(buf_name, b.clone())
    except Exception:
        pass


def _get_rope_theta(config: Any, default: float = 10000.0) -> float:
    """Best-effort extract of RoPE theta from Qwen configs across versions."""
    # Common in Qwen2/Qwen2.5 style configs.
    for key in ("rope_theta", "rotary_emb_base", "rotary_base", "rope_base"):
        val = getattr(config, key, None)
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue

    # Qwen3 (newer Transformers) may bundle this under `rope_parameters`.
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params is not None:
        try:
            if isinstance(rope_params, dict):
                val = rope_params.get("rope_theta", None)
                if val is None:
                    val = rope_params.get("theta", None)
                if val is not None:
                    return float(val)
        except Exception:
            pass
        try:
            val = getattr(rope_params, "rope_theta", None)
            if val is not None:
                return float(val)
        except Exception:
            pass

    return float(default)


class SVD_LlamaAttention(_SVD_LlamaAttention):
    """Qwen variant of the LLaMA SVD attention (RoPE base from config)."""

    def __init__(
        self,
        config,
        ratio=1,
        compat_ranks: Optional[bool] = None,
        compat_attention: Optional[bool] = None,
        base_attn: Optional[Any] = None,
    ):
        super().__init__(config=config, ratio=ratio, compat_ranks=compat_ranks, compat_attention=compat_attention)
        rope_theta = _get_rope_theta(config, default=10000.0)
        # Rebuild rotary embedding with the correct base.
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=rope_theta,
        )

        # Qwen2/Qwen3 commonly apply per-head Q/K RMSNorm ("QK norm") inside attention.
        # If present on the original attention module, copy weights into local norms so
        # pickled checkpoints don't depend on remote/transformers-specific classes.
        if base_attn is not None:
            # Qwen2/Qwen2.5 can include biases in q/k/v (and sometimes o) projections.
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

            # Preserve layer-wise sliding window metadata when available (Qwen2/Qwen3).
            try:
                self.sliding_window = getattr(base_attn, "sliding_window", None)
            except Exception:
                pass

            q_src = None
            k_src = None
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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # Qwen3 can use sliding-window attention at long context. HF may provide a dense 4D mask which can
        # force SDPA into a slow/memory-heavy path. When possible, rebuild an efficient BlockMask and let the
        # base attention run via flex attention.
        attn_mask_2d = attention_mask if (attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.dim() == 2) else None
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None and (not output_attentions):
            try:
                from torch.nn.attention.flex_attention import BlockMask as _BlockMask
                if attention_mask is None or not isinstance(attention_mask, _BlockMask):
                    from transformers import masking_utils as _mu
                    try:
                        from transformers.cache_utils import Cache as _Cache
                        pkv = past_key_value if isinstance(past_key_value, _Cache) else None
                    except Exception:
                        pkv = None
                    cp = cache_position
                    if cp is None:
                        if position_ids is not None:
                            cp = position_ids[0]
                        else:
                            cp = torch.arange(hidden_states.shape[1], device=hidden_states.device)

                    early_exit, am_prep, packed_seq_mask, kv_len, kv_off = _mu._preprocess_mask_arguments(
                        config=self.config,
                        input_embeds=hidden_states,
                        attention_mask=attn_mask_2d,
                        cache_position=cp,
                        past_key_values=pkv,
                        position_ids=position_ids,
                        layer_idx=getattr(self, "layer_idx", 0),
                    )
                    if early_exit and am_prep is not None:
                        attention_mask = am_prep
                    else:
                        mask_fn = _mu.sliding_window_causal_mask_function(int(sliding_window))
                        if packed_seq_mask is not None and getattr(_mu, "_is_torch_greater_or_equal_than_2_6", True):
                            try:
                                mask_fn = _mu.and_masks(mask_fn, _mu.packed_sequence_mask_function(packed_seq_mask))
                            except Exception:
                                pass
                        attention_mask = _mu.flex_attention_mask(
                            batch_size=int(hidden_states.shape[0]),
                            cache_position=cp,
                            kv_length=int(kv_len),
                            kv_offset=int(kv_off),
                            mask_function=mask_fn,
                            attention_mask=am_prep,
                        )
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
            raise ValueError(f"Unexpected attention outputs length: {len(outputs)}")

        # Preserve HF behavior for padding queries when a 2D attention mask is available.
        if attn_mask_2d is not None:
            try:
                am = attn_mask_2d.to(torch.bool)
                if am.dim() == 2 and am.shape[1] == attn_output.shape[1]:
                    attn_output = attn_output.masked_fill(~am[:, :, None], 0.0)
            except Exception:
                pass
        if len(outputs) == 3:
            return attn_output, attn_weights, present
        return attn_output, attn_weights


class SVD_LlamaMLP(_SVD_LlamaMLP):
    """Qwen MLP is compatible with the LLaMA SVD MLP."""

    pass


# Friendly aliases for callers that prefer explicit Qwen naming.
SVD_QwenAttention = SVD_LlamaAttention
SVD_QwenMLP = SVD_LlamaMLP
