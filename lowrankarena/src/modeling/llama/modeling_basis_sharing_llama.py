from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaForCausalLM,
    LlamaAttention,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.utils import logging

try:
    from src.modeling.common import BasisCoefficient, build_basis_collection
except ImportError:  # pragma: no cover - used when copied into a standalone HF artifact.
    from .common import BasisCoefficient, build_basis_collection

from .configuration_basis_sharing_llama import BasisSharingLlamaConfig


logger = logging.get_logger(__name__)


class BasisSharingLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx: int, *, q_basis, k_basis, v_basis, o_basis):
        super().__init__(config, layer_idx)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.q_basis = q_basis
        self.k_basis = k_basis
        self.v_basis = v_basis
        self.o_basis = o_basis
        self.q_proj = BasisCoefficient(self.num_attention_heads * self.head_dim, config.num_basis_q)
        self.k_proj = BasisCoefficient(self.num_key_value_heads * self.head_dim, config.num_basis_k)
        self.v_proj = BasisCoefficient(self.num_key_value_heads * self.head_dim, config.num_basis_v)
        self.o_proj = BasisCoefficient(config.hidden_size, config.num_basis_o)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(self.q_basis(hidden_states)).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(self.k_basis(hidden_states)).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(self.v_basis(hidden_states)).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "Basis sharing attention is computing RoPE internally from position_ids. "
                "Transformers will eventually require external position_embeddings."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if output_attentions and self.config._attn_implementation != "flex_attention":
                logger.warning_once(
                    "Basis sharing attention is falling back to eager attention because "
                    "output_attentions=True is not supported by the current backend."
                )
                attention_interface = eager_attention_forward
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(self.o_basis(attn_output))
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class BasisSharingLlamaMLP(LlamaMLP):
    def __init__(self, config, *, gate_basis, up_basis, down_basis):
        super().__init__(config)
        self.gate_basis = gate_basis
        self.up_basis = up_basis
        self.down_basis = down_basis
        self.gate_proj = BasisCoefficient(self.intermediate_size, config.num_basis_gate)
        self.up_proj = BasisCoefficient(self.intermediate_size, config.num_basis_up)
        self.down_proj = BasisCoefficient(self.hidden_size, config.num_basis_down)

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Basis sharing MLP does not support pretraining_tp > 1.")
        gated = self.act_fn(self.gate_proj(self.gate_basis(x))) * self.up_proj(self.up_basis(x))
        return self.down_proj(self.down_basis(gated))


class BasisSharingLlamaModel(LlamaModel):
    config_class = BasisSharingLlamaConfig
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        self.k_basis = build_basis_collection(getattr(config, "k_groups", []), config.num_basis_k, config.hidden_size)
        self.q_basis = build_basis_collection(getattr(config, "q_groups", []), config.num_basis_q, config.hidden_size)
        self.v_basis = build_basis_collection(getattr(config, "v_groups", []), config.num_basis_v, config.hidden_size)
        self.o_basis = build_basis_collection(getattr(config, "o_groups", []), config.num_basis_o, config.hidden_size)
        self.gate_basis = build_basis_collection(
            getattr(config, "gate_groups", []),
            config.num_basis_gate,
            config.hidden_size,
        )
        self.up_basis = build_basis_collection(getattr(config, "up_groups", []), config.num_basis_up, config.hidden_size)
        self.down_basis = build_basis_collection(
            getattr(config, "down_groups", []),
            config.num_basis_down,
            config.intermediate_size,
        )

        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn = BasisSharingLlamaAttention(
                config,
                layer_idx,
                q_basis=self.q_basis[str(layer_idx)],
                k_basis=self.k_basis[str(layer_idx)],
                v_basis=self.v_basis[str(layer_idx)],
                o_basis=self.o_basis[str(layer_idx)],
            )
            layer.mlp = BasisSharingLlamaMLP(
                config,
                gate_basis=self.gate_basis[str(layer_idx)],
                up_basis=self.up_basis[str(layer_idx)],
                down_basis=self.down_basis[str(layer_idx)],
            )


class BasisSharingLlamaForCausalLM(LlamaForCausalLM):
    config_class = BasisSharingLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = BasisSharingLlamaModel(config)
