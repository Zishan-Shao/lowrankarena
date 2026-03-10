import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

try:
    from transformers.masking_utils import create_causal_mask
except Exception:
    create_causal_mask = None

try:
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention as LlamaBaseAttention
except ImportError:
    from transformers.models.llama.modeling_llama import LlamaAttention as LlamaBaseAttention

from models.model_utils import build_basis_collection, Coefficient

logger = logging.get_logger(__name__)


class ShareLlamaSdpaAttention(LlamaBaseAttention):
    def __init__(self, config, layer_idx, k_basis, q_basis, v_basis, o_basis):
        super().__init__(config, layer_idx)

        # transformers>=4.57 removed several attributes that Basis Sharing's
        # 4.45-era wrapper expects to exist.
        self.hidden_size = getattr(self, "hidden_size", config.hidden_size)
        self.num_heads = getattr(self, "num_heads", config.num_attention_heads)
        self.num_key_value_heads = getattr(self, "num_key_value_heads", config.num_key_value_heads)
        self.num_key_value_groups = getattr(
            self, "num_key_value_groups", self.num_heads // self.num_key_value_heads
        )
        self.max_position_embeddings = getattr(
            self, "max_position_embeddings", config.max_position_embeddings
        )
        self.rope_theta = getattr(self, "rope_theta", getattr(config, "rope_theta", 10000.0))
        if not hasattr(self, "rotary_emb"):
            self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.q_basis = q_basis
        self.q_proj = Coefficient(self.num_heads * self.head_dim, config.num_basis_q)
        self.k_basis = k_basis
        self.k_proj = Coefficient(self.num_key_value_heads * self.head_dim, config.num_basis_k)
        self.v_basis = v_basis
        self.v_proj = Coefficient(self.num_key_value_heads * self.head_dim, config.num_basis_v)
        self.o_basis = o_basis
        self.o_proj = Coefficient(self.hidden_size, config.num_basis_o)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if output_attentions:
            raise NotImplementedError

        bsz, q_len, _ = hidden_states.size()
        key_states = self.k_proj(self.k_basis(hidden_states))
        query_states = self.q_proj(self.q_basis(hidden_states))
        value_states = self.v_proj(self.v_basis(hidden_states))

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "ShareLlamaSdpaAttention received position_embeddings=None; falling back to internal RoPE."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(self.o_basis(attn_output))
        return attn_output, None, past_key_value


class ShareLlamaMLP(LlamaMLP):
    def __init__(self, config, layer_idx, up_basis, gate_basis, down_basis):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.gate_basis = gate_basis
        self.gate_proj = Coefficient(self.intermediate_size, config.num_basis_gate)
        self.up_basis = up_basis
        self.up_proj = Coefficient(self.intermediate_size, config.num_basis_up)
        self.down_basis = down_basis
        self.down_proj = Coefficient(self.hidden_size, config.num_basis_down)

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            raise NotImplementedError
        down = self.down_proj(
            self.down_basis(self.act_fn(self.gate_proj(self.gate_basis(x))) * self.up_proj(self.up_basis(x)))
        )
        return down


class ShareLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, k_basis, q_basis, v_basis, o_basis, up_basis, gate_basis, down_basis):
        super().__init__(config, layer_idx)
        self.self_attn = ShareLlamaSdpaAttention(
            config,
            layer_idx,
            k_basis[str(layer_idx)],
            q_basis[str(layer_idx)],
            v_basis[str(layer_idx)],
            o_basis[str(layer_idx)],
        )
        self.mlp = ShareLlamaMLP(
            config,
            layer_idx,
            up_basis[str(layer_idx)],
            gate_basis[str(layer_idx)],
            down_basis[str(layer_idx)],
        )

    @staticmethod
    def _in_group(groups, layer_idx):
        return any(layer_idx in group for group in groups)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class ShareLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "num_basis_k"):
            self.k_basis = build_basis_collection(config.k_groups, config.num_basis_k, config.hidden_size)
        else:
            self.k_basis = None
        if hasattr(config, "num_basis_q"):
            self.q_basis = build_basis_collection(config.q_groups, config.num_basis_q, config.hidden_size)
        else:
            self.q_basis = None
        if hasattr(config, "num_basis_v"):
            self.v_basis = build_basis_collection(config.v_groups, config.num_basis_v, config.hidden_size)
        else:
            self.v_basis = None
        if hasattr(config, "num_basis_o"):
            self.o_basis = build_basis_collection(config.o_groups, config.num_basis_o, config.hidden_size)
        else:
            self.o_basis = None
        if hasattr(config, "num_basis_gate"):
            self.gate_basis = build_basis_collection(config.gate_groups, config.num_basis_gate, config.hidden_size)
        else:
            self.gate_basis = None
        if hasattr(config, "num_basis_up"):
            self.up_basis = build_basis_collection(config.up_groups, config.num_basis_up, config.hidden_size)
        else:
            self.up_basis = None
        if hasattr(config, "num_basis_down"):
            self.down_basis = build_basis_collection(
                config.down_groups, config.num_basis_down, config.intermediate_size
            )
        else:
            self.down_basis = None

        self.layers = nn.ModuleList(
            [
                ShareLlamaDecoderLayer(
                    config,
                    layer_idx,
                    self.k_basis,
                    self.q_basis,
                    self.v_basis,
                    self.o_basis,
                    self.up_basis,
                    self.gate_basis,
                    self.down_basis,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def freeze_basis(self):
        for maybe_basis in [
            self.k_basis,
            self.q_basis,
            self.v_basis,
            self.o_basis,
            self.gate_basis,
            self.up_basis,
            self.down_basis,
        ]:
            if maybe_basis is not None:
                for param in maybe_basis.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and past_key_values is None:
            try:
                past_key_values = DynamicCache(config=self.config)
            except TypeError:
                past_key_values = DynamicCache()
        elif use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "Passing past_key_values as a tuple is deprecated; converting to DynamicCache."
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if create_causal_mask is not None:
            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        else:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ShareLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = ShareLlamaModel(config)
        self.config = config
