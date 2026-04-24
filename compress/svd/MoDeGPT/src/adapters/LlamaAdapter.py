from .model_adapter import (
    ModelAdapter,
    MLPComponents,
    MLPTensors,
    AttentionComponents,
    VOComponents,
    VOTensors,
    QKComponents,
    QKTensors,
)

import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass
import logging

from transformers.configuration_utils import PretrainedConfig

from src.model_utils import dtype_p, calib_device

import copy

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from collections.abc import Callable


import types


class LlamaAdapter(ModelAdapter):
    @property
    def arch(self) -> str:
        return "llama"

    def get_transformer_blocks(self) -> nn.ModuleList:
        return self.model.model.layers

    # def get_n_inner(self, block) -> int:
    #     return block.mlp.gate_proj.out_features

    @staticmethod
    def _attn_hook(layer_idx, cov_q_list, cov_k_list):
        @torch.no_grad()
        def hook(module, inp, out):
            n_q_heads = module.config.num_attention_heads
            n_kv_heads = module.config.num_key_value_heads
            head_dims = module.head_dim

            query_states = module.patched_q_states
            key_states = module.patched_k_states

            q = query_states.transpose(1, 2).reshape(-1, n_q_heads, head_dims)
            q = q.permute(1, 0, 2).to(dtype=dtype_p, device="cuda")  # [n_q_heads, B*T, head_dims]
            C_q = torch.bmm(q.transpose(1, 2), q).to(calib_device)
            cov_q_list[layer_idx] += C_q

            k = key_states.transpose(1, 2).reshape(-1, n_kv_heads, head_dims)
            k = k.permute(1, 0, 2).to(dtype=dtype_p, device="cuda")  # [n_kv_heads, B*T, head_dims]
            C_k = torch.bmm(k.transpose(1, 2), k).to(calib_device)
            cov_k_list[layer_idx] += C_k

            return None

        return hook

    def register_hooks(
        self, layer_idx, block, cov_mlp_list, cov_q_list, cov_k_list, cov_x_list, handles, logger
    ):
        handles.append(
            block.mlp.down_proj.register_forward_pre_hook(
                self._llama_pre_gate_hook(layer_idx, cov_mlp_list)
            )
        )
        handles.append(
            block.input_layernorm.register_forward_hook(self._input_hook(layer_idx, cov_x_list))
        )

        # using the states with pos emb applied actually performed worse overall
        # block.self_attn.forward = types.MethodType(patched_attn_forward, block.self_attn)

        # handles.append(
        #     block.self_attn.register_forward_hook(
        #         self._attn_hook(layer_idx, cov_q_list, cov_k_list),
        #     )
        # )
        handles.append(
            block.self_attn.k_proj.register_forward_hook(
                self._make_proj_hook(layer_idx, cov_k_list, self.n_kv_heads, self.head_dim, block)
            )
        )
        handles.append(
            block.self_attn.q_proj.register_forward_hook(
                self._make_proj_hook(layer_idx, cov_q_list, self.n_heads, self.head_dim, block)
            )
        )

    def compute_layer_energy(self, layer_idx: int, Ca: Tensor | None = None) -> MLPTensors:
        raise NotImplementedError("compute_layer_energy not imp for llama")

    def calibrate_model(
        self, n_samples: int, batch_size: int, target_layers: list[int], dataset="wikitext"
    ):
        raise NotImplementedError("custom calibrate model not impl for llama")

    """
    For all the hooks we compute the matmul in f32 then cast up when acc to f64
    Hopefully not missing any detail because it makes it faster for large models.
    """

    @staticmethod
    def _make_proj_hook(layer_idx, cov_list, n_heads, head_dim, d_model):
        @torch.no_grad()
        def hook(module, inp, out):
            proj_out = out.detach().to(dtype=dtype_p, device="cuda")
            proj = proj_out.view(-1, n_heads, head_dim)
            proj = proj.permute(1, 0, 2)  # [n_heads, B*T, head_dim]
            C_proj = torch.bmm(proj.transpose(1, 2), proj).to(dtype=dtype_p, device=calib_device)
            cov_list[layer_idx] += C_proj

        return hook

    @staticmethod
    def _llama_pre_gate_hook(layer_idx, cov_mlp_list):
        @torch.no_grad()
        def hook(module, input: Tuple[torch.Tensor]):
            x_input = input[0]
            H = x_input.detach().to(dtype=dtype_p).view(-1, x_input.size(-1))
            cov_mlp_list[layer_idx] += (H.T @ H).to(dtype=dtype_p, device=calib_device)
            return None

        return hook

    @staticmethod
    def _input_hook(layer_idx, cov_list):
        @torch.no_grad()
        def hook(module, inp, out):
            proj_out = out.detach().to(dtype=dtype_p, device="cuda")
            cov_list[layer_idx] += torch.sum(proj_out.mT @ proj_out, dim=0).to(
                dtype=dtype_p, device=calib_device
            )

        return hook

    def get_mlp_components(self, layer_idx: int, expert_idx: int | None = None) -> MLPComponents:
        block = self.get_transformer_blocks()[layer_idx]
        return MLPComponents(
            block=block,
            up_proj=block.mlp.up_proj,
            down_proj=block.mlp.down_proj,
            gate_proj=block.mlp.gate_proj,
        )

    def get_mlp_tensors(self, layer_idx: int, expert_idx: int | None = None) -> MLPTensors:
        block = self.get_transformer_blocks()[layer_idx]
        return MLPTensors(
            up_proj=block.mlp.up_proj.weight,
            down_proj=block.mlp.down_proj.weight,
            gate_proj=block.mlp.gate_proj.weight,
        )

    def get_vo_components(self, layer_idx: int, expert_idx: int | None = None) -> VOComponents:
        block = self.get_transformer_blocks()[layer_idx]
        return VOComponents(
            block=block,
            v_proj=block.self_attn.v_proj,
            o_proj=block.self_attn.o_proj,
        )

    def get_vo_tensors(self, layer_idx: int, expert_idx: int | None = None) -> VOTensors:
        block = self.get_transformer_blocks()[layer_idx]
        return VOTensors(
            v_proj=block.self_attn.v_proj.weight,
            o_proj=block.self_attn.o_proj.weight,
        )

    def get_qk_components(self, layer_idx: int, expert_idx: int | None = None) -> QKComponents:
        block = self.get_transformer_blocks()[layer_idx]
        return QKComponents(
            block=block,
            query_proj=block.self_attn.q_proj,
            key_proj=block.self_attn.k_proj,
        )

    def get_qk_tensors(self, layer_idx: int, expert_idx: int | None = None) -> QKTensors:
        block = self.get_transformer_blocks()[layer_idx]
        return QKTensors(
            query_proj=block.self_attn.q_proj.weight,
            key_proj=block.self_attn.k_proj.weight,
        )

    def replace_mlp_layers(
        self,
        layer_idx: int,
        new_up: nn.Module,
        new_down: nn.Module,
        new_gate: Optional[nn.Module] = None,
        expert_idx: Optional[int] = None,
    ) -> None:
        block = self.get_transformer_blocks()[layer_idx]
        block.mlp.up_proj = new_up
        block.mlp.down_proj = new_down
        if new_gate is not None:
            block.mlp.gate_proj = new_gate

    def get_attn_components(self, layer_idx: int) -> AttentionComponents:
        block = self.get_transformer_blocks()[layer_idx]
        return AttentionComponents(
            block=block,
            q_proj=block.self_attn.q_proj,
            k_proj=block.self_attn.k_proj,
            v_proj=block.self_attn.v_proj,
            o_proj=block.self_attn.o_proj,
        )

    def replace_attn_layers(
        self,
        layer_idx: int,
        new_q: Optional[nn.Module],
        new_k: Optional[nn.Module],
        new_v: Optional[nn.Module],
        new_o: Optional[nn.Module],
    ) -> None:
        block = self.get_transformer_blocks()[layer_idx]
        if new_q is not None:
            block.self_attn.q_proj = new_q
        if new_k is not None:
            block.self_attn.k_proj = new_k
        if new_v is not None:
            block.self_attn.v_proj = new_v
        if new_o is not None:
            block.self_attn.o_proj = new_o

    def get_qk_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.get_transformer_blocks()[layer_idx]
        W_q = block.self_attn.q_proj.weight
        W_k = block.self_attn.k_proj.weight
        return W_q, W_k

    def get_vo_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.get_transformer_blocks()[layer_idx]
        W_v = block.self_attn.v_proj.weight
        W_o = block.self_attn.o_proj.weight
        return W_v, W_o

    def patch_config(self):
        model = self.model

        og_config = copy.deepcopy(model.config)
        config = model.config

        n_layers, arch = config.num_hidden_layers, config.model_type

        # handle qk/vo separately in case only one stage is used
        q_ranks = []
        k_ranks = []
        v_ranks = []
        o_ranks = []
        gate_ranks = []
        for layer in range(n_layers):
            qk = self.get_qk_tensors(layer)
            vo = self.get_vo_tensors(layer)
            mlp = self.get_mlp_tensors(layer)

            query_weight, key_weight = qk.query_proj, qk.key_proj
            value_weight, output_weight = vo.v_proj, vo.o_proj
            gate_proj = mlp.gate_proj

            q_rank = query_weight.shape[0]  # q/k
            k_rank = key_weight.shape[0]  # q/k

            v_rank = value_weight.shape[0]  # v/o
            o_rank = output_weight.shape[1]
            gate_rank = gate_proj.shape[0]  # u/g

            q_ranks.append(q_rank)
            k_ranks.append(k_rank)
            v_ranks.append(v_rank)
            o_ranks.append(o_rank)
            gate_ranks.append(gate_rank)

        # just break these values so we know where's something's going wrong later
        config.ffn_dim = -1
        config.q_ranks = q_ranks
        config.k_ranks = k_ranks
        config.v_ranks = v_ranks
        config.o_ranks = o_ranks
        config.gate_ranks = gate_ranks
        if arch == "opt":
            config.auto_map = {"AutoModelForCausalLM": "OPTRebuild.OPTForCausalLM"}
        if arch == "llama":
            config.auto_map = {"AutoModelForCausalLM": "LlamaRebuild.LlamaForCausalLM"}
        if "qwen" in arch:
            config.auto_map = {
                "AutoModelForCausalLM": "DenseQwenRebuild.Qwen3ForCausalLM",
            }

        return og_config


def patched_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    self.patched_q_states = query_states
    self.patched_k_states = key_states
    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
