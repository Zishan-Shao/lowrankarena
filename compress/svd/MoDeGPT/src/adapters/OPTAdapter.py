from .model_adapter import ModelAdapter, MLPComponents, MLPTensors, AttentionComponents, VOComponents, VOTensors, QKComponents, QKTensors

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass
import logging

from transformers.configuration_utils import PretrainedConfig

from src.model_utils import dtype_p, calib_device


class OPTAdapter(ModelAdapter):
    @property
    def arch(self) -> str:
        return "opt"

    def get_transformer_blocks(self) -> nn.ModuleList:
        return self.model.model.decoder.layers

    def get_n_inner(self, block) -> int:
        return block.fc1.out_features

    def register_hooks(
        self, layer_idx, block, cov_mlp_list, cov_q_list, cov_k_list, cov_x_list, handles, logger
    ):
        handles.append(block.fc1.register_forward_hook(self._make_fc_hook(layer_idx, cov_mlp_list)))
        handles.append(
            block.self_attn.q_proj.register_forward_hook(
                self._make_proj_hook(
                    layer_idx, cov_q_list, self.n_heads, self.head_dim, self.d_model
                )
            )
        )
        handles.append(
            block.self_attn.k_proj.register_forward_hook(
                self._make_proj_hook(
                    layer_idx, cov_k_list, self.n_heads, self.head_dim, self.d_model
                )
            )
        )

    def on_batch_end_step(self, layer_idx, x_in, cov_x_list):
        cov_x_list[layer_idx] += torch.sum(x_in.mT @ x_in, dim=0).to(device=calib_device)

    def get_mlp_components(self, layer_idx: int) -> MLPComponents:
        block = self.get_transformer_blocks()[layer_idx]
        return MLPComponents(block=block, up_proj=block.fc1, down_proj=block.fc2)

    def get_mlp_tensors(self, layer_idx: int, expert_idx: int | None = None) -> MLPTensors:
        block = self.get_transformer_blocks()[layer_idx]
        return MLPTensors(
            up_proj=block.fc1.weight,
            down_proj=block.fc2.weight,
            gate_proj=None,
        )

    def get_vo_components(self, layer_idx: int, expert_idx: int | None = None) -> VOComponents:
        block = self.get_transformer_blocks()[layer_idx]
        return VOComponents(
            block=block,
            v_proj=block.self_attn.v_proj,
            o_proj=block.self_attn.out_proj,
        )

    def get_vo_tensors(self, layer_idx: int, expert_idx: int | None = None) -> VOTensors:
        block = self.get_transformer_blocks()[layer_idx]
        return VOTensors(
            v_proj=block.self_attn.v_proj.weight,
            o_proj=block.self_attn.out_proj.weight,
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
        block.fc1 = new_up
        block.fc2 = new_down

    def get_attn_components(self, layer_idx: int) -> AttentionComponents:
        block = self.get_transformer_blocks()[layer_idx]
        return AttentionComponents(
            block=block,
            q_proj=block.self_attn.q_proj,
            k_proj=block.self_attn.k_proj,
            v_proj=block.self_attn.v_proj,
            o_proj=block.self_attn.out_proj,
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
            block.self_attn.out_proj = new_o

    def get_qk_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.get_transformer_blocks()[layer_idx]
        W_q = block.self_attn.q_proj.weight
        W_k = block.self_attn.k_proj.weight
        return W_q, W_k

    def get_vo_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.get_transformer_blocks()[layer_idx]
        W_v = block.self_attn.v_proj.weight
        W_o = block.self_attn.out_proj.weight
        return W_v, W_o
