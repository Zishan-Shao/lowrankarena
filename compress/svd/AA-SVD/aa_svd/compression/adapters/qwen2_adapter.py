# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 Alibaba DAMO Academy and the HuggingFace Inc. team.
# https://www.apache.org/licenses/LICENSE-2.0

import torch
from torch import FloatTensor
from torch.nn import Linear, Module
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
)
from typing import List
import tqdm
import wandb
import gc
import copy
from math import cos, pi
from torch.optim.lr_scheduler import LambdaLR

from aa_svd.compression.model_adapter import LayerAdapter, ModelAdapter, HeadLayerAdapterMixin
from aa_svd.compression.utils import map_tensors


class Qwen2LayerAdapter(LayerAdapter):
    def __init__(self, layer: Qwen2DecoderLayer, modules_to_replace: List[str]) -> None:
        super().__init__(modules_to_replace=modules_to_replace)
        self._layer: Qwen2DecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def _build_compression_groups(self) -> dict[Module, List[Module]]:
        return {
            self.layer.self_attn.q_proj: [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj],
            self.layer.self_attn.o_proj: [self.layer.self_attn.o_proj],
            self.layer.mlp.gate_proj:    [self.layer.mlp.gate_proj, self.layer.mlp.up_proj],
            self.layer.mlp.down_proj:    [self.layer.mlp.down_proj],
        }


class Qwen2HeadAdapter(HeadLayerAdapterMixin, Qwen2LayerAdapter):
    def __init__(self, layer: Module, modules_to_replace: List[str]) -> None:
        super().__init__(layer=layer, modules_to_replace=modules_to_replace)

    def _build_compression_groups(self) -> dict[Module, List[Module]]:
        return {self.layer.lm_head: [self.layer.lm_head]}


class Qwen2ModelAdapter(ModelAdapter):
    def __init__(self, model: Qwen2ForCausalLM, modules_to_replace: List[str]) -> None:
        super().__init__(modules_to_replace=modules_to_replace)
        self._model: Qwen2ForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def layer_adapter_type(self) -> type:
        return Qwen2LayerAdapter

    @property
    def use_cache(self) -> bool:
        return self._model.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self._model.config.use_cache = value

    def get_layers(self) -> list[LayerAdapter]:
        adapters = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            if self._modules_to_replace is None:
                layer_modules_to_replace = None
            else:
                layer_modules_to_replace = [
                    m for m in self._modules_to_replace if m.startswith(f"model.layers.{layer_idx}.")
                ]

            adapters.append(self.layer_adapter_type(layer, modules_to_replace=layer_modules_to_replace))
        return adapters

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> List[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        return self.model.model.norm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._model.config.pad_token_id = tokenizer.pad_token_id

    def get_layer_name_prefix(self, idx) -> str:
        return f"model.layers.{idx}."

    def get_last_layer_to_output_adapter(self) -> Module:
        layer = self.get_last_layer_to_output_wrapper()

        candidate_modules = ["norm", "lm_head"]

        if self._modules_to_replace is None:
            layer_modules_to_replace = None
        else:
            def _matches_candidate(name: str) -> bool:
                return any(name == c or name.endswith(f".{c}") for c in candidate_modules)

            layer_modules_to_replace = [m for m in self._modules_to_replace if _matches_candidate(m)]

        adapter = Qwen2HeadAdapter(layer, modules_to_replace=layer_modules_to_replace)
        return adapter

    def get_last_layer_to_output_wrapper(self) -> Module:
        class _HeadWrapper(Module):
            def __init__(self, _model: Module, norm: Module = None, lm_head: Module = None) -> None:
                super().__init__()
                object.__setattr__(self, "_model", _model)

                if norm is None:
                    self.norm = getattr(_model.model, "norm")
                else:
                    self.norm = norm

                if lm_head is None:
                    self.lm_head = getattr(_model, "lm_head", None)
                else:
                    self.lm_head = lm_head

            def custom_clone(self) -> "_HeadWrapper":
                return _HeadWrapper(
                    _model=self._model,
                    norm=copy.deepcopy(self.norm),
                    lm_head=copy.deepcopy(self.lm_head),
                )

            def forward(self, *args, **kwargs) -> FloatTensor:
                x = args[0]
                x = self.norm(x)

                past_key_values = kwargs.get("past_key_values", None)

                output = BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values)
                hidden_states = output.last_hidden_state

                logits_to_keep = 0
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])

                return logits

        return _HeadWrapper(self._model)

    def finetune_layer(
        self,
        layer_adapter: LayerAdapter,
        layer_args,
        layer_kwargs,
        targets,
        device: torch.device | str,
        layer_idx: int,
        num_steps: int = 100,
        lr: float = 1e-4,
        loss_type: str = 'ce',
    ):
        layer_adapter.layer.to(device)

        optimizer = torch.optim.AdamW(layer_adapter.layer.parameters(), lr=lr, weight_decay=0.01)

        # Define schedule
        total_steps = num_steps * len(layer_args)
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))

        # Create scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        pbar = tqdm.tqdm(range(num_steps), total=num_steps, desc="Fine-tuning layer adapter")
        for step in pbar:
            epoch_loss = 0.0
            for i, (layer_args_batch, layer_kwargs_batch, target_logits) in enumerate(
                zip(layer_args, layer_kwargs, targets)
            ):
                layer_args_batch, layer_kwargs_batch, target_logits = map_tensors(
                    [layer_args_batch, layer_kwargs_batch, target_logits], device=device
                )
                logits = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)

                if loss_type == 'mse':
                    loss = torch.nn.functional.mse_loss(logits, target_logits)
                    pbar.set_postfix({'mse_loss': loss.item()})
                elif loss_type == 'kl':
                    log_probs = torch.nn.functional.log_softmax(logits / 1.0, dim=-1)
                    target_probs = torch.nn.functional.softmax(target_logits / 1.0, dim=-1)
                    loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction='batchmean') * (1.0 ** 2)
                    pbar.set_postfix({'kl_loss': loss.item()})
                elif loss_type == 'ce':
                    target_probits = torch.softmax(target_logits, dim=-1)
                    target_probits = target_probits.view(-1, target_probits.size(-1))
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_probits)
                    pbar.set_postfix({'ce_loss': loss.item(), 'ppl': torch.exp(loss).item()})
                elif loss_type == 'cosine':
                    logits_flat = logits.view(-1, logits.size(-1))
                    target_logits_flat = target_logits.view(-1, target_logits.size(-1))
                    loss = 1 - torch.nn.functional.cosine_similarity(logits_flat, target_logits_flat, dim=-1).mean()
                    pbar.set_postfix({'cosine_loss': loss.item()})
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")

                epoch_loss += loss.item()

                if wandb.run is not None:
                    wandb.log({
                        "layer_finetune_loss": loss.item(),
                        "finetune_step": step * len(layer_args) + i,
                    })

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(layer_adapter.layer.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                torch.cuda.empty_cache()

            if wandb.run is not None:
                wandb.log({
                    "layer_finetune_loss_epoch": epoch_loss,
                    "finetune_epoch": num_steps * layer_idx + step,
                })

            pbar.set_postfix({'epoch_loss': epoch_loss})

        layer_adapter.layer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
