# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0

import copy
import gc
import logging
from math import cos, pi
from typing import List

import torch
import tqdm
import wandb
from torch import FloatTensor
from torch.nn import Linear, Module
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
)

from aa_svd.compression.compressed_linear import CompressedLinear
from aa_svd.compression.model_adapter import LayerAdapter, ModelAdapter, HeadLayerAdapterMixin
from aa_svd.compression.utils import map_tensors

logger = logging.getLogger(__name__)


class LlamaLayerAdapter(LayerAdapter):
    def __init__(self, layer: LlamaDecoderLayer, modules_to_replace: List[str]) -> None:
        super().__init__(modules_to_replace=modules_to_replace)
        self._layer: LlamaDecoderLayer = layer

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


class LlamaHeadAdapter(HeadLayerAdapterMixin, LlamaLayerAdapter):
    def __init__(self, layer: Module, modules_to_replace: List[str]) -> None:
        super().__init__(layer=layer, modules_to_replace=modules_to_replace)

    def _build_compression_groups(self) -> dict[Module, List[Module]]:
        return {self.layer.lm_head: [self.layer.lm_head]}


class LlamaModelAdapter(ModelAdapter):
    def __init__(self, model: LlamaForCausalLM, modules_to_replace: List[str]) -> None:
        super().__init__(modules_to_replace=modules_to_replace)
        self._model: LlamaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def layer_adapter_type(self) -> type:
        return LlamaLayerAdapter

    @property
    def use_cache(self) -> bool:
        return self._model.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self._model.config.use_cache = value

    def get_layers(self) -> list[LayerAdapter]:
        adapters = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Filter the configured module names to only those that are present in this layer
            if self._modules_to_replace is None:
                layer_modules_to_replace = None
            else:
                layer_modules_to_replace = [m for m in self._modules_to_replace if m.startswith(f"model.layers.{layer_idx}.")]

            adapters.append(self.layer_adapter_type(layer, modules_to_replace=layer_modules_to_replace))
        return adapters

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> List[Module]:
        return [self.model.model.embed_tokens]

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Llama-2 and Llama-3 don't have a pad token by default
        tokenizer.pad_token = tokenizer.eos_token
        self._model.config.pad_token_id = tokenizer.pad_token_id

    def get_layer_name_prefix(self, idx) -> str:
        """
        Returns the prefix used to identify layers in the model. For example, in Llama, this would be "model.layers."
        """
        return f"model.layers.{idx}."

    def get_last_layer_to_output_adapter(self) -> Module:
        """
        Returns the module that produces the final output from the last
        layer's output. For Llama this wraps the model's final
        normalization (RMSNorm) and the language-model head (lm_head)
        so compression can target the lm_head.
        """
        layer = self.get_last_layer_to_output_wrapper()

        # Candidate exposed names on the wrapper. The wrapper exposes the
        # final model normalization as `norm` and the linear head as
        # `head` (backed by the model's `lm_head`). When the user supplies
        # configured module names, accept either exact matches or names
        # that end with the candidate (e.g. "model.norm" or "lm_head").
        candidate_modules = ["norm", "lm_head"]

        if self._modules_to_replace is None:
            layer_modules_to_replace = None
        else:
            def _matches_candidate(name: str) -> bool:
                return any(name == c or name.endswith(f".{c}") for c in candidate_modules)

            layer_modules_to_replace = [m for m in self._modules_to_replace if _matches_candidate(m)]

        adapter = LlamaHeadAdapter(layer, modules_to_replace=layer_modules_to_replace)
        return adapter

    def get_last_layer_to_output_wrapper(self) -> Module:
        """
        Returns a wrapper module that takes the last layer's output and
        produces the final logits. The wrapper exposes `norm` (the
        final RMSNorm on the inner model) and `head` (the top-level
        `lm_head`) so adapters can discover and replace them.
        """

        class _HeadWrapper(Module):
            """Wraps Llama's final normalization and lm_head so the adapter
            can treat them like a single layer. The wrapper exposes two
            registered submodules: `norm` (the inner model's RMSNorm) and
            `head` (the top-level `lm_head` Linear), when present.
            """

            def __init__(self, _model: Module, norm: Module = None, lm_head: Module = None) -> None:
                super().__init__()
                # Keep a reference to the full original model without
                # registering it as a submodule to avoid duplicating
                # entries in named_modules().
                object.__setattr__(self, "_model", _model)

                # Expose the final layer norm from the inner model if
                if norm is None:
                    self.norm = getattr(_model.model, "norm")
                else:
                    self.norm = norm

                # Expose the language-model head (lm_head) that produces
                # logits from hidden states. This is typically a Linear.
                if lm_head is None:
                    self.lm_head = getattr(_model, "lm_head", None)
                else:
                    self.lm_head = lm_head

            def custom_clone(self) -> '_HeadWrapper':
                return _HeadWrapper(
                    _model=self._model,
                    norm=copy.deepcopy(self.norm),
                    lm_head=copy.deepcopy(self.lm_head)
                )

            def forward(self, *args, **kwargs) -> FloatTensor:
                """Apply final norm (if present) and return a
                BaseModelOutputWithPast containing the normalized hidden
                states. This mirrors LlamaForCausalLM where the
                `lm_head` is applied outside the inner model using
                `outputs.last_hidden_state`.

                Expects hidden_states of shape (batch, seq_len, hidden_dim).
                Returns a BaseModelOutputWithPast with `last_hidden_state`
                set to the normalized hidden states and `past_key_values`
                forwarded from kwargs when present.
                """
                x = args[0]
                x = self.norm(x)

                # Forward any provided past_key_values (if present) so the
                # caller can access them from the returned object.
                past_key_values = kwargs.get("past_key_values", None)

                output = BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values)
                hidden_states = output.last_hidden_state

                logits_to_keep = 0
                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])

                return logits

        return _HeadWrapper(self._model)

    def finetune_layer(self, layer_adapter, layer_args, layer_kwargs, targets, device: torch.device | str,
                      layer_idx, num_steps: int = 100, lr: float = 1e-4, loss_type: str = 'ce'):
        layer_adapter.layer.to(device)

        optimizer = torch.optim.AdamW(layer_adapter.layer.parameters(), lr=lr, weight_decay=0.01)

        # Define schedule
        total_steps = num_steps*len(layer_args)
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))

        # Create scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # loss_fn = torch.nn.CrossEntropyLoss()

        pbar = tqdm.tqdm(range(num_steps), total=num_steps, desc="Fine-tuning layer adapter")
        for step in pbar:
            epoch_loss = 0.
            for i, (layer_args_batch, layer_kwargs_batch, target_logits) in enumerate(zip(layer_args, layer_kwargs, targets)):
                layer_args_batch, layer_kwargs_batch, target_logits = map_tensors(
                                                            [layer_args_batch, layer_kwargs_batch, target_logits], device=device
                                                        )
                logits = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
                # hidden_state_pos = layer_adapter.hidden_states_output_position
                # logits = out_batch[hidden_state_pos]

                # print('logits shape:', logits.shape)
                # print('target_logits shape:', target_logits.shape)

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
                        f"finetune_step_loss/layer_{layer_idx}": loss.item(),
                        "finetune_step": step*len(layer_args) + i
                    })

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(layer_adapter.layer.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # del batch, original_outputs, adapted_outputs, loss
                torch.cuda.empty_cache()

            if wandb.run is not None:
                wandb.log({
                    f"finetune_epoch_loss/layer_{layer_idx}": epoch_loss,
                    "finetune_epoch": step})

            pbar.set_postfix({'epoch_loss': epoch_loss})

        layer_adapter.layer.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
