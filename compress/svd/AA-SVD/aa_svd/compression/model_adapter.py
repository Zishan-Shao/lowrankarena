# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

import os
import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, List

import torch
from torch.nn import Linear, Module
from transformers import PreTrainedTokenizerBase
import logging

from .utils import get_submodule, replace_module
from .compressed_linear import CompressedLinear, QuantizedCompressedLinear

logger = logging.getLogger(__name__)


"""
To add support for a new model, you need to create a new adapter class that inherits from ModelAdapter, and a new
adapter class that inherits from LayerAdapter. The ModelAdapter class tells sliceGPT how to interact with the model,
an instance of which is stored at self.model. For example, how to access each of the layers of the model. Similarly,
the LayerAdapter class tells sliceGPT how to interact with each layer of the model. For example, how to access the
attention and MLP components of the layer, and how to update the arguments to the layer's forward method.
See src/slicegpt/adapters/llama_adapter.py for an example of how to implement these classes.
"""


class LayerAdapter(ABC):
    """
    To implement a new layer adapter, implement the interface defined in this class
    """

    def __init__(self, modules_to_replace: List[str]) -> None:
        super().__init__()
        self._modules_to_replace = modules_to_replace

    @property
    @abstractmethod
    def layer(self) -> Module:
        """
        Instance of the transformer layer to be wrapped. This contains the forward() method of the original model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_args_position(self) -> int:
        """
        Returns the position of the hidden_states argument in the layer's forward method.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_output_position(self) -> int:
        """
        Returns the position of the hidden_states in the output of the layer's forward method.
        """
        raise NotImplementedError

    def clone(self) -> LayerAdapter:
        """
        Returns a deep copy of the LayerAdapter and its underlying layer.
        """
        return copy.deepcopy(self)

    @abstractmethod
    def _build_compression_groups(self) -> dict[Module, List[Module]]:
        """
        Returns the raw grouping of modules for compression, before filtering.
        Keys are representative modules; values are lists of all modules in the group
        (including the key). Subclasses need not filter for Linear type or to_compress —
        that is handled by get_compression_mapping.
        """
        raise NotImplementedError

    def get_compression_mapping(self) -> dict[Module, List[Module]]:
        """
        Returns an ordered mapping from a representative module to its compression group,
        filtered to Linear modules that pass `to_compress`, with empty groups removed.
        """
        raw = self._build_compression_groups()
        filtered = {k: [m for m in v if isinstance(m, Linear) and self.to_compress(m)] for k, v in raw.items()}
        return {k: v for k, v in filtered.items() if v}

    def get_compression_order(self) -> list[Module]:
        """
        Returns the representative modules in compression order, filtered to those present
        in the compression mapping.
        """
        return list(self.get_compression_mapping().keys())

    def to_compress(self, module: Module) -> bool:
        """Returns True if the module should be compressed per the configured modules list."""
        from .compressed_linear import CompressedLinear  # local import avoids circularity at class-def time
        if isinstance(module, CompressedLinear):
            return False
        if self._modules_to_replace is None:
            return False
        module_name = self.get_module_name(module)
        for configured_name in self._modules_to_replace:
            if module_name == configured_name or configured_name.endswith(f".{module_name}"):
                return True
        return False

    def get_module_name(self, module: Module) -> str:
        """Returns the dotted name of `module` relative to this layer, or raises ValueError."""
        for name, mod in self.layer.named_modules():
            if mod is module:
                return name
        raise ValueError("Module not found in layer")

    def replace_module(self, current_module: Module, new_module: Module) -> None:
        """Replace a submodule within this layer in-place."""
        for attr_name in dir(self.layer):
            if getattr(self.layer, attr_name) is current_module:
                setattr(self.layer, attr_name, new_module)
                return
        for name, mod in self.layer.named_modules():
            if mod is current_module:
                parts = name.split('.')
                parent = self.layer
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_module)
                return
        raise ValueError(f"Module {current_module} not found in layer")

    def get_updated_args(self, hidden_states: Any, args: tuple) -> tuple:
        """
        `args` is a tuple of the arguments to the layer's forward method. hidden_states is the new value for the
        hidden_states argument. This method returns a new tuple of arguments with the hidden_states argument updated.
        """
        return (
            args[:self.hidden_states_args_position] + (hidden_states,) + args[self.hidden_states_args_position + 1:]
        )


class HeadLayerAdapterMixin:
    """
    Mixin for head adapters that wrap a model's final norm + lm_head inside a
    _HeadWrapper. Provides wrapper-aware replace_module (which mirrors changes
    onto the backing _model) and a generic clone via self.__class__.
    """

    @staticmethod
    def _set_attr_by_path(obj: Module, parts: list[str], value: Module) -> bool:
        try:
            parent = obj
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], value)
            return True
        except Exception:
            return False

    def replace_module(self, current_module: Module, new_module: Module) -> None:
        for attr_name in dir(self.layer):
            try:
                attr_value = getattr(self.layer, attr_name)
            except Exception:
                continue
            if attr_value is current_module:
                setattr(self.layer, attr_name, new_module)
                original_model = getattr(self.layer, "_model", None)
                if original_model is not None and hasattr(original_model, attr_name):
                    try:
                        setattr(original_model, attr_name, new_module)
                    except Exception:
                        pass
                return

        for name, mod in self.layer.named_modules():
            if mod is current_module:
                parts = name.split('.') if name else [name]
                if self._set_attr_by_path(self.layer, parts, new_module):
                    original_model = getattr(self.layer, "_model", None)
                    if original_model is not None:
                        self._set_attr_by_path(original_model, parts, new_module)
                    return

        raise ValueError(f"Module {current_module} not found in layer")

    def clone(self):
        return self.__class__(layer=self.layer.custom_clone(), modules_to_replace=self._modules_to_replace)


class ModelAdapter(ABC):
    """
    To implement a new model adapter, implement the interface defined in this class
    """

    def __init__(self, modules_to_replace: List[str]) -> None:
        super().__init__()
        self._modules_to_replace = modules_to_replace

    def load(self, path: str, layer_adapter=None, layer_idx=None, dobi_remapping=False) -> None:

        if layer_adapter is None:
            modules_to_replace = self._modules_to_replace
        else:
            modules_to_replace = layer_adapter._modules_to_replace

        loaded = False
        for module_name in modules_to_replace:
            current_module = get_submodule(self.model, module_name)
            assert isinstance(current_module, torch.nn.Linear)

            load_path = f"{path}/{module_name.replace('.', '_')}"

            if os.path.exists(load_path):
                loader = QuantizedCompressedLinear if dobi_remapping else CompressedLinear
                new_module = loader.from_path(load_path, bias=current_module.bias)
                replace_module(self.model, module_name, new_module)
                loaded = True
                logging.info(f"Loaded compressed module for {module_name} from {load_path}")
            else:
                logging.warning(f"No saved module found for {module_name} at {load_path}, skipping load.")

        finetuned = False
        if loaded:
            # also load norms if available
            if layer_adapter is not None:
                layer_name_prefix = self.get_layer_name_prefix(layer_idx)

                # find all norm modules in the layer adapter
                for module_name, module in layer_adapter.layer.named_modules():
                    module_name = f"{layer_name_prefix}{module_name}"
                    if 'norm' in module_name.lower():
                        load_path = f"{path}/{module_name.replace('.', '_')}/state_dict.pt"
                        if os.path.exists(load_path):
                            finetuned = True
                            module.load_state_dict(torch.load(load_path))
                            logging.info(f"Loaded norm module for {module_name} from {load_path}")
                        else:
                            finetuned = False
                            logging.warning(f"No saved norm module found for {module_name} at {load_path}, skipping load.")

        not_finetuned = not finetuned

        return not_finetuned

    @property
    @abstractmethod
    def model(self) -> Module:
        """
        The original model to compress.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def layer_adapter_type(self) -> type:
        """
        Type of the class implementing the LayerAdapter interface
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def use_cache(self) -> bool:
        """Must define a setter"""
        raise NotImplementedError

    @use_cache.setter
    @abstractmethod
    def use_cache(self, value: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_layers(self) -> Sequence[LayerAdapter]:
        """
        Returns a list of LayerAdapters, one for each layer in the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_raw_layer_at(self, index: int) -> Module:
        """
        Returns the raw layer (no adapter) at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        """
        Assigns the given layer to the model at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self) -> List[Module]:
        """
        Returns a list of the embedding modules in the model.
        """
        raise NotImplementedError


    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Called after the model is initialized. Override to set pad tokens or other tokenizer-level setup.
        """
        pass

    @abstractmethod
    def get_layer_name_prefix(self, idx=None) -> str:
        """
        Returns the prefix used to identify layers in the model. For example, in Llama, this would be "model.layers."
        """
        raise NotImplementedError

    @abstractmethod
    def get_last_layer_to_output_adapter(self) -> LayerAdapter:
        """
        Returns the adapter that captures the remaining forward pass from the last LayerAdapter. The inputs are the outputs of the last
        LayerAdapter. The adapter.layer should encapsulate all transformations as well as sub-modules after the last layer (LayerAdapter),
        up to the final output logits.
        """
        raise NotImplementedError

    @abstractmethod
    def finetune_layer(
        self,
        layer_adapter: LayerAdapter,
        layer_args: list,
        layer_kwargs: list,
        targets: list,
        device,
        layer_idx: int,
        num_steps: int = 100,
        lr: float = 1e-4,
        loss_type: str = 'ce',
    ) -> None:
        """
        Fine-tune a single transformer layer in-place using the provided activations and targets.
        """
        raise NotImplementedError