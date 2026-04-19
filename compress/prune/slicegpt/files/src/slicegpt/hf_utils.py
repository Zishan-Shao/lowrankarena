# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .layernorm_fusion import fuse_modules, replace_layers
from .model_adapter import ModelAdapter, SlicingConfig
from .rotate import slice_rotated_model


def do_not_initialize(func):
    """
    A decorator that prevents initialization of torch.nn modules.
    """

    def skip(*args, **kwargs) -> None:
        pass

    def wrapper(*args, **kwargs):
        kaiming_fn = torch.nn.init.kaiming_uniform_
        uniform_fn = torch.nn.init.uniform_
        normal_fn = torch.nn.init.normal_

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        result = func(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kaiming_fn
        torch.nn.init.uniform_ = uniform_fn
        torch.nn.init.normal_ = normal_fn

        return result

    return wrapper


def format_ratio_tag(value: float | None) -> str:
    if value is None:
        raise ValueError("ratio value must be provided")
    return f"{value:.1f}"


def get_sliced_artifact_names(model_name: str, sparsity: float | None) -> tuple[list[str], list[str]]:
    """
    Return preferred sliced artifact names.

    Canonical naming uses keep ratio so that filenames align with the rest of the project.
    For backward compatibility we also accept older sparsity-based names when loading.
    """
    if sparsity is None:
        raise ValueError("sparsity must be provided when loading sliced artifacts")

    model_suffix = pathlib.Path(model_name).name
    keep_ratio = format_ratio_tag(1.0 - sparsity)
    sparsity_ratio = format_ratio_tag(sparsity)

    ordered_tags = []
    for tag in (keep_ratio, sparsity_ratio):
        if tag not in ordered_tags:
            ordered_tags.append(tag)

    weight_names = [f"{model_suffix}_{tag}.pt" for tag in ordered_tags]
    config_names = [f"{model_suffix}_{tag}.json" for tag in ordered_tags]
    return weight_names, config_names


@do_not_initialize
def get_model_and_tokenizer(
    model_name: str,
    model_path: str | None = None,
    *,
    uninitialized: bool = False,
    dtype: torch.dtype = torch.float16,
    token: str | bool | None = None,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load the model and the tokenizer from the given path.
    Set uninitialized to True when loading a pre-rotated and sliced model; in this case no weights are loaded
    in this method.
    The corresponding model adapter class must be imported before calling this method.
    Scenarios:
    - Rotate & slice HF model: model_name = name, model_path = empty, uninitialized = False
        -> Obtain the model config and weights from HF through path = name.
        -> Ignore model_path if provided.
    - Slice pre-rotated HF model: model_name = name, model_path = empty or local path, uninitialized = True
        -> Obtain the model config from HF via path = name and create uninitialized model.
        -> If the model_path is provided, confirm this use case by checking that config.json does not exist.
        -> There are no other uses of model_path in this case.
    - Rotate & slice local model: model_name = name, model_path = local path, uninitialized = False
        -> Obtain the model config through path, and the pretrained weights from the local path.
        -> Use the model name only to determine the correct model adapter to use.
    - Slice pre-rotated local model: model_name = name, model_path = local path, uninitialized = True
        -> Obtain the model config from the local path and create an uninitialized model.
        -> Use the model name only to determine the correct model adapter to use.
        -> Confirm this case by checking that config.json exists.
    """
    model_type = "uninitialized" if uninitialized else "pretrained"
    local_model = model_path is not None

    if local_model and uninitialized:
        local_model = (pathlib.Path(model_path) / "config.json").exists()

    # for HF models the path to use is the model name
    if not local_model:
        model_path = model_name

    logging.info(
        f"Loading %s config %s from %s",
        model_name,
        "and model weights" if not uninitialized else "",
        model_path if local_model else 'Hugging Face',
    )

    model_adapter = ModelAdapter.from_model(
        model_name,
        model_path=model_path,
        model_type=model_type,
        dtype=dtype,
        local_files_only=local_model,
        token=token,
    )

    model = model_adapter.model
    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model_adapter.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token, local_files_only=local_model)

    model_adapter.post_init(tokenizer)
    logging.info("Loading model done")

    return model_adapter, tokenizer


@do_not_initialize
def load_sliced_model(
    model_name: str,
    sliced_model_path: str,
    *,
    token: str | None = None,
    lora_config: Any = None,
    sparsity: float | None = None,
    round_interval: int | None = 1,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load the sliced model and the tokenizer from the given path. If lora_config: peft.LoraConfig is supplied
    as an arg then this function will return a PEFT model (post-slicing finetuned model). Despite being declared as
    "Any", lora_config is supposed to have the type peft.LoraConfig. It has type "Any" in the function's signature,
    so that it would be possible to use it without taking a dependency on peft, when one is not required.
    The corresponding model adapter class must be imported before calling this method.
    """
    weight_name_candidates, config_name_candidates = get_sliced_artifact_names(model_name, sparsity)
    sliced_model_dir = pathlib.Path(sliced_model_path)

    model_adapter, tokenizer = get_model_and_tokenizer(
        model_name,
        model_path=sliced_model_path,
        uninitialized=True,
        token=token,
    )
    replace_layers(model_adapter)
    fuse_modules(model_adapter)

    hidden_size = model_adapter.hidden_size
    for layer_adapter in model_adapter.get_layers():
        if not model_adapter.parallel_blocks:
            layer_adapter.layer.mlp_shortcut_Q = torch.nn.Parameter(
                torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
            )
        layer_adapter.layer.attn_shortcut_Q = torch.nn.Parameter(
            torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
        )

    config_path = next((sliced_model_dir / name for name in config_name_candidates if (sliced_model_dir / name).exists()), None)

    if config_path is not None:
        model_adapter.slicing_conf = SlicingConfig.from_json_string(config_path.read_text())

    if model_adapter.slicing_conf is None:
        # assume the model was sliced with the const sparsity specified in the arguments to this method
        new_embedding_dimension = int((1 - sparsity) * hidden_size)
        new_embedding_dimension -= new_embedding_dimension % round_interval
        config = SlicingConfig()
        config.const_dimension = new_embedding_dimension
        model_adapter.slicing_conf = config

    slice_rotated_model(model_adapter)

    if lora_config:
        from peft import get_peft_model

        model_adapter.model = get_peft_model(model_adapter.model, lora_config)

    weight_path = next((sliced_model_dir / name for name in weight_name_candidates if (sliced_model_dir / name).exists()), None)
    if weight_path is None:
        raise FileNotFoundError(
            f"Could not find sliced model weights in {sliced_model_path}. Tried: {', '.join(weight_name_candidates)}"
        )

    logging.info(f"Loading sliced model weights from {weight_path}")
    model_adapter.model.load_state_dict(
        torch.load(str(weight_path), map_location="cpu")
    )
    model_adapter.model.eval()

    return model_adapter, tokenizer
