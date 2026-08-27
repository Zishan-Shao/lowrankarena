#coding:utf8
import os
import sys
import warnings
from typing import Dict

import torch
import torch.nn as nn

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")

def get_model_from_huggingface(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    from transformers.utils import logging as transformers_logging

    # Suppress transformers logging to hide LlamaTokenizer legacy behavior message
    prev_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()

    # Suppress known FutureWarnings during model loading
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

        name = (model_id or "").lower()

        # Use fast tokenizer only for DeepSeek
        if "deepseek" in name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,
            )
        elif "llama-3" in name or "Llama-3" in model_id:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
        elif ("llama" in name) or ("mistral" in name) or ("vicuna" in name):
            tokenizer = LlamaTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                legacy=True,
            )
        else:
            # OPT and everything else: force slow tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=False,
            )

        # Ensure pad_token is set
        # if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            # tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True)

    # Normalize BOS token for reproducibility across different tokenizer cache versions
    # if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id != 0:
    #     print(f"⚠️  Normalizing bos_token_id from {tokenizer.bos_token_id} to 0 for reproducibility")
    #     tokenizer.bos_token_id = 0

    # Restore original logging verbosity
    transformers_logging.set_verbosity(prev_verbosity)

    model.seqlen = 2048
    return model, tokenizer

def get_model_from_local(model_id):
    pruned_dict = torch.load(model_id, weights_only=False, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


# ---------------------------------------------------------------------------
# Low-rank linear layer
# ---------------------------------------------------------------------------


class LowRankLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that factors the weight into U/V
    components, mirroring how SVDLLM modules store q_u/q_v, etc.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.v_proj = nn.Linear(in_features, rank, bias=False)
        self.u_proj = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u_proj(self.v_proj(x))


# ---------------------------------------------------------------------------
# Target-module discovery for the compression pipeline
# ---------------------------------------------------------------------------


def get_layers(model_name: str, model):
    name = (model_name or "").lower()

    # DeepSeek family (LLaMA-style layout in practice)
    if "deepseek" in name:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        # Some wrappers nest one level deeper
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
            return model.model.model.layers
        raise AttributeError(f"Cannot find DeepSeek layers on object type {type(model)}")

    # LLaMA, Mistral, Vicuna family
    if ("llama" in name) or ("mistral" in name) or ("vicuna" in name):
        # Case A: model is a CausalLM wrapper, has .model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        # Case B: model is the base model itself, has .layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError(f"Cannot find layers on object type {type(model)}")

    # OPT family
    if "opt" in name:
        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            return model.model.decoder.layers
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers
        raise AttributeError(f"Cannot find OPT decoder layers on object type {type(model)}")

    raise ValueError(f"Unsupported model architecture: {model_name}")


TARGET_LEAF_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def find_target_modules_in_layer(layer) -> Dict[str, nn.Module]:
    """
    Find target modules in a layer, treating LowRankLinear as a single module.

    Returns a dict mapping logical module name, for example 'self_attn.q_proj',
    'self_attn.v_proj', 'mlp.gate_proj', to the module (either LowRankLinear
    or nn.Linear).
    """
    target_mods: Dict[str, nn.Module] = {}

    for name, mod in layer.named_modules():
        parts = name.split(".")
        leaf = parts[-1]

        # Only consider leaf names that we care about
        if leaf not in TARGET_LEAF_NAMES:
            continue

        # Case 1: the wrapper is already LowRankLinear, use it directly
        if isinstance(mod, LowRankLinear):
            target_mods[name] = mod
            continue

        # Case 2: dense nn.Linear projection
        if isinstance(mod, nn.Linear):
            # We must exclude inner u_proj and v_proj that live inside a LowRankLinear
            # For those inner modules, the parent name is itself a projection name
            # Example:
            #   'self_attn.v_proj'          -> parent leaf is 'self_attn'
            #   'self_attn.v_proj.v_proj'   -> parent leaf is 'v_proj' (wrapper)
            parent_leaf = parts[-2] if len(parts) > 1 else None

            # If parent name is itself a projection leaf, this is almost surely
            # an internal linear inside LowRankLinear, skip it.
            if parent_leaf in TARGET_LEAF_NAMES:
                continue

            # Otherwise this is a true top level projection like self_attn.v_proj
            target_mods[name] = mod

    return target_mods
