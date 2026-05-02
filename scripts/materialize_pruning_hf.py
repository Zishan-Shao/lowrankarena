#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "Duke-CEI-SVD/LowRankArena"
DEFAULT_INDEX_PATH = REPO_ROOT / "benchmark" / "speed" / "pruning_index.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "checkpoints" / "pruning_hf"
VLLM_WRAPPER_METADATA_NAME = "vllm_wrapper_meta.json"
INDEX_FIELDNAMES = [
    "name",
    "model_family",
    "variant",
    "method",
    "source",
    "repo_id",
    "revision",
    "subpath",
    "benchmarks",
    "enabled",
    "notes",
]

BASE_MODEL_BY_PREFIX = {
    "pruning/llama31_8b/": "meta-llama/Llama-3.1-8B",
    "pruning/llama_7b/": "huggyllama/llama-7b",
}

TOKENIZER_FILENAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
}

SLICEGPT_LLAMA_MODELING = '''from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from transformers import LlamaConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


def _layer_dim(mapping: dict[str, Any], layer_idx: int, default: int) -> int:
    value = mapping.get(str(layer_idx), mapping.get(layer_idx, default))
    return int(value)


class SliceGPTLlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, *, layer_idx: int, input_dim: int, output_dim: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(getattr(config, "slicegpt_head_dim", config.hidden_size // config.num_attention_heads))
        self.hidden_size = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(input_dim, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(input_dim, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(input_dim, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, output_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, output_attentions: bool = False, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class SliceGPTLlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig, *, input_dim: int, output_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, int(config.intermediate_size), bias=False)
        self.up_proj = nn.Linear(input_dim, int(config.intermediate_size), bias=False)
        self.down_proj = nn.Linear(int(config.intermediate_size), output_dim, bias=False)
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class SliceGPTLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        slicing = getattr(config, "slicegpt_slicing_config", {}) or {}
        embedding_dim = int(getattr(config, "embedding_size", config.hidden_size))
        original_hidden = int(getattr(config, "slicegpt_original_hidden_size", config.hidden_size))
        attn_in = _layer_dim(slicing.get("attention_input_dimensions", {}), layer_idx, embedding_dim)
        attn_out = _layer_dim(slicing.get("attention_output_dimensions", {}), layer_idx, attn_in)
        mlp_in = _layer_dim(slicing.get("mlp_input_dimensions", {}), layer_idx, attn_out)
        mlp_out = _layer_dim(slicing.get("mlp_output_dimensions", {}), layer_idx, mlp_in)
        self.self_attn = SliceGPTLlamaAttention(config, layer_idx=layer_idx, input_dim=attn_in, output_dim=attn_out)
        self.mlp = SliceGPTLlamaMLP(config, input_dim=mlp_in, output_dim=mlp_out)
        self.attn_shortcut_Q = nn.Parameter(torch.empty(attn_in, attn_out))
        self.mlp_shortcut_Q = nn.Parameter(torch.empty(mlp_in, mlp_out))

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, output_attentions: bool = False, **kwargs):
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = torch.matmul(residual, self.attn_shortcut_Q) + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.matmul(residual, self.mlp_shortcut_Q) + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class SliceGPTLlamaModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    _tp_plan = {}
    _pp_plan = {}

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        slicing = getattr(config, "slicegpt_slicing_config", {}) or {}
        embedding_dim = _layer_dim(slicing.get("embedding_dimensions", {}), 0, int(getattr(config, "embedding_size", config.hidden_size)))
        config.embedding_size = embedding_dim
        config.slicegpt_original_hidden_size = int(getattr(config, "slicegpt_original_hidden_size", config.hidden_size))
        config.slicegpt_head_dim = int(getattr(config, "slicegpt_head_dim", config.slicegpt_original_hidden_size // config.num_attention_heads))
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(int(config.vocab_size), embedding_dim, self.padding_idx)
        self.layers = nn.ModuleList([SliceGPTLlamaDecoderLayer(config, idx) for idx in range(int(config.num_hidden_layers))])
        self.norm = nn.Identity()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        self.config._attn_implementation = "eager"
        kwargs.pop("attention_instances", None)
        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states)
        return_dict = bool(return_dict) if return_dict is not None else bool(self.config.use_return_dict)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attns,)
            return outputs
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class SliceGPTLlamaForCausalLM(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    _tp_plan = {}
    _pp_plan = {}

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = SliceGPTLlamaModel(config)
        self.lm_head = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, labels=None, return_dict=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return_dict = bool(return_dict) if return_dict is not None else bool(self.config.use_return_dict)
        if not return_dict:
            return (logits, outputs.last_hidden_state)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
'''

LLM_PRUNER_CONFIGURATION = '''from __future__ import annotations

from transformers import PretrainedConfig


class LLMPrunerPickleConfig(PretrainedConfig):
    model_type = "llm_pruner_pickle"

    def __init__(
        self,
        *,
        base_model: str | None = None,
        pruned_model_file: str = "llm_pruner_model.bin",
        adapter_subdir: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.pruned_model_file = pruned_model_file
        self.adapter_subdir = adapter_subdir
'''

LLM_PRUNER_MODELING = '''from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import PreTrainedModel
from transformers.generation.utils import GenerationMixin

from .configuration_llm_pruner_pickle import LLMPrunerPickleConfig


def _torch_load(path: Path, *, map_location: str = "cpu") -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class LLMPrunerPickleForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LLMPrunerPickleConfig

    def __init__(self, config: LLMPrunerPickleConfig):
        super().__init__(config)
        self.wrapped_model = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = LLMPrunerPickleConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        package_path = Path(pretrained_model_name_or_path)
        payload = _torch_load(package_path / config.pruned_model_file)
        model = payload.get("model") if isinstance(payload, dict) and "model" in payload else payload
        adapter_subdir = getattr(config, "adapter_subdir", None)
        if adapter_subdir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, package_path / adapter_subdir)
        wrapper = cls(config)
        wrapper.wrapped_model = model
        wrapper.config = getattr(model, "config", config)
        return wrapper

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.wrapped_model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.wrapped_model, "prepare_inputs_for_generation"):
            return self.wrapped_model.prepare_inputs_for_generation(*args, **kwargs)
        return super().prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self):
        return self.wrapped_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.wrapped_model.set_input_embeddings(value)
'''

BLOCKPRUNER_CONFIGURATION = '''from __future__ import annotations

from transformers import PretrainedConfig


class BlockPrunerLlamaConfig(PretrainedConfig):
    model_type = "blockpruner_llama"

    def __init__(
        self,
        *,
        base_model: str | None = None,
        mask_file: str = "del_order_list.json",
        del_block_num: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.mask_file = mask_file
        self.del_block_num = del_block_num
'''

BLOCKPRUNER_MODELING = '''from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.generation.utils import GenerationMixin

from .configuration_blockpruner_llama import BlockPrunerLlamaConfig


class MaskedLlamaDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = None
        self.mlp = None
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.mask_block = ""

    def setting_layer(self, layer):
        if "mha" not in self.mask_block:
            self.input_layernorm = layer.input_layernorm
            self.self_attn = layer.self_attn
        if "mlp" not in self.mask_block:
            self.post_attention_layernorm = layer.post_attention_layernorm
            self.mlp = layer.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        self_attn_weights = None
        present_key_value = None
        if "mha" not in self.mask_block:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual.to(attn_outputs[0].device) + attn_outputs[0]
            if output_attentions and len(attn_outputs) >= 2:
                self_attn_weights = attn_outputs[1]
            if use_cache:
                cache_idx = 2 if output_attentions else 1
                if len(attn_outputs) > cache_idx:
                    present_key_value = attn_outputs[cache_idx]

        if "mlp" not in self.mask_block:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual.to(hidden_states.device) + self.mlp(hidden_states)

        if not output_attentions and not use_cache:
            return hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


def apply_block_masks(model, sequence):
    original_layers = {}
    for block_type, block_id in sequence:
        chosen_layer = model.model.layers[block_id]
        if isinstance(chosen_layer, MaskedLlamaDecoderLayer):
            chosen_layer.mask_block += block_type
            chosen_layer.setting_layer(original_layers[str(block_id)])
        else:
            new_layer = MaskedLlamaDecoderLayer()
            new_layer.mask_block += block_type
            new_layer.setting_layer(chosen_layer)
            original_layers[str(block_id)] = chosen_layer
            model.model.layers[block_id] = new_layer
    return model


class BlockPrunerLlamaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = BlockPrunerLlamaConfig

    def __init__(self, config: BlockPrunerLlamaConfig):
        super().__init__(config)
        self.wrapped_model = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = BlockPrunerLlamaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        package_path = Path(pretrained_model_name_or_path)
        torch_dtype = kwargs.pop("torch_dtype", None)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch_dtype,
            trust_remote_code=kwargs.pop("trust_remote_code", True),
        )
        payload = json.loads((package_path / config.mask_file).read_text(encoding="utf-8"))
        sequence = payload[str(config.del_block_num)]
        model = apply_block_masks(model, sequence)
        wrapper = cls(config)
        wrapper.wrapped_model = model
        wrapper.config = getattr(model, "config", config)
        return wrapper

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.wrapped_model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.wrapped_model, "prepare_inputs_for_generation"):
            return self.wrapped_model.prepare_inputs_for_generation(*args, **kwargs)
        return super().prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self):
        return self.wrapped_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.wrapped_model.set_input_embeddings(value)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize pruning artifacts as HF-friendly non-merged checkpoint packages."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", type=Path, help="Already-downloaded pruning artifact directory.")
    source.add_argument("--checkpoint", help="Checkpoint name from --index.")
    source.add_argument("--subpath", help="HF subpath under Duke-CEI-SVD/LowRankArena, e.g. pruning/...")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", choices=["auto", "slicegpt", "llmpruner", "blockpruner"], default="auto")
    parser.add_argument("--base-model", default=None, help="Base model for tokenizer/runtime-mask wrappers.")
    parser.add_argument("--tokenizer-source", default=None, help="Tokenizer source; defaults to --base-model.")
    parser.add_argument("--del-block-num", type=int, default=None, help="BlockPruner del_block_num override.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking them.")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Do not save tokenizer files from the base model.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--record-name", default=None, help="Checkpoint name to write when --write-index is used.")
    parser.add_argument("--write-index", type=Path, default=None, help="Optional CSV index to upsert a local row into.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug or "pruning-checkpoint"


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())


def _checkpoint_row(index_path: Path, checkpoint: str) -> dict[str, str]:
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["name"] == checkpoint:
                return row
    raise KeyError(f"Checkpoint {checkpoint!r} was not found in {index_path}.")


def _download_subpath(*, repo_id: str, revision: str, subpath: str, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=[f"{subpath}/*"],
        local_files_only=local_files_only,
    )
    return Path(snapshot) / subpath


def _default_base_model(subpath_or_path: str) -> str | None:
    normalized = subpath_or_path.replace("\\", "/")
    for prefix, base_model in BASE_MODEL_BY_PREFIX.items():
        if prefix in normalized:
            return base_model
    return None


def _resolve_source(args: argparse.Namespace) -> tuple[Path, str, str | None]:
    if args.source_dir is not None:
        source_dir = args.source_dir.expanduser().resolve()
        source_name = source_dir.name
        source_subpath = str(source_dir)
    elif args.checkpoint:
        row = _checkpoint_row(args.index, args.checkpoint)
        source_subpath = row["subpath"]
        source_name = row["name"]
        source_dir = _download_subpath(
            repo_id=row.get("repo_id") or args.repo_id,
            revision=row.get("revision") or args.revision,
            subpath=source_subpath,
            local_files_only=args.local_files_only,
        )
    else:
        source_subpath = args.subpath
        source_name = Path(args.subpath).name
        source_dir = _download_subpath(
            repo_id=args.repo_id,
            revision=args.revision,
            subpath=args.subpath,
            local_files_only=args.local_files_only,
        )

    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    return source_dir.resolve(), source_name, source_subpath


def _detect_method(source_dir: Path, source_subpath: str | None, requested: str) -> str:
    if requested != "auto":
        return requested
    text = f"{source_subpath or ''}/{source_dir}".lower()
    if "slicegpt" in text:
        return "slicegpt"
    if "llmpruner" in text or "llm_pruner" in text:
        return "llmpruner"
    if "blockpruner" in text:
        return "blockpruner"
    if (source_dir / "del_order_list.json").exists():
        return "blockpruner"
    if (source_dir / "pruned_model" / "pytorch_model.bin").exists() or (source_dir / "pytorch_model.bin").exists():
        return "llmpruner"
    if list(source_dir.glob("*.pt")):
        return "slicegpt"
    raise ValueError(f"Could not infer pruning method from {source_dir}. Pass --method explicitly.")


def _output_dir_for(args: argparse.Namespace, source_name: str) -> Path:
    return (args.output_dir or (args.output_root / _slugify(source_name))).expanduser().resolve()


def _model_family_for(source_subpath: str | None, source_dir: Path) -> str:
    text = f"{source_subpath or ''}/{source_dir}".lower()
    if "llama31_8b" in text:
        return "llama3.1"
    if "llama_7b" in text:
        return "llama"
    return ""


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _upsert_index_row(
    *,
    index_path: Path,
    record_name: str,
    model_family: str,
    method: str,
    output_dir: Path,
    notes: str,
) -> None:
    rows: list[dict[str, str]] = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    row = {
        "name": record_name,
        "model_family": model_family,
        "variant": "base",
        "method": method,
        "source": "local",
        "repo_id": "",
        "revision": "main",
        "subpath": _repo_relative(output_dir),
        "benchmarks": "speed|pruning|serving",
        "enabled": "true",
        "notes": notes,
    }
    merged = {item["name"]: item for item in rows}
    merged[record_name] = row
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDNAMES)
        writer.writeheader()
        for item in sorted(merged.values(), key=lambda value: value["name"].lower()):
            writer.writerow({field: item.get(field, "") for field in INDEX_FIELDNAMES})


def _single_root_pt(source_dir: Path) -> Path:
    ignored = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}
    candidates = [path for path in source_dir.glob("*.pt") if path.is_file() and path.name not in ignored]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one root .pt weight in {source_dir}; found {[p.name for p in candidates]}")
    return candidates[0]


def _load_tensor_state_dict(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a tensor state_dict in {path}, got {type(payload).__name__}.")
    non_tensor_keys = [key for key, value in payload.items() if not torch.is_tensor(value)]
    if non_tensor_keys:
        preview = ", ".join(non_tensor_keys[:8])
        raise ValueError(f"Expected tensor-only state_dict in {path}; non-tensor keys: {preview}")
    return {str(key): value.contiguous() for key, value in payload.items()}


def _save_state_dict_as_safetensors(source_weight: Path, output_path: Path) -> dict[str, Any]:
    from safetensors.torch import save_file

    state_dict = _load_tensor_state_dict(source_weight)
    save_file(state_dict, output_path, metadata={"format": "pt"})
    total_parameters = sum(int(tensor.numel()) for tensor in state_dict.values())
    return {
        "tensor_count": len(state_dict),
        "total_parameters": total_parameters,
    }


def _slicegpt_config_path(source_dir: Path) -> Path | None:
    ignored = {
        "artifact_metadata.json",
        "config.json",
        "generation_config.json",
        "keep_ratio_calibration.json",
        "pruning_materialization.json",
        "repo_native_eval.json",
        "standardized_7task.json",
        "standardized_ppl.json",
        VLLM_WRAPPER_METADATA_NAME,
    }
    candidates = [
        path
        for path in source_dir.glob("*.json")
        if path.name not in ignored and path.is_file()
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda path: ("Llama" not in path.name, path.name))
    return candidates[0]


def _patch_slicegpt_config_for_vllm(output_dir: Path, slicing_config_path: Path | None) -> dict[str, Any]:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = _load_json(config_path)
    slicing_config: dict[str, Any] = {}
    if slicing_config_path is not None:
        copied_slicing_config_path = output_dir / slicing_config_path.name
        slicing_config = _load_json(copied_slicing_config_path if copied_slicing_config_path.exists() else slicing_config_path)
    original_hidden_size = int(slicing_config.get("hidden_size") or config.get("hidden_size") or 0)
    embedding_dimensions = slicing_config.get("embedding_dimensions") if isinstance(slicing_config, dict) else None
    embedding_size = None
    if isinstance(embedding_dimensions, dict):
        embedding_size = embedding_dimensions.get("0") or embedding_dimensions.get(0)
    if embedding_size is None:
        embedding_size = config.get("hidden_size")
    config.update(
        {
            "architectures": ["TransformersForCausalLM"],
            "auto_map": {
                **dict(config.get("auto_map") or {}),
                "AutoModel": "modeling_slicegpt_llama.SliceGPTLlamaModel",
                "AutoModelForCausalLM": "modeling_slicegpt_llama.SliceGPTLlamaForCausalLM",
            },
            "embedding_size": int(embedding_size),
            "slicegpt_head_dim": int(original_hidden_size // int(config.get("num_attention_heads", 1))),
            "slicegpt_original_hidden_size": original_hidden_size,
            "slicegpt_slicing_config": slicing_config,
            "slicegpt_slicing_config_file": slicing_config_path.name if slicing_config_path is not None else None,
        }
    )
    _dump_json(config_path, config)
    (output_dir / "modeling_slicegpt_llama.py").write_text(SLICEGPT_LLAMA_MODELING, encoding="utf-8")
    (output_dir / "__init__.py").write_text("", encoding="utf-8")
    return {
        "slicing_config_file": slicing_config_path.name if slicing_config_path is not None else None,
        "embedding_size": int(embedding_size),
        "original_hidden_size": original_hidden_size,
    }


def _save_tokenizer(output_dir: Path, tokenizer_source: str | None, *, local_files_only: bool) -> None:
    if not tokenizer_source:
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    tokenizer.save_pretrained(output_dir)


def _copy_existing_tokenizer_files(source_dir: Path, output_dir: Path, *, copy: bool) -> bool:
    found = False
    for name in TOKENIZER_FILENAMES:
        src = source_dir / name
        if src.exists():
            _link_or_copy(src, output_dir / name, copy=copy)
            found = True
    return found


def _write_vllm_meta(output_dir: Path, payload: dict[str, Any]) -> None:
    _dump_json(
        output_dir / VLLM_WRAPPER_METADATA_NAME,
        {
            "wrapper_kind": payload["materialization_kind"],
            "preserves_pruned_form": True,
            "merged": False,
            **payload,
        },
    )


def materialize_slicegpt(source_dir: Path, output_dir: Path, *, copy: bool) -> dict[str, Any]:
    source_weight = _single_root_pt(source_dir)
    if not (source_dir / "config.json").exists():
        raise FileNotFoundError(source_dir / "config.json")
    slicing_config_path = _slicegpt_config_path(source_dir)
    _reset_output_dir(output_dir)
    for src in source_dir.iterdir():
        if src == source_weight:
            continue
        _link_or_copy(src, output_dir / src.name, copy=copy or src.name == "config.json")
    safetensor_stats = _save_state_dict_as_safetensors(source_weight, output_dir / "model.safetensors")
    config_stats = _patch_slicegpt_config_for_vllm(output_dir, slicing_config_path)
    metadata = {
        "materialization_kind": "slicegpt_safetensors_state_dict",
        "source_dir": str(source_dir),
        "source_weight_file": source_weight.name,
        "standard_weight_file": "model.safetensors",
        "vllm_modeling_file": "modeling_slicegpt_llama.py",
        **config_stats,
        **safetensor_stats,
    }
    _dump_json(output_dir / "pruning_materialization.json", {**metadata, "preserves_pruned_form": True, "merged": False})
    _write_vllm_meta(output_dir, metadata)
    return metadata


def materialize_llmpruner(
    source_dir: Path,
    output_dir: Path,
    *,
    base_model: str | None,
    tokenizer_source: str | None,
    copy: bool,
    skip_tokenizer: bool,
    local_files_only: bool,
) -> dict[str, Any]:
    source_weight = source_dir / "pruned_model" / "pytorch_model.bin"
    if not source_weight.exists():
        source_weight = source_dir / "pytorch_model.bin"
    if not source_weight.exists():
        raise FileNotFoundError(f"Could not find LLMPruner pytorch_model.bin under {source_dir}")

    _reset_output_dir(output_dir)
    _link_or_copy(source_weight, output_dir / "llm_pruner_model.bin", copy=copy)
    adapter_subdir = None
    if (source_dir / "adapter").exists():
        adapter_subdir = "adapter"
        _link_or_copy(source_dir / "adapter", output_dir / adapter_subdir, copy=copy)

    (output_dir / "configuration_llm_pruner_pickle.py").write_text(LLM_PRUNER_CONFIGURATION, encoding="utf-8")
    (output_dir / "modeling_llm_pruner_pickle.py").write_text(LLM_PRUNER_MODELING, encoding="utf-8")
    (output_dir / "__init__.py").write_text("", encoding="utf-8")
    config = {
        "model_type": "llm_pruner_pickle",
        "architectures": ["TransformersForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_llm_pruner_pickle.LLMPrunerPickleConfig",
            "AutoModel": "modeling_llm_pruner_pickle.LLMPrunerPickleForCausalLM",
            "AutoModelForCausalLM": "modeling_llm_pruner_pickle.LLMPrunerPickleForCausalLM",
        },
        "base_model": base_model,
        "pruned_model_file": "llm_pruner_model.bin",
        "adapter_subdir": adapter_subdir,
    }
    _dump_json(output_dir / "config.json", config)
    if not skip_tokenizer:
        if not _copy_existing_tokenizer_files(source_dir, output_dir, copy=copy):
            _save_tokenizer(output_dir, tokenizer_source or base_model, local_files_only=local_files_only)

    metadata = {
        "materialization_kind": "llmpruner_pickled_pruned_model",
        "source_dir": str(source_dir),
        "source_weight_file": str(source_weight.relative_to(source_dir)),
        "standard_weight_file": "llm_pruner_model.bin",
        "base_model": base_model,
        "adapter_subdir": adapter_subdir,
    }
    _dump_json(output_dir / "pruning_materialization.json", {**metadata, "preserves_pruned_form": True, "merged": False})
    _write_vllm_meta(output_dir, metadata)
    return metadata


def _requested_keep_ratio(source_dir: Path) -> float | None:
    match = re.search(r"_(0\.\d+)$", source_dir.name)
    return float(match.group(1)) if match else None


def _resolve_del_block_num(source_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    metadata_path = source_dir / "artifact_metadata.json"
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
        for key in ("selected_del_block_num", "del_block_num"):
            if metadata.get(key) is not None:
                return int(metadata[key])
    calibration_path = source_dir / "keep_ratio_calibration.json"
    if calibration_path.exists():
        calibration = _load_json(calibration_path)
        target = _requested_keep_ratio(source_dir)
        mappings = calibration.get("target_mapping")
        if target is not None and isinstance(mappings, list) and mappings:
            chosen = min(
                mappings,
                key=lambda item: abs(float(item["target_keep_ratio"]) - target),
            )
            return int(chosen["selected_del_block_num"])
    raise ValueError(f"Could not infer BlockPruner del_block_num for {source_dir}; pass --del-block-num.")


def materialize_blockpruner(
    source_dir: Path,
    output_dir: Path,
    *,
    base_model: str | None,
    tokenizer_source: str | None,
    del_block_num: int | None,
    copy: bool,
    skip_tokenizer: bool,
    local_files_only: bool,
) -> dict[str, Any]:
    if base_model is None:
        raise ValueError("BlockPruner materialization requires --base-model.")
    mask_path = source_dir / "del_order_list.json"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    selected_del_block_num = _resolve_del_block_num(source_dir, del_block_num)

    _reset_output_dir(output_dir)
    _link_or_copy(mask_path, output_dir / "del_order_list.json", copy=copy)
    for optional_name in ("keep_ratio_calibration.json", "artifact_metadata.json"):
        optional_path = source_dir / optional_name
        if optional_path.exists():
            _link_or_copy(optional_path, output_dir / optional_name, copy=copy)
    (output_dir / "configuration_blockpruner_llama.py").write_text(BLOCKPRUNER_CONFIGURATION, encoding="utf-8")
    (output_dir / "modeling_blockpruner_llama.py").write_text(BLOCKPRUNER_MODELING, encoding="utf-8")
    (output_dir / "__init__.py").write_text("", encoding="utf-8")
    config = {
        "model_type": "blockpruner_llama",
        "architectures": ["TransformersForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_blockpruner_llama.BlockPrunerLlamaConfig",
            "AutoModel": "modeling_blockpruner_llama.BlockPrunerLlamaForCausalLM",
            "AutoModelForCausalLM": "modeling_blockpruner_llama.BlockPrunerLlamaForCausalLM",
        },
        "base_model": base_model,
        "mask_file": "del_order_list.json",
        "del_block_num": selected_del_block_num,
    }
    _dump_json(output_dir / "config.json", config)
    if not skip_tokenizer:
        _save_tokenizer(output_dir, tokenizer_source or base_model, local_files_only=local_files_only)

    metadata = {
        "materialization_kind": "blockpruner_runtime_mask",
        "source_dir": str(source_dir),
        "base_model": base_model,
        "mask_file": "del_order_list.json",
        "del_block_num": selected_del_block_num,
    }
    _dump_json(output_dir / "pruning_materialization.json", {**metadata, "preserves_pruned_form": True, "merged": False})
    _write_vllm_meta(output_dir, metadata)
    return metadata


def main() -> None:
    args = parse_args()
    source_dir, source_name, source_subpath = _resolve_source(args)
    method = _detect_method(source_dir, source_subpath, args.method)
    output_dir = _output_dir_for(args, source_name)
    base_model = args.base_model or _default_base_model(source_subpath or str(source_dir))
    tokenizer_source = args.tokenizer_source or base_model

    if method == "slicegpt":
        metadata = materialize_slicegpt(source_dir, output_dir, copy=args.copy)
    elif method == "llmpruner":
        metadata = materialize_llmpruner(
            source_dir,
            output_dir,
            base_model=base_model,
            tokenizer_source=tokenizer_source,
            copy=args.copy,
            skip_tokenizer=args.skip_tokenizer,
            local_files_only=args.local_files_only,
        )
    elif method == "blockpruner":
        metadata = materialize_blockpruner(
            source_dir,
            output_dir,
            base_model=base_model,
            tokenizer_source=tokenizer_source,
            del_block_num=args.del_block_num,
            copy=args.copy,
            skip_tokenizer=args.skip_tokenizer,
            local_files_only=args.local_files_only,
        )
    else:
        raise AssertionError(method)

    record_name = args.record_name or _slugify(source_name)
    if args.write_index is not None:
        _upsert_index_row(
            index_path=args.write_index,
            record_name=record_name,
            model_family=_model_family_for(source_subpath, source_dir),
            method=method,
            output_dir=output_dir,
            notes=(
                f"Local HF-friendly non-merged pruning package materialized from "
                f"{source_subpath or source_dir}."
            ),
        )

    print(json.dumps({"output_dir": str(output_dir), "method": method, **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
