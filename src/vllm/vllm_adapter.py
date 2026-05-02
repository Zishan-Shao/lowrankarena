from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.inference_adapter import prepare_model_for_inference
from src.load import LoadedCheckpoint
from src.modeling.artifacts import (
    ensure_asvd_llama_vllm_wrapper,
    ensure_basis_sharing_llama_wrapper,
)
from src.utils import dump_json, load_json, user_cache_path
from .prepare_svdllm_vllm_model import (
    WRAPPER_METADATA_NAME,
    ensure_svdllm_llama_wrapper,
)


DEFAULT_WRAPPER_CACHE_ROOT = Path(
    os.environ.get("LRA_VLLM_CACHE_ROOT", str(user_cache_path("vllm")))
).expanduser()
PRUNING_WEIGHT_WRAPPER_METADATA_NAME = "pruning_weight_wrapper_meta.json"
TRANSFORMERS_BACKEND_WRAPPER_METADATA_NAME = "transformers_backend_wrapper_meta.json"

_STANDARD_WEIGHT_FILENAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}
_IGNORED_PT_FILENAMES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "training_args.bin",
}
_ASVD_QWEN3_VLLM_MODELING = '''try:
    from transformers import Qwen3Model
except ImportError:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

try:
    from .configuration_asvd_qwen3 import ASVDQwen3Config
    from .modeling_asvd_qwen3 import ASVDLinear
except ImportError:
    from configuration_asvd_qwen3 import ASVDQwen3Config
    from modeling_asvd_qwen3 import ASVDLinear

import torch.nn as nn


class ASVDQwen3ModelForVLLM(Qwen3Model):
    config_class = ASVDQwen3Config

    def __init__(self, config: ASVDQwen3Config):
        super().__init__(config)
        self.truncation_ranks = config.truncation_ranks or {}
        if not self.truncation_ranks:
            return

        full_name_dict = {module: name for name, module in self.named_modules()}
        linear_info = {}
        modules = [self]
        while modules:
            submodule = modules.pop()
            for child_name, child in submodule.named_children():
                if isinstance(child, nn.Linear):
                    full_name = full_name_dict[child]
                    linear_info[child] = {
                        "father": submodule,
                        "name": child_name,
                        "full_name": full_name,
                    }
                else:
                    modules.append(child)

        rank_by_local_name = {}
        for name, rank in self.truncation_ranks.items():
            if name.startswith("model."):
                rank_by_local_name[name[len("model."):]] = rank
            else:
                rank_by_local_name[name] = rank

        for full_name, module in list(self.named_modules()):
            if full_name not in rank_by_local_name:
                continue
            info = linear_info[module]
            new_layer = ASVDLinear(
                module.in_features,
                module.out_features,
                int(rank_by_local_name[full_name]),
                bias=module.bias is not None,
            )
            setattr(info["father"], info["name"], new_layer)
'''
_MODEGPT_QWEN3_VLLM_MODELING = '''try:
    from .DenseQwenRebuild import Qwen3Model as _Qwen3Model
except ImportError:
    from DenseQwenRebuild import Qwen3Model as _Qwen3Model


class Qwen3ModelForVLLM(_Qwen3Model):
    def forward(self, *args, **kwargs):
        self.config._attn_implementation = "eager"
        kwargs.pop("attention_instances", None)
        return super().forward(*args, **kwargs)
'''
_MODEGPT_LLAMA_VLLM_MODELING = '''try:
    from .LlamaRebuild import LlamaModel as _LlamaModel
except ImportError:
    from LlamaRebuild import LlamaModel as _LlamaModel


class LlamaModelForVLLM(_LlamaModel):
    def forward(self, *args, **kwargs):
        self.config._attn_implementation = "eager"
        kwargs.pop("attention_instances", None)
        return super().forward(*args, **kwargs)
'''


@dataclass(slots=True)
class PreparedVllmModel:
    model_path: str
    tokenizer_path: str
    tokenizer_mode: str
    model_impl: str | None
    preparation_kind: str
    source_model_path: str
    extra_llm_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def build_tokenizer_kwargs(self, *, trust_remote_code: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if self.tokenizer_mode == "slow":
            kwargs["use_fast"] = False
        return kwargs

    def build_llm_kwargs(
        self,
        *,
        tensor_parallel_size: int,
        trust_remote_code: bool,
        gpu_memory_utilization: float,
        dtype: str,
        enforce_eager: bool,
        max_model_len: int | None,
        disable_log_stats: bool = True,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "tokenizer": self.tokenizer_path,
            "tokenizer_mode": self.tokenizer_mode,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
            "disable_log_stats": disable_log_stats,
            "use_tqdm_on_load": False,
        }
        if self.model_impl is not None:
            kwargs["model_impl"] = self.model_impl
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        kwargs.update(self.extra_llm_kwargs)
        return kwargs


def _slugify(text: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return collapsed or "model"


def _local_model_path(loaded: LoadedCheckpoint) -> Path:
    if loaded.local_path:
        return Path(loaded.local_path).expanduser().resolve()
    raise ValueError(
        "prepare_model_for_vllm requires a local checkpoint path. "
        "Call load_checkpoint(..., download=True) or pass a local source."
    )


def _config_for(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return load_json(config_path)


def _tokenizer_mode_for(model_path: Path, config: dict[str, Any]) -> str:
    if config.get("model_type") == "svdllm-llama":
        return "slow"
    if not (model_path / "tokenizer.json").exists():
        return "slow"
    return "auto"


def _is_ready_svdllm_wrapper(model_path: Path, config: dict[str, Any]) -> bool:
    auto_map = dict(config.get("auto_map") or {})
    metadata_path = model_path / WRAPPER_METADATA_NAME
    return (
        config.get("model_type") == "svdllm-llama"
        and list(config.get("architectures") or []) == ["TransformersForCausalLM"]
        and auto_map.get("AutoModel") == "modeling_svdllm_llama.SVDLLMLlamaModel"
        and auto_map.get("AutoModelForCausalLM") == "modeling_svdllm_llama.SVDLLMLlamaForCausalLM"
        and (model_path / "modeling_svdllm_llama.py").exists()
        and (model_path / "configuration_svdllm_llama.py").exists()
        and metadata_path.exists()
    )


def _is_transformers_ready_wrapper(model_path: Path, config: dict[str, Any]) -> bool:
    auto_map = dict(config.get("auto_map") or {})
    return (
        list(config.get("architectures") or []) == ["TransformersForCausalLM"]
        and "AutoModel" in auto_map
        and "AutoModelForCausalLM" in auto_map
        and (model_path / WRAPPER_METADATA_NAME).exists()
    )


def _auto_causal_model_name(config: dict[str, Any]) -> str:
    auto_map = dict(config.get("auto_map") or {})
    return str(auto_map.get("AutoModelForCausalLM") or "")


def _is_asvd_llama_checkpoint(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    causal_model = _auto_causal_model_name(config)
    return "ASVDLlamaForCausalLM" in architectures or causal_model.endswith("ASVDLlamaForCausalLM")


def _is_asvd_qwen3_checkpoint(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    causal_model = _auto_causal_model_name(config)
    return "ASVDQwen3ForCausalLM" in architectures or causal_model.endswith("ASVDQwen3ForCausalLM")


def _is_dobi_checkpoint(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    causal_model = _auto_causal_model_name(config)
    return any(name in architectures for name in {"DobiSVDLlamaForCausalLM", "DobiSVDQwen3ForCausalLM"}) or (
        causal_model.endswith("DobiSVDLlamaForCausalLM") or causal_model.endswith("DobiSVDQwen3ForCausalLM")
    )


def _is_lowrank_qwen3_checkpoint(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    causal_model = _auto_causal_model_name(config)
    return (
        config.get("model_type") == "lowrank_qwen3"
        and "LowRankQwen3ForCausalLM" in architectures
        and causal_model.endswith("LowRankQwen3ForCausalLM")
    )


def _is_native_qwen3_checkpoint(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    return config.get("model_type") == "qwen3" and "Qwen3ForCausalLM" in architectures


def _is_modegpt_checkpoint(config: dict[str, Any]) -> bool:
    causal_model = _auto_causal_model_name(config)
    return causal_model in {
        "LlamaRebuild.LlamaForCausalLM",
        "DenseQwenRebuild.Qwen3ForCausalLM",
    }


def _modegpt_auto_model_name(config: dict[str, Any]) -> str | None:
    causal_model = _auto_causal_model_name(config)
    if causal_model == "LlamaRebuild.LlamaForCausalLM":
        return "modeling_lra_vllm_modegpt.LlamaModelForVLLM"
    if causal_model == "DenseQwenRebuild.Qwen3ForCausalLM":
        return "modeling_lra_vllm_modegpt.Qwen3ModelForVLLM"
    return None


def _modegpt_extra_files(config: dict[str, Any]) -> dict[str, str]:
    causal_model = _auto_causal_model_name(config)
    if causal_model == "LlamaRebuild.LlamaForCausalLM":
        return {"modeling_lra_vllm_modegpt.py": _MODEGPT_LLAMA_VLLM_MODELING}
    if causal_model == "DenseQwenRebuild.Qwen3ForCausalLM":
        return {"modeling_lra_vllm_modegpt.py": _MODEGPT_QWEN3_VLLM_MODELING}
    return {}


def _default_wrapper_dir(loaded: LoadedCheckpoint, cache_root: Path) -> Path:
    if loaded.record.name:
        base = _slugify(loaded.record.name)
    else:
        base = _slugify(_local_model_path(loaded).name)
    return cache_root / f"{base}_vllm"


def _has_standard_transformers_weights(model_path: Path) -> bool:
    if any((model_path / name).exists() for name in _STANDARD_WEIGHT_FILENAMES):
        return True
    return bool(list(model_path.glob("model-*.safetensors")) or list(model_path.glob("pytorch_model-*.bin")))


def _is_pruning_checkpoint_candidate(loaded: LoadedCheckpoint, model_path: Path) -> bool:
    method = loaded.record.method.lower().replace("-", "_")
    subpath = loaded.record.subpath.lower().replace("\\", "/")
    path_text = str(model_path).lower().replace("\\", "/")
    return (
        "pruning/" in subpath
        or "pruning/" in path_text
        or method in {"slicegpt", "slice_gpt", "llm_pruner", "llmpruner", "blockpruner"}
    )


def _single_root_pt_weight(model_path: Path) -> Path | None:
    if _has_standard_transformers_weights(model_path):
        return None
    candidates = [
        path
        for path in model_path.glob("*.pt")
        if path.is_file() and path.name not in _IGNORED_PT_FILENAMES
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _can_reuse_pruning_weight_wrapper(
    output_dir: Path,
    *,
    source_model_path: Path,
    source_weight_path: Path,
) -> bool:
    metadata_path = output_dir / PRUNING_WEIGHT_WRAPPER_METADATA_NAME
    converted_weight_path = output_dir / "model.safetensors"
    if not metadata_path.exists() or not converted_weight_path.exists():
        return False
    try:
        metadata = load_json(metadata_path)
    except Exception:
        return False
    return (
        metadata.get("wrapper_kind") == "pruning_root_pt_safetensors_wrapper_v1"
        and metadata.get("source_model") == str(source_model_path.resolve())
        and metadata.get("source_weight_file") == str(source_weight_path.resolve())
    )


def _save_tensor_state_dict_as_safetensors(source_weight_path: Path, output_path: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    payload = torch.load(source_weight_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Expected tensor state_dict in {source_weight_path}, got {type(payload).__name__}.")
    non_tensor_keys = [key for key, value in payload.items() if not torch.is_tensor(value)]
    if non_tensor_keys:
        preview = ", ".join(str(key) for key in non_tensor_keys[:8])
        raise ValueError(f"Expected tensor-only state_dict in {source_weight_path}; non-tensor keys: {preview}")
    tensor_state = {str(key): value.contiguous() for key, value in payload.items()}
    save_file(tensor_state, output_path, metadata={"format": "pt"})
    return {
        "tensor_count": len(tensor_state),
        "total_parameters": sum(int(tensor.numel()) for tensor in tensor_state.values()),
    }


def _materialize_pruning_weight_wrapper(
    *,
    source_model_path: Path,
    output_dir: Path,
    source_weight_path: Path,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        if _can_reuse_pruning_weight_wrapper(
            output_dir,
            source_model_path=source_model_path,
            source_weight_path=source_weight_path,
        ):
            return output_dir
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for src in source_model_path.iterdir():
        dst = output_dir / src.name
        if src == source_weight_path:
            continue
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    safetensor_stats = _save_tensor_state_dict_as_safetensors(source_weight_path, output_dir / "model.safetensors")
    dump_json(
        {
            "wrapper_kind": "pruning_root_pt_safetensors_wrapper_v1",
            "source_model": str(source_model_path.resolve()),
            "source_weight_file": str(source_weight_path.resolve()),
            "converted_weight_file": "model.safetensors",
            "preserves_pruned_form": True,
            "merged": False,
            **safetensor_stats,
        },
        output_dir / PRUNING_WEIGHT_WRAPPER_METADATA_NAME,
    )
    return output_dir


def _maybe_prepare_pruning_weight_wrapper(
    loaded: LoadedCheckpoint,
    model_path: Path,
    cache_root: Path,
) -> tuple[Path, list[str]]:
    if not _is_pruning_checkpoint_candidate(loaded, model_path):
        return model_path, []
    source_weight_path = _single_root_pt_weight(model_path)
    if source_weight_path is None:
        return model_path, []

    wrapper_dir = _default_wrapper_dir(loaded, cache_root)
    materialized_path = _materialize_pruning_weight_wrapper(
        source_model_path=model_path,
        output_dir=wrapper_dir,
        source_weight_path=source_weight_path,
    )
    return materialized_path, [
        f"pruning checkpoint root weight {source_weight_path.name!r} "
        "was converted to model.safetensors for vLLM/Transformers loading without merging"
    ]


def _can_reuse_transformers_backend_wrapper(
    output_dir: Path,
    *,
    source_model_path: Path,
    wrapper_kind: str,
    auto_model_override: str | None,
) -> bool:
    metadata_path = output_dir / TRANSFORMERS_BACKEND_WRAPPER_METADATA_NAME
    config_path = output_dir / "config.json"
    if not metadata_path.exists() or not config_path.exists():
        return False
    try:
        metadata = load_json(metadata_path)
        config = load_json(config_path)
    except Exception:
        return False
    return (
        metadata.get("wrapper_kind") == wrapper_kind
        and metadata.get("source_model") == str(source_model_path.resolve())
        and metadata.get("auto_model_override") == auto_model_override
        and list(config.get("architectures") or []) == ["TransformersForCausalLM"]
    )


def _materialize_transformers_backend_wrapper(
    *,
    source_model_path: Path,
    output_dir: Path,
    wrapper_kind: str,
    auto_model_override: str | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        if _can_reuse_transformers_backend_wrapper(
            output_dir,
            source_model_path=source_model_path,
            wrapper_kind=wrapper_kind,
            auto_model_override=auto_model_override,
        ):
            return output_dir
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    extra_files = extra_files or {}
    for src in source_model_path.iterdir():
        if src.name in {"config.json", TRANSFORMERS_BACKEND_WRAPPER_METADATA_NAME} or src.name in extra_files:
            continue
        dst = output_dir / src.name
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    for name, content in extra_files.items():
        (output_dir / name).write_text(content, encoding="utf-8")

    config = _config_for(source_model_path)
    original_architectures = list(config.get("architectures") or [])
    config["architectures"] = ["TransformersForCausalLM"]
    auto_map = dict(config.get("auto_map") or {})
    if auto_model_override is not None:
        auto_map["AutoModel"] = auto_model_override
    elif "AutoModelForCausalLM" in auto_map and "AutoModel" not in auto_map:
        auto_map["AutoModel"] = auto_map["AutoModelForCausalLM"]
    if auto_map:
        config["auto_map"] = auto_map
    dump_json(config, output_dir / "config.json")
    dump_json(
        {
            "wrapper_kind": wrapper_kind,
            "source_model": str(source_model_path.resolve()),
            "original_architectures": original_architectures,
            "architectures": ["TransformersForCausalLM"],
            "preserves_remote_code_model": True,
            "auto_model_override": auto_model_override,
        },
        output_dir / TRANSFORMERS_BACKEND_WRAPPER_METADATA_NAME,
    )
    return output_dir


def prepare_model_for_vllm(
    loaded: LoadedCheckpoint,
    *,
    wrapper_cache_root: str | Path | None = None,
) -> PreparedVllmModel:
    source_model_path = _local_model_path(loaded)
    config = _config_for(source_model_path)
    cache_root = Path(wrapper_cache_root or DEFAULT_WRAPPER_CACHE_ROOT).expanduser().resolve()

    if _is_ready_svdllm_wrapper(source_model_path, config):
        return PreparedVllmModel(
            model_path=str(source_model_path),
            tokenizer_path=str(source_model_path),
            tokenizer_mode="slow",
            model_impl="transformers",
            preparation_kind="already_prepared_svdllm_llama_wrapper",
            source_model_path=str(source_model_path),
            notes=["checkpoint already exposes AutoModel and can be handed directly to vLLM"],
        )

    if config.get("model_type") == "svdllm-llama":
        wrapper_dir = _default_wrapper_dir(loaded, cache_root)
        materialized = ensure_svdllm_llama_wrapper(source_model_path, wrapper_dir)
        notes = [
            "factorized linear weights stay in low-rank form as u_proj(v_proj(x))",
            "KV cache stays dense",
            "vLLM must use tokenizer_mode=slow and model_impl=transformers for this wrapper",
        ]
        if materialized.reused:
            notes.append("reused an existing compatible wrapper directory")
        return PreparedVllmModel(
            model_path=str(materialized.output_dir),
            tokenizer_path=str(materialized.output_dir),
            tokenizer_mode="slow",
            model_impl="transformers",
            preparation_kind="svdllm_llama_wrapper",
            source_model_path=str(source_model_path),
            notes=notes,
        )

    if _is_asvd_qwen3_checkpoint(config):
        wrapper_dir = _default_wrapper_dir(loaded, cache_root)
        materialized = _materialize_transformers_backend_wrapper(
            source_model_path=source_model_path,
            output_dir=wrapper_dir,
            wrapper_kind="asvd_qwen3_transformers_backend_wrapper_v1",
            auto_model_override="modeling_lra_vllm_asvd_qwen3.ASVDQwen3ModelForVLLM",
            extra_files={"modeling_lra_vllm_asvd_qwen3.py": _ASVD_QWEN3_VLLM_MODELING},
        )
        return PreparedVllmModel(
            model_path=str(materialized),
            tokenizer_path=str(materialized),
            tokenizer_mode=_tokenizer_mode_for(materialized, _config_for(materialized)),
            model_impl="transformers",
            preparation_kind="asvd_qwen3_transformers_wrapper",
            source_model_path=str(source_model_path),
            notes=[
                "ASVD Qwen3 checkpoints preserve custom low-rank modules through remote-code AutoModel classes.",
                "A lightweight wrapper rewrites architectures to TransformersForCausalLM so vLLM honors the remote-code AutoModel class.",
                "KV cache stays dense.",
            ],
        )

    if _is_asvd_llama_checkpoint(config):
        wrapper_dir = _default_wrapper_dir(loaded, cache_root)
        materialized = ensure_asvd_llama_vllm_wrapper(source_model_path, wrapper_dir)
        notes = [
            "ASVD low-rank layers stay factorized in the base model.",
            "vLLM's Transformers backend owns lm_head, so the wrapper materializes a dense lm_head.weight only.",
            "KV cache stays dense.",
        ]
        if materialized.reused:
            notes.append("reused an existing compatible vLLM wrapper directory")
        return PreparedVllmModel(
            model_path=str(materialized.output_dir),
            tokenizer_path=str(materialized.output_dir),
            tokenizer_mode=_tokenizer_mode_for(materialized.output_dir, _config_for(materialized.output_dir)),
            model_impl="transformers",
            preparation_kind="asvd_llama_vllm_wrapper",
            source_model_path=str(source_model_path),
            notes=notes,
        )

    if _is_dobi_checkpoint(config):
        wrapper_dir = _default_wrapper_dir(loaded, cache_root)
        materialized = _materialize_transformers_backend_wrapper(
            source_model_path=source_model_path,
            output_dir=wrapper_dir,
            wrapper_kind="dobi_transformers_backend_wrapper_v1",
        )
        return PreparedVllmModel(
            model_path=str(materialized),
            tokenizer_path=str(materialized),
            tokenizer_mode=_tokenizer_mode_for(materialized, _config_for(materialized)),
            model_impl="transformers",
            preparation_kind="dobi_transformers_wrapper",
            source_model_path=str(source_model_path),
            notes=[
                "DoBi checkpoints are mixed-format: modules listed in dobi_target_modules stay factorized as BLinear(ALinear(x)).",
                "Any module not listed in dobi_target_modules stays dense.",
                "A lightweight wrapper rewrites architectures to TransformersForCausalLM so vLLM honors the remote-code AutoModel class.",
            ],
        )

    if _is_lowrank_qwen3_checkpoint(config):
        wrapper_dir = _default_wrapper_dir(loaded, cache_root)
        materialized = _materialize_transformers_backend_wrapper(
            source_model_path=source_model_path,
            output_dir=wrapper_dir,
            wrapper_kind="lowrank_qwen3_transformers_backend_wrapper_v1",
        )
        return PreparedVllmModel(
            model_path=str(materialized),
            tokenizer_path=str(materialized),
            tokenizer_mode=_tokenizer_mode_for(materialized, _config_for(materialized)),
            model_impl="transformers",
            preparation_kind="lowrank_qwen3_transformers_wrapper",
            source_model_path=str(source_model_path),
            notes=[
                "Qwen3 SVD-LLM/Basis Sharing checkpoints preserve low-rank modules through LowRankQwen3ForCausalLM.",
                "A lightweight wrapper rewrites architectures to TransformersForCausalLM so vLLM honors the remote-code AutoModel class.",
                "KV cache stays dense.",
            ],
        )

    if _is_modegpt_checkpoint(config):
        wrapper_dir = _default_wrapper_dir(loaded, cache_root)
        materialized = _materialize_transformers_backend_wrapper(
            source_model_path=source_model_path,
            output_dir=wrapper_dir,
            wrapper_kind="modegpt_transformers_backend_wrapper_v1",
            auto_model_override=_modegpt_auto_model_name(config),
            extra_files=_modegpt_extra_files(config),
        )
        return PreparedVllmModel(
            model_path=str(materialized),
            tokenizer_path=str(materialized),
            tokenizer_mode=_tokenizer_mode_for(materialized, _config_for(materialized)),
            model_impl="transformers",
            preparation_kind="modegpt_transformers_wrapper",
            source_model_path=str(source_model_path),
            notes=[
                "MoDeGPT checkpoints preserve rebuilt low-rank modules through custom remote-code AutoModel classes.",
                "A lightweight wrapper rewrites architectures to TransformersForCausalLM so vLLM honors LlamaRebuild/DenseQwenRebuild.",
                "KV cache stays dense.",
            ],
        )

    if _is_native_qwen3_checkpoint(config):
        return PreparedVllmModel(
            model_path=str(source_model_path),
            tokenizer_path=str(source_model_path),
            tokenizer_mode=_tokenizer_mode_for(source_model_path, config),
            model_impl="transformers",
            preparation_kind="qwen3_transformers_direct",
            source_model_path=str(source_model_path),
            notes=[
                "Dense Qwen3 is directly loadable, but this local vLLM stack is more reliable with the Transformers backend.",
                "KV cache stays dense.",
            ],
        )

    prepared = prepare_model_for_inference(loaded)
    prepared_model_path = Path(prepared.model_path)
    prepared_model_path, pruning_notes = _maybe_prepare_pruning_weight_wrapper(
        loaded,
        prepared_model_path,
        cache_root,
    )
    prepared_config = _config_for(prepared_model_path)

    if _is_transformers_ready_wrapper(prepared_model_path, prepared_config):
        notes = list(prepared.notes) + pruning_notes
        notes.append("wrapper already exposes AutoModel and can be handed directly to vLLM.")
        return PreparedVllmModel(
            model_path=str(prepared_model_path),
            tokenizer_path=str(prepared_model_path) if pruning_notes else prepared.tokenizer_path,
            tokenizer_mode=prepared.tokenizer_mode,
            model_impl="transformers",
            preparation_kind=prepared.preparation_kind,
            source_model_path=prepared.source_model_path,
            notes=notes,
        )

    if prepared.preparation_kind == "basis_sharing_llama_wrapper":
        wrapper_dir = _default_wrapper_dir(loaded, cache_root)
        materialized = ensure_basis_sharing_llama_wrapper(source_model_path, wrapper_dir)
        notes = list(prepared.notes)
        if materialized.reused:
            notes.append("reused an existing compatible vLLM wrapper directory")
        return PreparedVllmModel(
            model_path=str(materialized.output_dir),
            tokenizer_path=str(materialized.output_dir),
            tokenizer_mode=_tokenizer_mode_for(materialized.output_dir, _config_for(materialized.output_dir)),
            model_impl="transformers",
            preparation_kind="basis_sharing_llama_wrapper",
            source_model_path=str(source_model_path),
            notes=notes,
        )

    return PreparedVllmModel(
        model_path=str(prepared_model_path),
        tokenizer_path=str(prepared_model_path) if pruning_notes else prepared.tokenizer_path,
        tokenizer_mode=prepared.tokenizer_mode,
        model_impl=None,
        preparation_kind="pruning_root_pt_safetensors_wrapper" if pruning_notes else prepared.preparation_kind,
        source_model_path=prepared.source_model_path,
        notes=list(prepared.notes) + pruning_notes + ["checkpoint looks directly loadable by vLLM"],
    )
