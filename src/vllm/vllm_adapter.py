from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.inference_adapter import prepare_model_for_inference
from src.load import LoadedCheckpoint
from src.modeling.artifacts import (
    ensure_asvd_llama_vllm_wrapper,
    ensure_basis_sharing_llama_wrapper,
)
from src.utils import load_json, project_path
from .prepare_svdllm_vllm_model import (
    WRAPPER_METADATA_NAME,
    ensure_svdllm_llama_wrapper,
)


DEFAULT_WRAPPER_CACHE_ROOT = project_path("checkpoints", "vllm")


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


def _is_asvd_checkpoint(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    auto_map = dict(config.get("auto_map") or {})
    return "ASVDLlamaForCausalLM" in architectures or auto_map.get("AutoModelForCausalLM", "").endswith(
        "ASVDLlamaForCausalLM"
    )


def _is_dobi_checkpoint(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    auto_map = dict(config.get("auto_map") or {})
    return "DobiSVDLlamaForCausalLM" in architectures or auto_map.get("AutoModelForCausalLM", "").endswith(
        "DobiSVDLlamaForCausalLM"
    )


def _default_wrapper_dir(loaded: LoadedCheckpoint, cache_root: Path) -> Path:
    if loaded.record.name:
        base = _slugify(loaded.record.name)
    else:
        base = _slugify(_local_model_path(loaded).name)
    return cache_root / f"{base}_vllm"


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

    if _is_asvd_checkpoint(config):
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
        return PreparedVllmModel(
            model_path=str(source_model_path),
            tokenizer_path=str(source_model_path),
            tokenizer_mode=_tokenizer_mode_for(source_model_path, config),
            model_impl="transformers",
            preparation_kind="dobi_llama_direct",
            source_model_path=str(source_model_path),
            notes=[
                "DoBi checkpoints are mixed-format: modules listed in dobi_target_modules stay factorized as BLinear(ALinear(x)).",
                "Any module not listed in dobi_target_modules stays dense.",
                "vLLM must use the Transformers backend for this custom AutoModel checkpoint.",
            ],
        )

    prepared = prepare_model_for_inference(loaded)
    prepared_model_path = Path(prepared.model_path)
    prepared_config = _config_for(prepared_model_path)

    if _is_transformers_ready_wrapper(prepared_model_path, prepared_config):
        notes = list(prepared.notes)
        notes.append("wrapper already exposes AutoModel and can be handed directly to vLLM.")
        return PreparedVllmModel(
            model_path=str(prepared_model_path),
            tokenizer_path=prepared.tokenizer_path,
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
        tokenizer_path=prepared.tokenizer_path,
        tokenizer_mode=prepared.tokenizer_mode,
        model_impl=None,
        preparation_kind=prepared.preparation_kind,
        source_model_path=prepared.source_model_path,
        notes=list(prepared.notes) + ["checkpoint looks directly loadable by vLLM"],
    )
