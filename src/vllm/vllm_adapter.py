from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.load import LoadedCheckpoint
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
        cache_root = Path(wrapper_cache_root or DEFAULT_WRAPPER_CACHE_ROOT).expanduser().resolve()
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

    return PreparedVllmModel(
        model_path=str(source_model_path),
        tokenizer_path=str(source_model_path),
        tokenizer_mode="auto",
        model_impl=None,
        preparation_kind="direct",
        source_model_path=str(source_model_path),
        notes=["checkpoint looks directly loadable by vLLM"],
    )
