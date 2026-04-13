from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.load import LoadedCheckpoint
from src.modeling.artifacts import ensure_basis_sharing_llama_wrapper
from src.utils import load_json, user_cache_path


DEFAULT_WRAPPER_CACHE_ROOT = Path(
    os.environ.get("LRA_INFERENCE_CACHE_ROOT", str(user_cache_path("inference")))
).expanduser()


@dataclass(slots=True)
class PreparedInferenceModel:
    model_path: str
    tokenizer_path: str
    tokenizer_mode: str
    preparation_kind: str
    source_model_path: str
    notes: list[str] = field(default_factory=list)

    def build_tokenizer_kwargs(self, *, trust_remote_code: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if self.tokenizer_mode == "slow":
            kwargs["use_fast"] = False
        return kwargs


def _slugify(text: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return collapsed or "model"


def _local_model_path(loaded: LoadedCheckpoint) -> Path:
    if loaded.local_path:
        return Path(loaded.local_path).expanduser().resolve()
    raise ValueError(
        "prepare_model_for_inference requires a local checkpoint path. "
        "Call load_checkpoint(..., download=True) or pass a local source."
    )


def _config_for(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return load_json(config_path)


def _default_wrapper_dir(loaded: LoadedCheckpoint, cache_root: Path, suffix: str) -> Path:
    if loaded.record.name:
        base = _slugify(loaded.record.name)
    else:
        base = _slugify(_local_model_path(loaded).name)
    return cache_root / f"{base}_{suffix}"


def _tokenizer_mode_for(model_path: Path, config: dict[str, Any]) -> str:
    if config.get("model_type") == "svdllm-llama":
        return "slow"
    if not (model_path / "tokenizer.json").exists():
        return "slow"
    return "auto"


def _is_basis_sharing_checkpoint(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    return "ShareLlamaForCausalLM" in architectures or config.get("model_type") == "basis_sharing_llama"


def _is_dobi_llama_checkpoint(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    auto_map = dict(config.get("auto_map") or {})
    return "DobiSVDLlamaForCausalLM" in architectures or auto_map.get("AutoModelForCausalLM", "").endswith(
        "DobiSVDLlamaForCausalLM"
    )


def prepare_model_for_inference(
    loaded: LoadedCheckpoint,
    *,
    wrapper_cache_root: str | Path | None = None,
) -> PreparedInferenceModel:
    source_model_path = _local_model_path(loaded)
    config = _config_for(source_model_path)
    cache_root = Path(wrapper_cache_root or DEFAULT_WRAPPER_CACHE_ROOT).expanduser().resolve()

    if _is_basis_sharing_checkpoint(config):
        wrapper_dir = _default_wrapper_dir(loaded, cache_root, "hf")
        materialized = ensure_basis_sharing_llama_wrapper(source_model_path, wrapper_dir)
        notes = ["basis sharing checkpoint requires a local wrapper that restores the custom runtime."]
        if materialized.reused:
            notes.append("reused an existing compatible inference wrapper directory")
        return PreparedInferenceModel(
            model_path=str(materialized.output_dir),
            tokenizer_path=str(materialized.output_dir),
            tokenizer_mode=_tokenizer_mode_for(materialized.output_dir, _config_for(materialized.output_dir)),
            preparation_kind="basis_sharing_llama_wrapper",
            source_model_path=str(source_model_path),
            notes=notes,
        )

    if _is_dobi_llama_checkpoint(config):
        return PreparedInferenceModel(
            model_path=str(source_model_path),
            tokenizer_path=str(source_model_path),
            tokenizer_mode=_tokenizer_mode_for(source_model_path, config),
            preparation_kind="dobi_llama_direct",
            source_model_path=str(source_model_path),
            notes=[
                "DoBi checkpoints are mixed-format: modules listed in dobi_target_modules stay factorized as BLinear(ALinear(x)).",
                "Any module not listed in dobi_target_modules stays dense under the upstream Llama runtime.",
                "The checkpoint already exposes custom AutoConfig and AutoModel entries, so Transformers can load it directly.",
            ],
        )

    return PreparedInferenceModel(
        model_path=str(source_model_path),
        tokenizer_path=str(source_model_path),
        tokenizer_mode=_tokenizer_mode_for(source_model_path, config),
        preparation_kind="direct",
        source_model_path=str(source_model_path),
        notes=["checkpoint is directly loadable through Transformers from the local snapshot"],
    )
