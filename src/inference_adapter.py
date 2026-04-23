from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dtype_utils import normalize_config_torch_dtype_name
from src.load import LoadedCheckpoint
from src.modeling.artifacts import ensure_basis_sharing_llama_wrapper
from src.utils import dump_json, load_json, user_cache_path


DEFAULT_WRAPPER_CACHE_ROOT = Path(
    os.environ.get("LRA_INFERENCE_CACHE_ROOT", str(user_cache_path("inference")))
).expanduser()
CONFIG_DTYPE_WRAPPER_METADATA_NAME = "config_dtype_wrapper_meta.json"


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


def _config_with_normalized_torch_dtype(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if "torch_dtype" not in config or config.get("torch_dtype") is None:
        return config, False

    normalized = normalize_config_torch_dtype_name(config.get("torch_dtype"))
    if config.get("torch_dtype") == normalized:
        return config, False

    updated = dict(config)
    updated["torch_dtype"] = normalized
    return updated, True


def _can_reuse_config_dtype_wrapper(
    output_dir: Path,
    *,
    source_model_path: Path,
    normalized_torch_dtype: str,
) -> bool:
    metadata_path = output_dir / CONFIG_DTYPE_WRAPPER_METADATA_NAME
    config_path = output_dir / "config.json"
    if not metadata_path.exists() or not config_path.exists():
        return False
    try:
        metadata = load_json(metadata_path)
        config = load_json(config_path)
    except Exception:
        return False
    return (
        metadata.get("wrapper_kind") == "config_torch_dtype_wrapper_v1"
        and metadata.get("source_model") == str(source_model_path.resolve())
        and metadata.get("normalized_torch_dtype") == normalized_torch_dtype
        and config.get("torch_dtype") == normalized_torch_dtype
    )


def _symlink_checkpoint_contents(source_model_path: Path, output_dir: Path) -> None:
    for src in source_model_path.iterdir():
        if src.name == "config.json":
            continue
        dst = output_dir / src.name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())


def _materialize_config_dtype_wrapper(
    *,
    source_model_path: Path,
    output_dir: Path,
    source_config: dict[str, Any],
    normalized_config: dict[str, Any],
) -> Path:
    normalized_torch_dtype = str(normalized_config.get("torch_dtype"))
    if output_dir.exists() and any(output_dir.iterdir()):
        if _can_reuse_config_dtype_wrapper(
            output_dir,
            source_model_path=source_model_path,
            normalized_torch_dtype=normalized_torch_dtype,
        ):
            return output_dir
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    _symlink_checkpoint_contents(source_model_path, output_dir)
    dump_json(normalized_config, output_dir / "config.json")
    dump_json(
        {
            "wrapper_kind": "config_torch_dtype_wrapper_v1",
            "source_model": str(source_model_path.resolve()),
            "source_checkpoint_name": source_model_path.name,
            "source_torch_dtype": source_config.get("torch_dtype"),
            "normalized_torch_dtype": normalized_torch_dtype,
        },
        output_dir / CONFIG_DTYPE_WRAPPER_METADATA_NAME,
    )
    return output_dir


def _ensure_config_torch_dtype_compatible(
    loaded: LoadedCheckpoint,
    source_model_path: Path,
    config: dict[str, Any],
    cache_root: Path,
) -> tuple[Path, dict[str, Any], list[str]]:
    normalized_config, changed = _config_with_normalized_torch_dtype(config)
    if not changed:
        return source_model_path, config, []

    wrapper_dir = _default_wrapper_dir(loaded, cache_root, "config")
    materialized_path = _materialize_config_dtype_wrapper(
        source_model_path=source_model_path,
        output_dir=wrapper_dir,
        source_config=config,
        normalized_config=normalized_config,
    )
    return materialized_path, normalized_config, [
        "config torch_dtype was normalized in a local wrapper for this Transformers version"
    ]


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
    original_model_path = _local_model_path(loaded)
    source_model_path = original_model_path
    config = _config_for(source_model_path)
    cache_root = Path(wrapper_cache_root or DEFAULT_WRAPPER_CACHE_ROOT).expanduser().resolve()
    source_model_path, config, compatibility_notes = _ensure_config_torch_dtype_compatible(
        loaded,
        source_model_path,
        config,
        cache_root,
    )

    if _is_basis_sharing_checkpoint(config):
        wrapper_dir = _default_wrapper_dir(loaded, cache_root, "hf")
        materialized = ensure_basis_sharing_llama_wrapper(source_model_path, wrapper_dir)
        notes = compatibility_notes + ["basis sharing checkpoint requires a local wrapper that restores the custom runtime."]
        if materialized.reused:
            notes.append("reused an existing compatible inference wrapper directory")
        return PreparedInferenceModel(
            model_path=str(materialized.output_dir),
            tokenizer_path=str(materialized.output_dir),
            tokenizer_mode=_tokenizer_mode_for(materialized.output_dir, _config_for(materialized.output_dir)),
            preparation_kind="basis_sharing_llama_wrapper",
            source_model_path=str(original_model_path),
            notes=notes,
        )

    if _is_dobi_llama_checkpoint(config):
        return PreparedInferenceModel(
            model_path=str(source_model_path),
            tokenizer_path=str(source_model_path),
            tokenizer_mode=_tokenizer_mode_for(source_model_path, config),
            preparation_kind="dobi_llama_direct",
            source_model_path=str(original_model_path),
            notes=compatibility_notes
            + [
                "DoBi checkpoints are mixed-format: modules listed in dobi_target_modules stay factorized as BLinear(ALinear(x)).",
                "Any module not listed in dobi_target_modules stays dense under the upstream Llama runtime.",
                "The checkpoint already exposes custom AutoConfig and AutoModel entries, so Transformers can load it directly.",
            ],
        )

    return PreparedInferenceModel(
        model_path=str(source_model_path),
        tokenizer_path=str(source_model_path),
        tokenizer_mode=_tokenizer_mode_for(source_model_path, config),
        preparation_kind="config_torch_dtype_wrapper" if compatibility_notes else "direct",
        source_model_path=str(original_model_path),
        notes=compatibility_notes + ["checkpoint is directly loadable through Transformers from the local snapshot"],
    )
