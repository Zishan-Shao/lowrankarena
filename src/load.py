from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.registry import CheckpointRecord, DEFAULT_INDEX_PATH, get_checkpoint
from src.utils import project_path


HF_URI_PREFIX = "hf://"


@dataclass(slots=True)
class LoadedCheckpoint:
    record: CheckpointRecord
    locator: str
    loader: str
    metadata: dict[str, Any]
    local_path: str | None = None
    config: Any | None = None
    tokenizer: Any | None = None
    model: Any | None = None


class CheckpointLoadError(RuntimeError):
    pass


def build_locator(record: CheckpointRecord) -> str:
    if record.source == "huggingface":
        suffix = f"{record.repo_id}@{record.revision}/{record.subpath}".rstrip("/")
        return f"{HF_URI_PREFIX}{suffix}"
    if record.source == "local":
        return str(project_path(record.subpath).resolve())
    return record.subpath or record.name


def build_hf_kwargs(
    record: CheckpointRecord,
    *,
    token: str | None = None,
    trust_remote_code: bool = True,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "revision": record.revision,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if record.subpath:
        kwargs["subfolder"] = record.subpath
    if token:
        kwargs["token"] = token
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    return kwargs


def _resolve_hf_token(explicit_token: str | None = None) -> str | None:
    if explicit_token:
        return explicit_token
    try:
        from huggingface_hub import HfFolder
    except ImportError:
        return None
    return HfFolder.get_token()


def download_hf_snapshot(
    record: CheckpointRecord,
    *,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional runtime packages.
        raise CheckpointLoadError("huggingface_hub is required to materialize Hugging Face checkpoints.") from exc

    resolved_token = _resolve_hf_token(token)
    resolved_patterns = allow_patterns
    if resolved_patterns is None and record.subpath:
        resolved_patterns = [f"{record.subpath}/*"]
    snapshot_path = snapshot_download(
        repo_id=record.repo_id,
        repo_type="model",
        revision=record.revision,
        allow_patterns=resolved_patterns,
        token=resolved_token,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )
    return Path(snapshot_path)


def load_from_record(
    record: CheckpointRecord,
    *,
    load_config: bool = False,
    load_tokenizer: bool = False,
    load_model: bool = False,
    download: bool = False,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    device_map: str | dict[str, Any] | None = "auto",
    torch_dtype: str | Any = "auto",
) -> LoadedCheckpoint:
    locator = build_locator(record)

    if record.source == "local":
        local_path = Path(record.subpath).expanduser()
        if not local_path.is_absolute():
            local_path = project_path(record.subpath)
        return LoadedCheckpoint(
            record=record,
            locator=str(local_path.resolve()),
            loader="local",
            local_path=str(local_path.resolve()),
            metadata={"status": "resolved", "source": "local"},
        )

    if record.source != "huggingface":
        return LoadedCheckpoint(
            record=record,
            locator=locator,
            loader="unimplemented",
            metadata={
                "status": "unimplemented",
                "message": f"Unsupported checkpoint source: {record.source}",
            },
        )

    resolved_token = _resolve_hf_token(token)
    metadata: dict[str, Any] = {
        "status": "resolved",
        "source": "huggingface",
        "repo_id": record.repo_id,
        "revision": record.revision,
        "subpath": record.subpath,
        "has_token": bool(resolved_token),
    }

    local_path: str | None = None
    if download:
        snapshot_path = download_hf_snapshot(
            record,
            allow_patterns=allow_patterns,
            token=resolved_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        local_path = str((snapshot_path / record.subpath).resolve()) if record.subpath else str(snapshot_path.resolve())
        metadata["status"] = "downloaded"
        metadata["snapshot_path"] = str(snapshot_path.resolve())

    if not any([load_config, load_tokenizer, load_model]):
        return LoadedCheckpoint(
            record=record,
            locator=locator,
            loader="huggingface",
            metadata=metadata,
            local_path=local_path,
        )

    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime packages.
        raise CheckpointLoadError("transformers is required to load Hugging Face checkpoints.") from exc

    hf_kwargs = build_hf_kwargs(
        record,
        token=resolved_token,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    config = None
    tokenizer = None
    model = None
    try:
        if load_config:
            config = AutoConfig.from_pretrained(record.repo_id, **hf_kwargs)
        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(record.repo_id, **hf_kwargs)
        if load_model:
            model = AutoModelForCausalLM.from_pretrained(
                record.repo_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                **hf_kwargs,
            )
    except Exception as exc:
        subpath_hint = f"{record.repo_id}/{record.subpath}".rstrip("/")
        raise CheckpointLoadError(
            "Failed to load Hugging Face checkpoint "
            f"'{record.name}' from '{subpath_hint}'. "
            "Check that your HF token has accepted the gated repo and that the subpath is correct."
        ) from exc

    metadata["status"] = "loaded"
    metadata["loaded_config"] = load_config
    metadata["loaded_tokenizer"] = load_tokenizer
    metadata["loaded_model"] = load_model
    return LoadedCheckpoint(
        record=record,
        locator=locator,
        loader="transformers",
        metadata=metadata,
        local_path=local_path,
        config=config,
        tokenizer=tokenizer,
        model=model,
    )


def load_checkpoint(
    name: str,
    index_path: str | None = None,
    *,
    load_config: bool = False,
    load_tokenizer: bool = False,
    load_model: bool = False,
    download: bool = False,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    device_map: str | dict[str, Any] | None = "auto",
    torch_dtype: str | Any = "auto",
) -> LoadedCheckpoint:
    resolved_index = index_path or str(DEFAULT_INDEX_PATH)
    record = get_checkpoint(name, path=resolved_index)
    return load_from_record(
        record,
        load_config=load_config,
        load_tokenizer=load_tokenizer,
        load_model=load_model,
        download=download,
        allow_patterns=allow_patterns,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
