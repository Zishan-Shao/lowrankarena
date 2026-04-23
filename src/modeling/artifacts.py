from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.dtype_utils import normalize_config_torch_dtype_name


WRAPPER_METADATA_NAME = "inference_wrapper_meta.json"
LM_HEAD_DENSE_SHARD_NAME = "model-lm_head.safetensors"
LM_HEAD_CHUNK_ROWS = 4096

_PASSTHROUGH_NAMES = [
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELING_ROOT = _REPO_ROOT / "src" / "modeling"
_LLAMA_MODELING_ROOT = _MODELING_ROOT / "llama"


@dataclass(slots=True)
class WrapperMaterialization:
    output_dir: Path
    created: bool
    reused: bool
    metadata: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def symlink_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def _read_repo_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _copy_repo_support_file(src: Path, dst: Path) -> None:
    write_text(dst, _read_repo_file(src))


def _find_weight_index_path(source_model: Path) -> Path:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        candidate = source_model / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported weight index file found under {source_model}")


def _iter_weight_shards(source_model: Path, weight_index: dict[str, str]) -> list[Path]:
    shard_names = sorted(set(weight_index.values()))
    return [source_model / shard_name for shard_name in shard_names]


def _wrapper_exists(output_dir: Path, wrapper_kind: str, source_model: Path) -> tuple[bool, dict[str, Any] | None]:
    metadata_path = output_dir / WRAPPER_METADATA_NAME
    if not metadata_path.exists():
        return False, None
    try:
        metadata = load_json(metadata_path)
    except Exception:
        return False, None
    is_match = (
        metadata.get("wrapper_kind") == wrapper_kind
        and metadata.get("source_model") == str(source_model.resolve())
    )
    return is_match, metadata


def _prepare_output_dir(
    output_dir: Path,
    *,
    source_model: Path,
    wrapper_kind: str,
    overwrite: bool,
    allow_reuse: bool,
) -> dict[str, Any] | None:
    if output_dir.exists() and any(output_dir.iterdir()):
        is_match, metadata = _wrapper_exists(output_dir, wrapper_kind, source_model)
        if is_match and allow_reuse and not overwrite:
            return metadata
        if not overwrite and not allow_reuse:
            raise FileExistsError(f"{output_dir} already exists and is not empty.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return None


def _symlink_common_source_files(source_model: Path, output_dir: Path) -> None:
    for name in _PASSTHROUGH_NAMES:
        src = source_model / name
        if src.exists():
            symlink_file(src, output_dir / name)


def _symlink_weight_files(source_model: Path, output_dir: Path, weight_map: dict[str, str]) -> None:
    for shard in _iter_weight_shards(source_model, weight_map):
        symlink_file(shard, output_dir / shard.name)


def _build_wrapper_metadata(*, wrapper_kind: str, source_model: Path, source_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "wrapper_kind": wrapper_kind,
        "source_model": str(source_model.resolve()),
        "source_model_type": source_config.get("model_type"),
        "source_checkpoint_name": source_model.name,
    }


def _with_normalized_config_torch_dtype(config: dict[str, Any]) -> dict[str, Any]:
    if "torch_dtype" not in config or config.get("torch_dtype") is None:
        return config
    normalized = normalize_config_torch_dtype_name(config.get("torch_dtype"))
    if config.get("torch_dtype") == normalized:
        return config
    updated = dict(config)
    updated["torch_dtype"] = normalized
    return updated


def _build_init_file(symbols: list[str], config_module: str, modeling_module: str) -> str:
    config_symbol = symbols[0]
    model_symbols = ", ".join(symbols[1:])
    exports = ",\n    ".join(f'"{item}"' for item in symbols)
    return (
        f"from .{config_module} import {config_symbol}\n"
        f"from .{modeling_module} import {model_symbols}\n\n"
        "__all__ = [\n"
        f"    {exports},\n"
        "]\n"
    )


def _basis_sharing_config_from(source_config: dict[str, Any]) -> dict[str, Any]:
    config = _with_normalized_config_torch_dtype(dict(source_config))
    config["model_type"] = "basis_sharing_llama"
    config["architectures"] = ["TransformersForCausalLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_basis_sharing_llama.BasisSharingLlamaConfig",
        "AutoModel": "modeling_basis_sharing_llama.BasisSharingLlamaModel",
        "AutoModelForCausalLM": "modeling_basis_sharing_llama.BasisSharingLlamaForCausalLM",
    }
    return config


def _load_safetensor_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _materialize_dense_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    output = torch.empty((left.shape[0], right.shape[1]), dtype=left.dtype, device="cpu")
    compute_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with torch.inference_mode():
        right_device = right.to(compute_device)
        for start in range(0, left.shape[0], LM_HEAD_CHUNK_ROWS):
            end = min(start + LM_HEAD_CHUNK_ROWS, left.shape[0])
            left_chunk = left[start:end].to(compute_device)
            product = torch.matmul(left_chunk, right_device).to("cpu")
            output[start:end].copy_(product)
        if compute_device.type == "cuda":
            torch.cuda.synchronize(compute_device)
    return output.contiguous()


def _write_filtered_safetensor_shard(source_path: Path, output_path: Path, *, drop_keys: set[str]) -> None:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(source_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key in drop_keys:
                continue
            tensors[key] = handle.get_tensor(key)
    save_file(tensors, str(output_path))


def _build_asvd_vllm_weight_index(
    source_model: Path,
) -> tuple[dict[str, str], int, torch.Tensor, set[str], set[str]]:
    index_path = _find_weight_index_path(source_model)
    if index_path.name != "model.safetensors.index.json":
        raise ValueError("ASVD vLLM wrapper currently expects safetensors checkpoints.")
    payload = load_json(index_path)
    weight_map = dict(payload.get("weight_map") or {})
    lm_head_a_name = "lm_head.ALinear.weight"
    lm_head_b_name = "lm_head.BLinear.weight"
    if lm_head_a_name not in weight_map or lm_head_b_name not in weight_map:
        raise ValueError("ASVD checkpoint does not expose low-rank lm_head weights.")

    a_tensor = _load_safetensor_tensor(source_model / weight_map[lm_head_a_name], lm_head_a_name)
    b_tensor = _load_safetensor_tensor(source_model / weight_map[lm_head_b_name], lm_head_b_name)
    dense_lm_head = _materialize_dense_product(a_tensor, b_tensor)

    updated_weight_map = {
        name: shard_name
        for name, shard_name in weight_map.items()
        if name not in {lm_head_a_name, lm_head_b_name}
    }
    updated_weight_map["lm_head.weight"] = LM_HEAD_DENSE_SHARD_NAME

    original_total_size = int((payload.get("metadata") or {}).get("total_size", 0))
    updated_total_size = original_total_size - a_tensor.numel() * a_tensor.element_size()
    updated_total_size -= b_tensor.numel() * b_tensor.element_size()
    updated_total_size += dense_lm_head.numel() * dense_lm_head.element_size()
    affected_shards = {weight_map[lm_head_a_name], weight_map[lm_head_b_name]}
    dropped_keys = {lm_head_a_name, lm_head_b_name}
    return updated_weight_map, updated_total_size, dense_lm_head, affected_shards, dropped_keys


def _asvd_vllm_config_from(source_config: dict[str, Any]) -> dict[str, Any]:
    config = _with_normalized_config_torch_dtype(dict(source_config))
    truncation_ranks = dict(config.get("truncation_ranks") or {})
    truncation_ranks.pop("lm_head", None)
    config["truncation_ranks"] = truncation_ranks
    config["model_type"] = "asvd_llama"
    config["architectures"] = ["TransformersForCausalLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_asvd_llama.ASVDLlamaConfig",
        "AutoModel": "modeling_asvd_llama.ASVDLlamaModel",
        "AutoModelForCausalLM": "modeling_asvd_llama.ASVDLlamaForCausalLM",
    }
    return config


def materialize_basis_sharing_llama_wrapper(
    source_model: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    allow_reuse: bool = True,
) -> WrapperMaterialization:
    source_model = Path(source_model).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    source_config = load_json(source_model / "config.json")
    wrapper_kind = "basis_sharing_llama_wrapper_v2"
    metadata = _prepare_output_dir(
        output_dir,
        source_model=source_model,
        wrapper_kind=wrapper_kind,
        overwrite=overwrite,
        allow_reuse=allow_reuse,
    )
    if metadata is not None:
        return WrapperMaterialization(output_dir=output_dir, created=False, reused=True, metadata=metadata)

    weight_index_path = _find_weight_index_path(source_model)
    weight_index_payload = load_json(weight_index_path)
    weight_map = dict(weight_index_payload.get("weight_map") or {})

    _symlink_common_source_files(source_model, output_dir)
    _symlink_weight_files(source_model, output_dir, weight_map)
    symlink_file(weight_index_path, output_dir / weight_index_path.name)

    _copy_repo_support_file(_MODELING_ROOT / "common.py", output_dir / "common.py")
    _copy_repo_support_file(
        _LLAMA_MODELING_ROOT / "configuration_basis_sharing_llama.py",
        output_dir / "configuration_basis_sharing_llama.py",
    )
    _copy_repo_support_file(
        _LLAMA_MODELING_ROOT / "modeling_basis_sharing_llama.py",
        output_dir / "modeling_basis_sharing_llama.py",
    )
    write_text(
        output_dir / "__init__.py",
        _build_init_file(
            [
                "BasisSharingLlamaConfig",
                "BasisSharingLlamaForCausalLM",
                "BasisSharingLlamaModel",
            ],
            "configuration_basis_sharing_llama",
            "modeling_basis_sharing_llama",
        ),
    )

    metadata = _build_wrapper_metadata(
        wrapper_kind=wrapper_kind,
        source_model=source_model,
        source_config=source_config,
    )
    write_json(output_dir / "config.json", _basis_sharing_config_from(source_config))
    write_json(output_dir / WRAPPER_METADATA_NAME, metadata)
    return WrapperMaterialization(output_dir=output_dir, created=True, reused=False, metadata=metadata)


def materialize_asvd_llama_vllm_wrapper(
    source_model: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    allow_reuse: bool = True,
) -> WrapperMaterialization:
    source_model = Path(source_model).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    source_config = load_json(source_model / "config.json")
    wrapper_kind = "asvd_llama_vllm_wrapper_v1"
    metadata = _prepare_output_dir(
        output_dir,
        source_model=source_model,
        wrapper_kind=wrapper_kind,
        overwrite=overwrite,
        allow_reuse=allow_reuse,
    )
    if metadata is not None:
        return WrapperMaterialization(output_dir=output_dir, created=False, reused=True, metadata=metadata)

    original_weight_map = dict(load_json(_find_weight_index_path(source_model)).get("weight_map") or {})
    updated_weight_map, total_size, dense_lm_head, affected_shards, dropped_keys = _build_asvd_vllm_weight_index(
        source_model
    )

    _symlink_common_source_files(source_model, output_dir)
    for shard in _iter_weight_shards(source_model, original_weight_map):
        if shard.name in affected_shards:
            continue
        symlink_file(shard, output_dir / shard.name)
    for shard_name in affected_shards:
        _write_filtered_safetensor_shard(
            source_model / shard_name,
            output_dir / shard_name,
            drop_keys=dropped_keys,
        )
    _copy_repo_support_file(_MODELING_ROOT / "common.py", output_dir / "common.py")
    _copy_repo_support_file(
        _LLAMA_MODELING_ROOT / "configuration_asvd_llama.py",
        output_dir / "configuration_asvd_llama.py",
    )
    _copy_repo_support_file(
        _LLAMA_MODELING_ROOT / "modeling_asvd_llama.py",
        output_dir / "modeling_asvd_llama.py",
    )
    write_text(
        output_dir / "__init__.py",
        _build_init_file(
            [
                "ASVDLlamaConfig",
                "ASVDLlamaForCausalLM",
                "ASVDLlamaModel",
            ],
            "configuration_asvd_llama",
            "modeling_asvd_llama",
        ),
    )

    save_file({"lm_head.weight": dense_lm_head}, str(output_dir / LM_HEAD_DENSE_SHARD_NAME))
    write_json(
        output_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total_size},
            "weight_map": updated_weight_map,
        },
    )

    metadata = _build_wrapper_metadata(
        wrapper_kind=wrapper_kind,
        source_model=source_model,
        source_config=source_config,
    )
    metadata["merged_dense_weights"] = ["lm_head.weight"]
    metadata["dropped_low_rank_weights"] = ["lm_head.ALinear.weight", "lm_head.BLinear.weight"]
    write_json(output_dir / "config.json", _asvd_vllm_config_from(source_config))
    write_json(output_dir / WRAPPER_METADATA_NAME, metadata)
    return WrapperMaterialization(output_dir=output_dir, created=True, reused=False, metadata=metadata)


def ensure_basis_sharing_llama_wrapper(source_model: str | Path, output_dir: str | Path) -> WrapperMaterialization:
    return materialize_basis_sharing_llama_wrapper(
        source_model,
        output_dir,
        overwrite=False,
        allow_reuse=True,
    )


def ensure_asvd_llama_vllm_wrapper(source_model: str | Path, output_dir: str | Path) -> WrapperMaterialization:
    return materialize_asvd_llama_vllm_wrapper(
        source_model,
        output_dir,
        overwrite=False,
        allow_reuse=True,
    )
