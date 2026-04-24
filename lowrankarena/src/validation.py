from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch

from src.dtype_utils import normalize_config_torch_dtype_name, normalize_dtype_name
from src.utils import dump_json, load_json, project_path

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - optional until dependencies are installed.
    safe_open = None


DEFAULT_VALIDATION_ROOT = project_path("results", "validation")
VALIDATION_SCHEMA_VERSION = "1.2"

_LLAMA_LINEAR_SUFFIXES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def _find_weight_index_path(model_path: Path) -> Path | None:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        candidate = model_path / name
        if candidate.exists():
            return candidate
    return None


def _weight_map_for(model_path: Path) -> dict[str, str]:
    index_path = _find_weight_index_path(model_path)
    if index_path is None:
        return {}
    return dict(load_json(index_path).get("weight_map") or {})


def _load_tensor_from_weight_map(
    model_path: Path,
    weight_map: dict[str, str],
    tensor_name: str,
    shard_cache: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    shard_name = weight_map[tensor_name]
    shard_path = model_path / shard_name
    if shard_path.suffix == ".safetensors":
        if safe_open is None:
            raise RuntimeError("safetensors is required to inspect safetensors checkpoints.")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(tensor_name)

    if shard_name not in shard_cache:
        shard_cache[shard_name] = torch.load(shard_path, map_location="cpu")
    return shard_cache[shard_name][tensor_name]


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _normalize_dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return normalize_dtype_name(value)
    except ValueError:
        return str(value).replace("torch.", "").strip().lower()


def _normalize_config_dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return normalize_config_torch_dtype_name(value)
    except ValueError:
        return str(value).replace("torch.", "").strip().lower()


def _canonical_llama_modules(num_layers: int) -> list[str]:
    modules: list[str] = []
    for layer_idx in range(int(num_layers)):
        for suffix in _LLAMA_LINEAR_SUFFIXES:
            modules.append(f"model.layers.{layer_idx}.{suffix}")
    return modules


def _expected_llama_dense_shape(config: dict[str, Any], module_name: str) -> tuple[int, int] | None:
    hidden_size = config.get("hidden_size")
    intermediate_size = config.get("intermediate_size")
    vocab_size = config.get("vocab_size")
    num_attention_heads = config.get("num_attention_heads")
    num_kv_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = config.get("head_dim")
    if hidden_size is not None and head_dim is None and num_attention_heads:
        head_dim = int(hidden_size) // int(num_attention_heads)

    if module_name == "lm_head" and vocab_size is not None and hidden_size is not None:
        return (int(vocab_size), int(hidden_size))

    match = re.match(r"^model\.layers\.\d+\.(.+)$", module_name)
    if not match:
        return None

    suffix = match.group(1)
    if suffix in {"self_attn.q_proj", "self_attn.o_proj"} and hidden_size is not None:
        return (int(hidden_size), int(hidden_size))
    if suffix in {"self_attn.k_proj", "self_attn.v_proj"} and hidden_size is not None and num_kv_heads is not None and head_dim is not None:
        return (int(num_kv_heads) * int(head_dim), int(hidden_size))
    if suffix in {"mlp.gate_proj", "mlp.up_proj"} and hidden_size is not None and intermediate_size is not None:
        return (int(intermediate_size), int(hidden_size))
    if suffix == "mlp.down_proj" and hidden_size is not None and intermediate_size is not None:
        return (int(hidden_size), int(intermediate_size))
    return None


def _is_asvd_config(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    auto_map = dict(config.get("auto_map") or {})
    return config.get("model_type") == "asvd_llama" or "ASVDLlamaForCausalLM" in architectures or auto_map.get(
        "AutoModelForCausalLM", ""
    ).endswith("ASVDLlamaForCausalLM")


def _is_dobi_config(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    auto_map = dict(config.get("auto_map") or {})
    return "DobiSVDLlamaForCausalLM" in architectures or auto_map.get("AutoModelForCausalLM", "").endswith(
        "DobiSVDLlamaForCausalLM"
    )


def _is_basis_sharing_config(config: dict[str, Any]) -> bool:
    architectures = set(config.get("architectures") or [])
    auto_map = dict(config.get("auto_map") or {})
    return config.get("model_type") == "basis_sharing_llama" or "ShareLlamaForCausalLM" in architectures or auto_map.get(
        "AutoModelForCausalLM", ""
    ).endswith("BasisSharingLlamaForCausalLM")


def _is_low_rank_config(config: dict[str, Any]) -> bool:
    low_rank_modules = config.get("low_rank_modules")
    return isinstance(low_rank_modules, dict) and bool(low_rank_modules)


def _extract_expected_rank(spec: Any) -> int:
    if isinstance(spec, dict):
        return int(spec["rank"])
    return int(spec)


def _factorized_modules_from_weight_map(weight_map: dict[str, str], left_suffix: str, right_suffix: str) -> set[str]:
    modules: set[str] = set()
    for name in weight_map:
        if name.endswith(left_suffix):
            base = name[: -len(left_suffix)]
            right_name = f"{base}{right_suffix}"
            if right_name in weight_map:
                modules.add(base.rstrip("."))
    return modules


def _factor_layout_candidates(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    expected_rank: int,
) -> list[dict[str, Any]]:
    if len(left_shape) != 2 or len(right_shape) != 2:
        return []

    candidates: list[dict[str, Any]] = []
    if left_shape[1] == expected_rank and right_shape[0] == expected_rank:
        candidates.append(
            {
                "style": "out_rank__rank_in",
                "dense_shape": (int(left_shape[0]), int(right_shape[1])),
                "rank_axes": {"left": 1, "right": 0},
            }
        )
    if left_shape[0] == expected_rank and right_shape[1] == expected_rank:
        candidates.append(
            {
                "style": "rank_in__out_rank",
                "dense_shape": (int(right_shape[0]), int(left_shape[1])),
                "rank_axes": {"left": 0, "right": 1},
            }
        )
    return candidates


def _validate_factorized_layout(
    *,
    model_path: Path,
    config: dict[str, Any],
    expected_specs: dict[str, Any],
    weight_map: dict[str, str],
    left_suffix: str,
    right_suffix: str,
    layout_kind: str,
    canonical_modules: list[str] | None = None,
) -> dict[str, Any]:
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}
    issues: list[str] = []
    expected_modules = sorted(expected_specs)
    found_modules = sorted(_factorized_modules_from_weight_map(weight_map, left_suffix, right_suffix))
    found_set = set(found_modules)
    expected_set = set(expected_modules)
    missing_modules = sorted(expected_set - found_set)
    unexpected_low_rank_modules = sorted(found_set - expected_set)
    low_rank_factor_dtypes: set[str] = set()
    observed_ranks: dict[str, int] = {}
    observed_shapes: dict[str, dict[str, list[int]]] = {}
    observed_factor_layouts: dict[str, str] = {}
    expected_dense_shapes: dict[str, list[int]] = {}
    observed_dense_shapes: dict[str, list[int]] = {}
    reference_torch_dtype = _normalize_config_dtype_name(config.get("torch_dtype"))

    for module_name in expected_modules:
        left_name = f"{module_name}{left_suffix}"
        right_name = f"{module_name}{right_suffix}"
        if left_name not in weight_map or right_name not in weight_map:
            continue
        left_tensor = _load_tensor_from_weight_map(model_path, weight_map, left_name, shard_cache)
        right_tensor = _load_tensor_from_weight_map(model_path, weight_map, right_name, shard_cache)
        low_rank_factor_dtypes.add(_normalize_dtype_name(_dtype_name(left_tensor)) or _dtype_name(left_tensor))
        low_rank_factor_dtypes.add(_normalize_dtype_name(_dtype_name(right_tensor)) or _dtype_name(right_tensor))
        expected_rank = _extract_expected_rank(expected_specs[module_name])
        left_shape = tuple(int(item) for item in left_tensor.shape)
        right_shape = tuple(int(item) for item in right_tensor.shape)
        observed_shapes[module_name] = {
            "left": list(left_shape),
            "right": list(right_shape),
        }

        candidates = _factor_layout_candidates(left_shape, right_shape, expected_rank)
        if not candidates:
            issues.append(
                f"{module_name}: expected rank {expected_rank} but found incompatible factor shapes "
                f"{left_name}={left_shape} and {right_name}={right_shape}"
            )
            continue

        expected_dense_shape = _expected_llama_dense_shape(config, module_name)
        if expected_dense_shape is not None:
            expected_dense_shapes[module_name] = list(expected_dense_shape)

        selected = None
        if expected_dense_shape is not None:
            for candidate in candidates:
                if tuple(candidate["dense_shape"]) == expected_dense_shape:
                    selected = candidate
                    break
            if selected is None:
                issues.append(
                    f"{module_name}: factor shapes {left_shape} and {right_shape} do not reconstruct the expected dense "
                    f"shape {expected_dense_shape} for this runtime module"
                )
                continue
        else:
            selected = candidates[0]

        observed_ranks[module_name] = expected_rank
        observed_factor_layouts[module_name] = str(selected["style"])
        observed_dense_shapes[module_name] = [int(dim) for dim in selected["dense_shape"]]

    if missing_modules:
        issues.append(f"Missing factorized modules: {', '.join(missing_modules)}")
    if unexpected_low_rank_modules:
        issues.append(f"Unexpected factorized modules: {', '.join(unexpected_low_rank_modules)}")

    dense_modules: list[str] = []
    missing_dense_or_factorized: list[str] = []
    if canonical_modules is not None:
        for module_name in canonical_modules:
            dense_name = f"{module_name}.weight"
            is_factorized = module_name in found_set
            is_dense = dense_name in weight_map
            if not is_factorized and is_dense:
                dense_modules.append(module_name)
            if not is_factorized and not is_dense:
                missing_dense_or_factorized.append(module_name)
        if missing_dense_or_factorized:
            issues.append(
                "Canonical linear modules missing both dense and factorized weights: "
                + ", ".join(missing_dense_or_factorized)
            )

    uniform_low_rank_precision = len(low_rank_factor_dtypes) <= 1
    if not uniform_low_rank_precision:
        issues.append(
            "Low-rank factor tensors use mixed dtypes: " + ", ".join(sorted(low_rank_factor_dtypes))
        )
    matches_reference_torch_dtype = (
        reference_torch_dtype is None
        or not low_rank_factor_dtypes
        or low_rank_factor_dtypes == {reference_torch_dtype}
    )
    if reference_torch_dtype is not None and low_rank_factor_dtypes and not matches_reference_torch_dtype:
        issues.append(
            "Low-rank factor tensors do not match config torch_dtype "
            f"{reference_torch_dtype}: " + ", ".join(sorted(low_rank_factor_dtypes))
        )

    return {
        "validation_version": VALIDATION_SCHEMA_VERSION,
        "layout_kind": layout_kind,
        "model_type": config.get("model_type"),
        "expected_low_rank_module_count": len(expected_modules),
        "validated_low_rank_module_count": len(found_modules),
        "dense_module_count": len(dense_modules),
        "expected_low_rank_modules": expected_modules,
        "validated_low_rank_modules": found_modules,
        "dense_modules": dense_modules,
        "missing_modules": missing_modules,
        "unexpected_low_rank_modules": unexpected_low_rank_modules,
        "missing_dense_or_factorized_modules": missing_dense_or_factorized,
        "observed_ranks": observed_ranks,
        "observed_shapes": observed_shapes,
        "observed_dense_shapes": observed_dense_shapes,
        "expected_dense_shapes": expected_dense_shapes,
        "observed_factor_layouts": observed_factor_layouts,
        "precision": {
            "reference_torch_dtype": reference_torch_dtype,
            "low_rank_factor_dtypes": sorted(low_rank_factor_dtypes),
            "uniform_low_rank_precision": uniform_low_rank_precision,
            "matches_reference_torch_dtype": matches_reference_torch_dtype,
        },
        "passed": not issues,
        "issues": issues,
    }


def _validate_basis_sharing_layout(model_path: Path, config: dict[str, Any], weight_map: dict[str, str]) -> dict[str, Any]:
    num_layers = int(config.get("num_hidden_layers", 0))
    issues: list[str] = []
    coverage: dict[str, dict[str, Any]] = {}
    for basis_name in ("q", "k", "v", "o", "gate", "up", "down"):
        groups = list(config.get(f"{basis_name}_groups") or [])
        seen: dict[int, int] = {}
        for group in groups:
            for layer_idx in group:
                seen[int(layer_idx)] = seen.get(int(layer_idx), 0) + 1
        missing = [layer_idx for layer_idx in range(num_layers) if seen.get(layer_idx, 0) == 0]
        duplicates = [layer_idx for layer_idx, count in seen.items() if count > 1]
        coverage[basis_name] = {
            "group_count": len(groups),
            "missing_layers": missing,
            "duplicate_layers": duplicates,
        }
        if missing:
            issues.append(f"{basis_name}_groups missing layers: {missing}")
        if duplicates:
            issues.append(f"{basis_name}_groups duplicate layers: {duplicates}")

    coefficient_examples = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]
    for key in coefficient_examples:
        if weight_map and key not in weight_map:
            issues.append(f"Missing basis-sharing coefficient tensor: {key}")

    return {
        "validation_version": VALIDATION_SCHEMA_VERSION,
        "layout_kind": "shared_basis",
        "model_type": config.get("model_type"),
        "coverage": coverage,
        "precision": {
            "reference_torch_dtype": _normalize_config_dtype_name(config.get("torch_dtype")),
            "uniform_low_rank_precision": True,
            "low_rank_factor_dtypes": [_normalize_config_dtype_name(config.get("torch_dtype"))]
            if config.get("torch_dtype")
            else [],
            "matches_reference_torch_dtype": True,
        },
        "passed": not issues,
        "issues": issues,
    }


def _validate_dense_layout(config: dict[str, Any], weight_map: dict[str, str]) -> dict[str, Any]:
    return {
        "validation_version": VALIDATION_SCHEMA_VERSION,
        "layout_kind": "dense",
        "model_type": config.get("model_type"),
        "parameter_tensor_count": len(weight_map),
        "precision": {
            "reference_torch_dtype": _normalize_config_dtype_name(config.get("torch_dtype")),
            "uniform_low_rank_precision": True,
            "low_rank_factor_dtypes": [],
            "matches_reference_torch_dtype": True,
        },
        "passed": True,
        "issues": [],
    }


def _summary_for_model_path(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = load_json(config_path)
    weight_map = _weight_map_for(model_path)

    if config.get("model_type") == "svdllm-llama":
        summary = _validate_factorized_layout(
            model_path=model_path,
            config=config,
            expected_specs=dict(config.get("svd_ranks") or {}),
            weight_map=weight_map,
            left_suffix=".u_proj.weight",
            right_suffix=".v_proj.weight",
            layout_kind="svdllm_factorized",
        )
    elif _is_asvd_config(config):
        summary = _validate_factorized_layout(
            model_path=model_path,
            config=config,
            expected_specs=dict(config.get("truncation_ranks") or {}),
            weight_map=weight_map,
            left_suffix=".ALinear.weight",
            right_suffix=".BLinear.weight",
            layout_kind="asvd_factorized",
        )
    elif _is_dobi_config(config):
        summary = _validate_factorized_layout(
            model_path=model_path,
            config=config,
            expected_specs=dict(config.get("dobi_target_modules") or {}),
            weight_map=weight_map,
            left_suffix=".ALinear.weight",
            right_suffix=".BLinear.weight",
            layout_kind="dobi_mixed_factorized",
            canonical_modules=_canonical_llama_modules(int(config.get("num_hidden_layers", 0))),
        )
    elif _is_basis_sharing_config(config):
        summary = _validate_basis_sharing_layout(model_path, config, weight_map)
    elif _is_low_rank_config(config):
        method = str(config.get("low_rank_method") or "lowrank").replace("-", "_")
        summary = _validate_factorized_layout(
            model_path=model_path,
            config=config,
            expected_specs=dict(config.get("low_rank_modules") or {}),
            weight_map=weight_map,
            left_suffix=".ALinear.weight",
            right_suffix=".BLinear.weight",
            layout_kind=f"{method}_factorized",
        )
    else:
        summary = _validate_dense_layout(config, weight_map)

    summary["model_path"] = str(model_path)
    return summary


def _cache_path_for(model_path: Path, cache_root: Path) -> Path:
    digest = hashlib.sha1(str(model_path).encode("utf-8")).hexdigest()[:12]
    return cache_root / f"{model_path.name}__{digest}__v{VALIDATION_SCHEMA_VERSION}.json"


def validate_checkpoint_layout(
    model_path: str | Path,
    *,
    cache_root: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    resolved_model_path = Path(model_path).expanduser().resolve()
    validation_root = Path(cache_root or DEFAULT_VALIDATION_ROOT).expanduser().resolve()
    cache_path = _cache_path_for(resolved_model_path, validation_root)

    if cache_path.exists():
        payload = load_json(cache_path)
        if (
            payload.get("model_path") == str(resolved_model_path)
            and payload.get("validation_version") == VALIDATION_SCHEMA_VERSION
        ):
            if strict and not payload.get("passed", False):
                raise ValueError("; ".join(payload.get("issues", [])) or "Checkpoint validation failed.")
            return payload

    payload = _summary_for_model_path(resolved_model_path)
    dump_json(payload, cache_path)
    if strict and not payload.get("passed", False):
        raise ValueError("; ".join(payload.get("issues", [])) or "Checkpoint validation failed.")
    return payload
