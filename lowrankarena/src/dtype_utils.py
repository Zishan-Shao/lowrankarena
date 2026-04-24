from __future__ import annotations

from typing import Any


_DTYPE_ALIASES = {
    "auto": "auto",
    "float": "float16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat": "bfloat16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
}

_CONFIG_TORCH_DTYPE_ALIASES = {
    "auto": "auto",
    "bfloat": "bfloat16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "double": "float64",
    "float": "float32",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
    "half": "float16",
}


def normalize_dtype_name(value: Any, *, default: str = "auto") -> str:
    if value is None:
        return default
    key = str(value).replace("torch.", "").strip().lower()
    if not key:
        return default
    if key not in _DTYPE_ALIASES:
        supported = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(f"Unsupported dtype {value!r}. Expected one of: {supported}")
    return _DTYPE_ALIASES[key]


def normalize_config_torch_dtype_name(value: Any, *, default: str = "auto") -> str:
    if value is None:
        return default
    key = str(value).strip().lower()
    if key.startswith("torch."):
        key = key.removeprefix("torch.")
    if not key:
        return default
    if key not in _CONFIG_TORCH_DTYPE_ALIASES:
        supported = ", ".join(sorted(_CONFIG_TORCH_DTYPE_ALIASES))
        raise ValueError(f"Unsupported config torch_dtype {value!r}. Expected one of: {supported}")
    return _CONFIG_TORCH_DTYPE_ALIASES[key]
