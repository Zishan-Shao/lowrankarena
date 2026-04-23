from __future__ import annotations

import os
from typing import Any, Sequence


def bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024**3)


def _torch_module() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _resolve_cuda_device_index(torch: Any, device: Any | None) -> int | None:
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return device

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    if resolved.index is None:
        return int(torch.cuda.current_device())
    return int(resolved.index)


def describe_cuda_device(device: Any | None = None) -> dict[str, Any]:
    requested_device = str(device) if device is not None else None
    torch = _torch_module()
    if torch is None:
        return {
            "available": False,
            "device_count": 0,
            "requested_device": requested_device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "error": "torch is not installed",
        }

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        return {
            "available": False,
            "device_count": int(torch.cuda.device_count()),
            "requested_device": requested_device,
            "cuda_visible_devices": cuda_visible_devices,
        }

    device_count = int(torch.cuda.device_count())
    device_index = _resolve_cuda_device_index(torch, device)
    if device_index is None:
        return {
            "available": True,
            "device_count": device_count,
            "requested_device": requested_device,
            "cuda_visible_devices": cuda_visible_devices,
            "device_type": str(torch.device(device).type),
        }
    if device_index < 0 or device_index >= device_count:
        return {
            "available": True,
            "device_count": device_count,
            "requested_device": requested_device,
            "cuda_visible_devices": cuda_visible_devices,
            "device_index": device_index,
            "error": "device index is outside the visible CUDA range",
        }

    props = torch.cuda.get_device_properties(device_index)
    major = int(getattr(props, "major", 0))
    minor = int(getattr(props, "minor", 0))
    total_memory_bytes = int(getattr(props, "total_memory", 0))
    return {
        "available": True,
        "device": f"cuda:{device_index}",
        "device_index": device_index,
        "device_count": device_count,
        "requested_device": requested_device,
        "cuda_visible_devices": cuda_visible_devices,
        "name": str(getattr(props, "name", "")),
        "compute_capability": f"{major}.{minor}",
        "compute_capability_major": major,
        "compute_capability_minor": minor,
        "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
        "total_memory_bytes": total_memory_bytes,
        "total_memory_gib": bytes_to_gib(total_memory_bytes),
    }


def describe_cuda_runtime(
    *,
    devices: Sequence[Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    torch = _torch_module()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch is None:
        return {
            "available": False,
            "device_count": 0,
            "cuda_visible_devices": cuda_visible_devices,
            "devices": [],
            "error": "torch is not installed",
        }

    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    if not available:
        return {
            "available": False,
            "device_count": device_count,
            "cuda_visible_devices": cuda_visible_devices,
            "devices": [],
        }

    if devices is not None:
        selected_devices = list(devices)
    else:
        selected_count = device_count if limit is None else min(int(limit), device_count)
        selected_devices = [f"cuda:{idx}" for idx in range(max(selected_count, 0))]

    return {
        "available": True,
        "device_count": device_count,
        "cuda_visible_devices": cuda_visible_devices,
        "devices": [describe_cuda_device(device) for device in selected_devices],
    }
