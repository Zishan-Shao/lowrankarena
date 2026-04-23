from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.dtype_utils import normalize_config_torch_dtype_name
from src.hardware import describe_cuda_runtime
from src.memory_runner import estimate_dense_kv_bytes, pick_filler_token_id, resolve_dtype


def test_resolve_dtype_maps_expected_aliases() -> None:
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("fp16") is torch.float16
    assert resolve_dtype("float") is torch.float16
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("bfloat") is torch.bfloat16
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("fp32") is torch.float32
    assert resolve_dtype("auto") == "auto"


def test_normalize_config_torch_dtype_strips_torch_prefix_without_retyping_float() -> None:
    assert normalize_config_torch_dtype_name("torch.float16") == "float16"
    assert normalize_config_torch_dtype_name("torch.bfloat16") == "bfloat16"
    assert normalize_config_torch_dtype_name("torch.float") == "float32"
    assert normalize_config_torch_dtype_name("float") == "float32"


def test_pick_filler_token_id_prefers_configured_special_tokens() -> None:
    config = SimpleNamespace(vocab_size=32000, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    assert pick_filler_token_id(config) == 1


def test_estimate_dense_kv_bytes_matches_llama_7b_fp16_shape() -> None:
    config = SimpleNamespace(
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_size=4096,
        head_dim=128,
    )
    estimate = estimate_dense_kv_bytes(
        config,
        batch_size=1,
        prompt_length=32,
        generation_length=8,
        bytes_per_elem=2,
    )

    assert estimate["bytes_per_token"] == 524288
    assert estimate["cached_tokens_at_peak"] == 39
    assert estimate["estimated_peak_kv_bytes"] == 524288 * 39


def test_describe_cuda_runtime_records_gpu_model_metadata() -> None:
    props = SimpleNamespace(
        name="Fake A100",
        major=8,
        minor=0,
        multi_processor_count=108,
        total_memory=80 * 1024**3,
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.get_device_properties", return_value=props),
    ):
        runtime = describe_cuda_runtime(limit=1)

    assert runtime["available"] is True
    assert runtime["device_count"] == 1
    assert runtime["devices"][0]["name"] == "Fake A100"
    assert runtime["devices"][0]["compute_capability"] == "8.0"
    assert runtime["devices"][0]["total_memory_gib"] == 80.0
