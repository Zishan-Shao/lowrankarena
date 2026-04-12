from __future__ import annotations

from types import SimpleNamespace

import torch

from src.memory_runner import estimate_dense_kv_bytes, pick_filler_token_id, resolve_dtype


def test_resolve_dtype_maps_expected_aliases() -> None:
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("auto") == "auto"


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
