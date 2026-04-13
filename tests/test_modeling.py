from __future__ import annotations

import torch

from src.modeling.common import LowRankLinear
from src.modeling.mistral import LowRankMistralConfig, LowRankMistralForCausalLM


def test_lowrank_mistral_replaces_target_modules_and_runs_forward() -> None:
    config = LowRankMistralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        low_rank_modules={
            "model.layers.0.self_attn.q_proj": {"rank": 4},
            "model.layers.0.self_attn.o_proj": {"rank": 4},
            "model.layers.0.mlp.gate_proj": {"rank": 6},
        },
    )

    model = LowRankMistralForCausalLM(config)
    layer = model.model.layers[0]

    assert isinstance(layer.self_attn.q_proj, LowRankLinear)
    assert isinstance(layer.self_attn.o_proj, LowRankLinear)
    assert isinstance(layer.mlp.gate_proj, LowRankLinear)
    assert set(model.model.replaced_low_rank_modules) == set(config.low_rank_modules)
    assert model.model.missing_low_rank_modules == []

    outputs = model(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert outputs.logits.shape == (1, 3, 32)
