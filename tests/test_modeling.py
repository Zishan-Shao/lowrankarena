from __future__ import annotations

import torch

from src.modeling.common import LowRankLinear
from src.modeling.llama import (
    ASVDLlamaConfig,
    ASVDLlamaForCausalLM,
    BasisSharingLlamaConfig,
    BasisSharingLlamaForCausalLM,
)
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


def test_asvd_and_basis_sharing_llama_support_gqa_shapes() -> None:
    asvd_config = ASVDLlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        truncation_ranks={
            "model.layers.0.self_attn.q_proj": {"rank": 4},
            "model.layers.0.self_attn.k_proj": {"rank": 3},
            "model.layers.0.self_attn.v_proj": {"rank": 3},
        },
    )
    asvd_model = ASVDLlamaForCausalLM(asvd_config)
    asvd_attn = asvd_model.model.layers[0].self_attn

    assert isinstance(asvd_attn.k_proj, LowRankLinear)
    assert asvd_attn.k_proj.ALinear.out_features == 8
    assert asvd_attn.v_proj.ALinear.out_features == 8
    assert asvd_model(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long)).logits.shape == (1, 3, 32)

    basis_config = BasisSharingLlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_basis_q=4,
        num_basis_k=3,
        num_basis_v=3,
        num_basis_o=4,
        num_basis_gate=6,
        num_basis_up=6,
        num_basis_down=6,
        q_groups=[[0]],
        k_groups=[[0]],
        v_groups=[[0]],
        o_groups=[[0]],
        gate_groups=[[0]],
        up_groups=[[0]],
        down_groups=[[0]],
    )
    basis_model = BasisSharingLlamaForCausalLM(basis_config)
    basis_attn = basis_model.model.layers[0].self_attn

    assert basis_attn.k_proj.out_features == 8
    assert basis_attn.v_proj.out_features == 8
    assert basis_model(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long)).logits.shape == (1, 3, 32)
