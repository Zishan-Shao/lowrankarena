from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import save_file

from src.validation import validate_checkpoint_layout


def test_validate_checkpoint_layout_accepts_uniform_dobi_precision(tmp_path: Path) -> None:
    model_dir = tmp_path / "dobi"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["DobiSVDLlamaForCausalLM"],
                "torch_dtype": "float16",
                "num_hidden_layers": 1,
                "dobi_target_modules": {
                    "model.layers.0.self_attn.q_proj": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.layers.0.self_attn.q_proj.ALinear.weight": __import__("torch").zeros((4, 2), dtype=__import__("torch").float16),
            "model.layers.0.self_attn.q_proj.BLinear.weight": __import__("torch").zeros((2, 4), dtype=__import__("torch").float16),
            "model.layers.0.self_attn.k_proj.weight": __import__("torch").zeros((4, 4), dtype=__import__("torch").float16),
            "model.layers.0.self_attn.v_proj.weight": __import__("torch").zeros((4, 4), dtype=__import__("torch").float16),
            "model.layers.0.self_attn.o_proj.weight": __import__("torch").zeros((4, 4), dtype=__import__("torch").float16),
            "model.layers.0.mlp.gate_proj.weight": __import__("torch").zeros((4, 4), dtype=__import__("torch").float16),
            "model.layers.0.mlp.up_proj.weight": __import__("torch").zeros((4, 4), dtype=__import__("torch").float16),
            "model.layers.0.mlp.down_proj.weight": __import__("torch").zeros((4, 4), dtype=__import__("torch").float16),
        },
        str(model_dir / "model-00001-of-00001.safetensors"),
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.ALinear.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.q_proj.BLinear.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    summary = validate_checkpoint_layout(model_dir, strict=True)

    assert summary["passed"] is True
    assert summary["layout_kind"] == "dobi_mixed_factorized"
    assert summary["precision"]["uniform_low_rank_precision"] is True


def test_validate_checkpoint_layout_rejects_mixed_low_rank_precision(tmp_path: Path) -> None:
    model_dir = tmp_path / "dobi_mixed"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["DobiSVDLlamaForCausalLM"],
                "torch_dtype": "float16",
                "num_hidden_layers": 1,
                "dobi_target_modules": {
                    "model.layers.0.self_attn.q_proj": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    import torch

    save_file(
        {
            "model.layers.0.self_attn.q_proj.ALinear.weight": torch.zeros((4, 2), dtype=torch.float16),
            "model.layers.0.self_attn.q_proj.BLinear.weight": torch.zeros((2, 4), dtype=torch.bfloat16),
            "model.layers.0.self_attn.k_proj.weight": torch.zeros((4, 4), dtype=torch.float16),
            "model.layers.0.self_attn.v_proj.weight": torch.zeros((4, 4), dtype=torch.float16),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros((4, 4), dtype=torch.float16),
            "model.layers.0.mlp.gate_proj.weight": torch.zeros((4, 4), dtype=torch.float16),
            "model.layers.0.mlp.up_proj.weight": torch.zeros((4, 4), dtype=torch.float16),
            "model.layers.0.mlp.down_proj.weight": torch.zeros((4, 4), dtype=torch.float16),
        },
        str(model_dir / "model-00001-of-00001.safetensors"),
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.ALinear.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.q_proj.BLinear.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_checkpoint_layout(model_dir, strict=True)
    except ValueError as exc:
        assert "mixed dtypes" in str(exc)
    else:  # pragma: no cover - test should fail first
        raise AssertionError("Expected validation to reject mixed low-rank precision.")


def test_validate_checkpoint_layout_accepts_asvd_factor_order(tmp_path: Path) -> None:
    model_dir = tmp_path / "asvd"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["ASVDLlamaForCausalLM"],
                "torch_dtype": "float16",
                "hidden_size": 4,
                "intermediate_size": 8,
                "vocab_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "truncation_ranks": {
                    "model.layers.0.self_attn.q_proj": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    import torch

    save_file(
        {
            "model.layers.0.self_attn.q_proj.ALinear.weight": torch.zeros((4, 2), dtype=torch.float16),
            "model.layers.0.self_attn.q_proj.BLinear.weight": torch.zeros((2, 4), dtype=torch.float16),
        },
        str(model_dir / "model-00001-of-00001.safetensors"),
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.ALinear.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.q_proj.BLinear.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    summary = validate_checkpoint_layout(model_dir, strict=True)

    assert summary["passed"] is True
    assert summary["layout_kind"] == "asvd_factorized"
    assert summary["observed_factor_layouts"]["model.layers.0.self_attn.q_proj"] == "out_rank__rank_in"


def test_validate_checkpoint_layout_checks_lowrank_gqa_kv_shape(tmp_path: Path) -> None:
    model_dir = tmp_path / "lowrank_gqa"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "lowrank_llama",
                "architectures": ["LowRankLlamaForCausalLM"],
                "torch_dtype": "float16",
                "hidden_size": 16,
                "intermediate_size": 32,
                "vocab_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "low_rank_method": "basis-sharing",
                "low_rank_modules": {
                    "model.layers.0.self_attn.k_proj": {"rank": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    import torch

    save_file(
        {
            "model.layers.0.self_attn.k_proj.ALinear.weight": torch.zeros((8, 3), dtype=torch.float16),
            "model.layers.0.self_attn.k_proj.BLinear.weight": torch.zeros((3, 16), dtype=torch.float16),
        },
        str(model_dir / "model-00001-of-00001.safetensors"),
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model.layers.0.self_attn.k_proj.ALinear.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.k_proj.BLinear.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    summary = validate_checkpoint_layout(model_dir, strict=True)

    assert summary["passed"] is True
    assert summary["layout_kind"] == "basis_sharing_factorized"
    assert summary["observed_dense_shapes"]["model.layers.0.self_attn.k_proj"] == [8, 16]


def test_validate_checkpoint_layout_rejects_lowrank_gqa_kv_full_head_shape(tmp_path: Path) -> None:
    model_dir = tmp_path / "lowrank_gqa_bad"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "lowrank_llama",
                "architectures": ["LowRankLlamaForCausalLM"],
                "torch_dtype": "float16",
                "hidden_size": 16,
                "intermediate_size": 32,
                "vocab_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "low_rank_method": "basis-sharing",
                "low_rank_modules": {
                    "model.layers.0.self_attn.k_proj": {"rank": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    import torch

    save_file(
        {
            "model.layers.0.self_attn.k_proj.ALinear.weight": torch.zeros((16, 3), dtype=torch.float16),
            "model.layers.0.self_attn.k_proj.BLinear.weight": torch.zeros((3, 16), dtype=torch.float16),
        },
        str(model_dir / "model-00001-of-00001.safetensors"),
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model.layers.0.self_attn.k_proj.ALinear.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.k_proj.BLinear.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_checkpoint_layout(model_dir, strict=True)
    except ValueError as exc:
        assert "do not reconstruct the expected dense shape (8, 16)" in str(exc)
    else:  # pragma: no cover - test should fail first
        raise AssertionError("Expected validation to reject full-head K/V factor shapes for GQA.")
