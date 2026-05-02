from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.materialize_pruning_hf import (
    VLLM_WRAPPER_METADATA_NAME,
    materialize_blockpruner,
    materialize_llmpruner,
    materialize_slicegpt,
)


def test_materialize_slicegpt_preserves_root_state_dict(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "hidden_act": "silu",
                "hidden_size": 4,
                "intermediate_size": 8,
                "model_type": "llama",
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 16,
            }
        ),
        encoding="utf-8",
    )
    (source / "Llama-3.1-8B_0.6.json").write_text(
        json.dumps(
            {
                "hidden_size": 4,
                "embedding_dimensions": {"0": 3},
                "attention_input_dimensions": {"0": 3},
                "attention_output_dimensions": {"0": 3},
                "mlp_input_dimensions": {"0": 3},
                "mlp_output_dimensions": {"0": 4},
            }
        ),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    weight = source / "Llama-3.1-8B_0.6.pt"
    torch.save({"model.layers.0.self_attn.q_proj.weight": torch.ones(2, 3)}, weight)

    metadata = materialize_slicegpt(source, output, copy=False)

    assert metadata["materialization_kind"] == "slicegpt_safetensors_state_dict"
    assert metadata["standard_weight_file"] == "model.safetensors"
    assert not (output / "pytorch_model.bin").exists()
    saved = load_file(output / "model.safetensors")
    assert saved["model.layers.0.self_attn.q_proj.weight"].shape == (2, 3)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    wrapper_meta = json.loads((output / VLLM_WRAPPER_METADATA_NAME).read_text(encoding="utf-8"))
    assert config["architectures"] == ["TransformersForCausalLM"]
    assert config["auto_map"]["AutoModel"].endswith("SliceGPTLlamaModel")
    assert config["embedding_size"] == 3
    assert not (output / "config.json").is_symlink()
    assert (output / "modeling_slicegpt_llama.py").exists()
    assert wrapper_meta["preserves_pruned_form"] is True
    materialization = json.loads((output / "pruning_materialization.json").read_text(encoding="utf-8"))
    assert materialization["preserves_pruned_form"] is True
    assert materialization["merged"] is False


def test_materialize_llmpruner_writes_pickled_runtime_wrapper(tmp_path: Path) -> None:
    source = tmp_path / "llmpruner"
    output = tmp_path / "out"
    (source / "adapter").mkdir(parents=True)
    (source / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    weight = source / "pytorch_model.bin"
    weight.write_bytes(b"pickled-model")

    metadata = materialize_llmpruner(
        source,
        output,
        base_model="meta-llama/Llama-3.1-8B",
        tokenizer_source=None,
        copy=False,
        skip_tokenizer=True,
        local_files_only=True,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    wrapper_meta = json.loads((output / VLLM_WRAPPER_METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["materialization_kind"] == "llmpruner_pickled_pruned_model"
    assert (output / "llm_pruner_model.bin").resolve() == weight
    assert (output / "adapter").resolve() == source / "adapter"
    assert config["auto_map"]["AutoModelForCausalLM"].endswith("LLMPrunerPickleForCausalLM")
    assert wrapper_meta["preserves_pruned_form"] is True
    assert wrapper_meta["merged"] is False


def test_materialize_blockpruner_writes_runtime_mask_wrapper(tmp_path: Path) -> None:
    source = tmp_path / "prune_only_0.6"
    output = tmp_path / "out"
    source.mkdir()
    (source / "del_order_list.json").write_text(json.dumps({"7": [["mha", 0], ["mlp", 1]]}), encoding="utf-8")
    (source / "keep_ratio_calibration.json").write_text(
        json.dumps(
            {
                "target_mapping": [
                    {
                        "target_keep_ratio": 0.6,
                        "selected_del_block_num": 7,
                        "achieved_keep_ratio": 0.59,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    metadata = materialize_blockpruner(
        source,
        output,
        base_model="meta-llama/Llama-3.1-8B",
        tokenizer_source=None,
        del_block_num=None,
        copy=False,
        skip_tokenizer=True,
        local_files_only=True,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    wrapper_meta = json.loads((output / VLLM_WRAPPER_METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["materialization_kind"] == "blockpruner_runtime_mask"
    assert config["del_block_num"] == 7
    assert config["auto_map"]["AutoModelForCausalLM"].endswith("BlockPrunerLlamaForCausalLM")
    assert (output / "del_order_list.json").resolve() == source / "del_order_list.json"
    assert wrapper_meta["preserves_pruned_form"] is True
    assert wrapper_meta["merged"] is False
