from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.load import LoadedCheckpoint
from src.registry import CheckpointRecord
from src.vllm.prepare_svdllm_vllm_model import WRAPPER_METADATA_NAME
from src.vllm.vllm_adapter import (
    PRUNING_WEIGHT_WRAPPER_METADATA_NAME,
    TRANSFORMERS_BACKEND_WRAPPER_METADATA_NAME,
    prepare_model_for_vllm,
)


def sample_record() -> CheckpointRecord:
    return CheckpointRecord(
        name="demo",
        model_family="llama",
        variant="base",
        method="svdllm_v1_update",
        source="local",
        repo_id="",
        revision="main",
        subpath="",
        benchmarks=["speed"],
        enabled=True,
        notes="test row",
    )


def write_ready_wrapper(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "svdllm-llama",
        "architectures": ["TransformersForCausalLM"],
        "auto_map": {
            "AutoModel": "modeling_svdllm_llama.SVDLLMLlamaModel",
            "AutoModelForCausalLM": "modeling_svdllm_llama.SVDLLMLlamaForCausalLM",
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "modeling_svdllm_llama.py").write_text("# test\n", encoding="utf-8")
    (path / "configuration_svdllm_llama.py").write_text("# test\n", encoding="utf-8")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}), encoding="utf-8")
    (path / WRAPPER_METADATA_NAME).write_text(json.dumps({"ok": True}), encoding="utf-8")
    return path


def test_prepare_model_for_vllm_recognizes_ready_wrapper(tmp_path: Path) -> None:
    wrapper_dir = write_ready_wrapper(tmp_path / "wrapper")
    loaded = LoadedCheckpoint(
        record=sample_record(),
        locator=str(wrapper_dir),
        loader="local",
        local_path=str(wrapper_dir),
        metadata={},
    )

    prepared = prepare_model_for_vllm(loaded)

    assert prepared.preparation_kind == "already_prepared_svdllm_llama_wrapper"
    assert prepared.model_path == str(wrapper_dir)
    assert prepared.tokenizer_mode == "slow"
    assert prepared.build_tokenizer_kwargs(trust_remote_code=True) == {
        "trust_remote_code": True,
        "use_fast": False,
    }

    llm_kwargs = prepared.build_llm_kwargs(
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        dtype="float16",
        enforce_eager=True,
        max_model_len=2048,
    )
    assert llm_kwargs["model"] == str(wrapper_dir)
    assert llm_kwargs["tokenizer_mode"] == "slow"
    assert llm_kwargs["model_impl"] == "transformers"
    assert llm_kwargs["disable_log_stats"] is True
    assert llm_kwargs["max_model_len"] == 2048


def test_prepare_model_for_vllm_direct_path_for_non_svd_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    loaded = LoadedCheckpoint(
        record=sample_record(),
        locator=str(model_dir),
        loader="local",
        local_path=str(model_dir),
        metadata={},
    )

    prepared = prepare_model_for_vllm(loaded)

    assert prepared.preparation_kind == "direct"
    assert prepared.model_path == str(model_dir)
    assert prepared.tokenizer_mode == "auto"


def test_prepare_model_for_vllm_normalizes_torch_dtype_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "torch_dtype": "torch.float16"}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    loaded = LoadedCheckpoint(
        record=sample_record(),
        locator=str(model_dir),
        loader="local",
        local_path=str(model_dir),
        metadata={},
    )

    prepared = prepare_model_for_vllm(loaded, wrapper_cache_root=tmp_path / "cache")

    prepared_config = json.loads((Path(prepared.model_path) / "config.json").read_text(encoding="utf-8"))
    assert prepared.preparation_kind == "config_torch_dtype_wrapper"
    assert prepared.model_path != str(model_dir)
    assert prepared.source_model_path == str(model_dir)
    assert prepared_config["torch_dtype"] == "float16"
    assert (Path(prepared.model_path) / "tokenizer.json").exists()


def test_prepare_model_for_vllm_uses_transformers_backend_for_dobi_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "dobi"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["DobiSVDLlamaForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_dobisvd_llama.DobiSVDLlamaConfig",
                    "AutoModel": "modeling_dobisvd_llama.DobiSVDLlamaModel",
                    "AutoModelForCausalLM": "modeling_dobisvd_llama.DobiSVDLlamaForCausalLM",
                },
                "dobi_target_modules": {
                    "model.layers.0.self_attn.q_proj": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = LoadedCheckpoint(
        record=sample_record(),
        locator=str(model_dir),
        loader="local",
        local_path=str(model_dir),
        metadata={},
    )

    prepared = prepare_model_for_vllm(loaded)

    prepared_path = Path(prepared.model_path)
    prepared_config = json.loads((prepared_path / "config.json").read_text(encoding="utf-8"))
    assert prepared.preparation_kind == "dobi_transformers_wrapper"
    assert prepared_path != model_dir
    assert prepared_config["architectures"] == ["TransformersForCausalLM"]
    assert prepared.model_impl == "transformers"


def test_prepare_model_for_vllm_uses_transformers_backend_for_qwen_custom_checkpoints(tmp_path: Path) -> None:
    cases = [
        (
            "asvd",
            {
                "model_type": "qwen3",
                "architectures": ["ASVDQwen3ForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_asvd_qwen3.ASVDQwen3Config",
                    "AutoModelForCausalLM": "modeling_asvd_qwen3.ASVDQwen3ForCausalLM",
                },
            },
            "asvd_qwen3_transformers_wrapper",
        ),
        (
            "dobi",
            {
                "model_type": "qwen3",
                "architectures": ["DobiSVDQwen3ForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_dobisvd_qwen3.DobiSVDQwen3Config",
                    "AutoModel": "modeling_dobisvd_qwen3.DobiSVDQwen3Model",
                    "AutoModelForCausalLM": "modeling_dobisvd_qwen3.DobiSVDQwen3ForCausalLM",
                },
                "dobi_target_modules": {"model.layers.0.self_attn.q_proj": 128},
            },
            "dobi_transformers_wrapper",
        ),
        (
            "lowrank",
            {
                "model_type": "lowrank_qwen3",
                "architectures": ["LowRankQwen3ForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_lowrank_qwen3.LowRankQwen3Config",
                    "AutoModel": "modeling_lowrank_qwen3.LowRankQwen3Model",
                    "AutoModelForCausalLM": "modeling_lowrank_qwen3.LowRankQwen3ForCausalLM",
                },
            },
            "lowrank_qwen3_transformers_wrapper",
        ),
    ]
    for dirname, config, expected_kind in cases:
        model_dir = tmp_path / dirname
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        loaded = LoadedCheckpoint(
            record=sample_record(),
            locator=str(model_dir),
            loader="local",
            local_path=str(model_dir),
            metadata={},
        )

        prepared = prepare_model_for_vllm(loaded)

        assert prepared.preparation_kind == expected_kind
        prepared_path = Path(prepared.model_path)
        assert prepared_path != model_dir
        prepared_config = json.loads((prepared_path / "config.json").read_text(encoding="utf-8"))
        wrapper_meta = json.loads(
            (prepared_path / TRANSFORMERS_BACKEND_WRAPPER_METADATA_NAME).read_text(encoding="utf-8")
        )
        assert prepared_config["architectures"] == ["TransformersForCausalLM"]
        assert prepared_config["auto_map"]["AutoModelForCausalLM"] == config["auto_map"]["AutoModelForCausalLM"]
        assert wrapper_meta["source_model"] == str(model_dir.resolve())
        assert prepared.tokenizer_mode == "auto"
        assert prepared.model_impl == "transformers"


def test_prepare_model_for_vllm_uses_transformers_backend_for_dense_qwen3(tmp_path: Path) -> None:
    model_dir = tmp_path / "qwen3"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    loaded = LoadedCheckpoint(
        record=sample_record(),
        locator=str(model_dir),
        loader="local",
        local_path=str(model_dir),
        metadata={},
    )

    prepared = prepare_model_for_vllm(loaded)

    assert prepared.preparation_kind == "qwen3_transformers_direct"
    assert prepared.model_path == str(model_dir)
    assert prepared.tokenizer_mode == "auto"
    assert prepared.model_impl == "transformers"


def test_prepare_model_for_vllm_uses_transformers_backend_for_modegpt_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "modegpt"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "auto_map": {
                    "AutoModelForCausalLM": "LlamaRebuild.LlamaForCausalLM",
                },
                "mask_path": "rotary_masks.pt",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    loaded = LoadedCheckpoint(
        record=CheckpointRecord(
            name="llama31-8b-modegpt-0.6-serving",
            model_family="llama3.1",
            variant="base",
            method="modegpt",
            source="local",
            repo_id="",
            revision="main",
            subpath="llama31_8b/MoDeGPT/keep60",
            benchmarks=["serving", "speed"],
            enabled=True,
            notes="test row",
        ),
        locator=str(model_dir),
        loader="local",
        local_path=str(model_dir),
        metadata={},
    )

    prepared = prepare_model_for_vllm(loaded)

    prepared_path = Path(prepared.model_path)
    prepared_config = json.loads((prepared_path / "config.json").read_text(encoding="utf-8"))
    assert prepared.preparation_kind == "modegpt_transformers_wrapper"
    assert prepared_path != model_dir
    assert prepared_config["architectures"] == ["TransformersForCausalLM"]
    assert prepared.tokenizer_mode == "auto"
    assert prepared.model_impl == "transformers"
    llm_kwargs = prepared.build_llm_kwargs(
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        dtype="float16",
        enforce_eager=False,
        max_model_len=None,
    )
    assert llm_kwargs["model_impl"] == "transformers"


def test_prepare_model_for_vllm_wraps_pruning_root_pt_weight(tmp_path: Path) -> None:
    model_dir = tmp_path / "slicegpt"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    source_weight = model_dir / "Llama-3.1-8B_0.6.pt"
    torch.save({"model.layers.0.self_attn.q_proj.weight": torch.ones(2, 3)}, source_weight)
    loaded = LoadedCheckpoint(
        record=CheckpointRecord(
            name="llama31-8b-slicegpt-prune-only-0.6",
            model_family="llama3.1",
            variant="base",
            method="slicegpt",
            source="local",
            repo_id="",
            revision="main",
            subpath="pruning/llama31_8b/SliceGPT/prune_only_0.6",
            benchmarks=["speed", "pruning"],
            enabled=True,
            notes="test row",
        ),
        locator=str(model_dir),
        loader="local",
        local_path=str(model_dir),
        metadata={},
    )

    prepared = prepare_model_for_vllm(loaded, wrapper_cache_root=tmp_path / "cache")

    prepared_path = Path(prepared.model_path)
    converted_weight = prepared_path / "model.safetensors"
    metadata_path = prepared_path / PRUNING_WEIGHT_WRAPPER_METADATA_NAME
    assert prepared.preparation_kind == "pruning_root_pt_safetensors_wrapper"
    assert prepared_path != model_dir
    assert prepared.tokenizer_path == str(prepared_path)
    assert converted_weight.exists()
    assert not (prepared_path / "pytorch_model.bin").exists()
    assert load_file(converted_weight)["model.layers.0.self_attn.q_proj.weight"].shape == (2, 3)
    assert (prepared_path / "config.json").resolve() == model_dir / "config.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_weight_file"] == str(source_weight.resolve())
    assert metadata["converted_weight_file"] == "model.safetensors"
    assert metadata["merged"] is False
    assert "model.safetensors" in " ".join(prepared.notes)
