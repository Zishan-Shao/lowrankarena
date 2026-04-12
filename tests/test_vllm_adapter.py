from __future__ import annotations

import json
from pathlib import Path

from src.load import LoadedCheckpoint
from src.registry import CheckpointRecord
from src.vllm.prepare_svdllm_vllm_model import WRAPPER_METADATA_NAME
from src.vllm.vllm_adapter import prepare_model_for_vllm


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
