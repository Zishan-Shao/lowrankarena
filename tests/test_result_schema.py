from __future__ import annotations

from pathlib import Path

from src.result_schema import build_result_payload
from src.registry import CheckpointRecord


def sample_record() -> CheckpointRecord:
    return CheckpointRecord(
        name="demo",
        model_family="llama3.1",
        variant="base",
        method="svdllm_v1_update",
        source="huggingface",
        repo_id="Duke-CEI-SVD/LowRankArena",
        revision="main",
        subpath="llama31_8b/SVDLLMv1/hf_whitening_then_update_0.6",
        benchmarks=["main", "speed"],
        enabled=True,
        notes="test row",
    )


def test_build_result_payload_uses_shared_top_level_shape() -> None:
    payload = build_result_payload(
        kind="speed",
        record=sample_record(),
        locator="hf://demo",
        backend_name="vllm",
        backend_version="0.18.1",
        suite_path=Path("benchmark") / "speed" / "speed.yaml",
        suite_name="speed",
        config={"repeat": 1},
        metrics={"mean_latency_seconds": 1.23},
        artifacts={"raw_path": "/tmp/demo.json"},
        runtime={"model_path": "/tmp/model"},
        details={"cases": []},
    )

    assert payload["schema_version"] == "1.0"
    assert payload["kind"] == "speed"
    assert payload["checkpoint"]["name"] == "demo"
    assert payload["checkpoint"]["locator"] == "hf://demo"
    assert payload["suite"] == {
        "id": "speed",
        "path": "speed/speed",
        "name": "speed",
    }
    assert payload["backend"] == {"name": "vllm", "version": "0.18.1"}
    assert payload["config"]["repeat"] == 1
    assert payload["metrics"]["mean_latency_seconds"] == 1.23
    assert payload["artifacts"]["raw_path"] == "/tmp/demo.json"
    assert payload["runtime"]["model_path"] == "/tmp/model"
    assert payload["details"]["cases"] == []


def test_build_result_payload_supports_runner_without_suite_path() -> None:
    payload = build_result_payload(
        kind="memory",
        record=sample_record(),
        locator="hf://demo",
        backend_name="transformers",
        backend_version=None,
        suite_path=None,
        suite_name="memory",
    )

    assert payload["suite"] == {
        "id": "memory",
        "path": None,
        "name": "memory",
    }
