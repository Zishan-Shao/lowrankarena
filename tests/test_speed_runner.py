from __future__ import annotations

import json
from types import SimpleNamespace
import tempfile
from pathlib import Path

from src.registry import CheckpointRecord, save_checkpoint_index
from src.speed_runner import (
    VllmSpeedRequest,
    _normalize_bench_serve_case,
    _resolve_bench_serve_profiles,
    _resolve_speed_case_specs,
    run_speed_suite,
)


def test_resolve_speed_case_specs_prefers_named_suite_cases() -> None:
    speed_config = {
        "cases": [
            {
                "name": "interactive_short",
                "batch_size": 1,
                "prompt_length": 512,
                "generation_length": 128,
            },
            {
                "name": "throughput_long_decode",
                "batch_size": 4,
                "prompt_length": 512,
                "generation_length": 512,
            },
        ]
    }
    request = VllmSpeedRequest(
        checkpoint_name="demo",
        suite_path=Path("benchmark/speed/serve.yaml"),
        index_path=Path("checkpoints/index.csv"),
    )

    cases = _resolve_speed_case_specs(speed_config, request)

    assert cases == [
        {
            "name": "interactive_short",
            "batch_size": 1,
            "prompt_length": 512,
            "generation_length": 128,
        },
        {
            "name": "throughput_long_decode",
            "batch_size": 4,
            "prompt_length": 512,
            "generation_length": 512,
        },
    ]


def test_resolve_speed_case_specs_lets_cli_axes_override_named_cases() -> None:
    speed_config = {
        "cases": [
            {
                "name": "interactive_short",
                "batch_size": 1,
                "prompt_length": 512,
                "generation_length": 128,
            }
        ],
        "batch_sizes": [1],
        "prompt_lengths": [512],
        "generation_lengths": [128],
    }
    request = VllmSpeedRequest(
        checkpoint_name="demo",
        suite_path=Path("benchmark/speed/serve.yaml"),
        index_path=Path("checkpoints/index.csv"),
        batch_sizes=[4],
        prompt_lengths=[256],
        generation_lengths=[64],
    )

    cases = _resolve_speed_case_specs(speed_config, request)

    assert cases == [
        {
            "name": "batch4_prompt256_gen64",
            "batch_size": 4,
            "prompt_length": 256,
            "generation_length": 64,
        }
    ]


def test_resolve_bench_serve_profiles_prefers_named_profiles() -> None:
    speed_config = {
        "profiles": [
            {"name": "prefill", "input_length": 4096, "output_length": 32},
            {"name": "decode", "input_length": 512, "output_length": 512},
        ]
    }
    request = VllmSpeedRequest(
        checkpoint_name="demo",
        suite_path=Path("benchmark/speed/serve_e2e.yaml"),
        index_path=Path("checkpoints/index.csv"),
    )

    profiles = _resolve_bench_serve_profiles(speed_config, request)

    assert profiles == [
        {"name": "prefill", "input_length": 4096, "output_length": 32},
        {"name": "decode", "input_length": 512, "output_length": 512},
    ]


def test_normalize_bench_serve_case_extracts_latency_percentiles(tmp_path: Path) -> None:
    raw_payload = {
        "completed": 8,
        "failed": 0,
        "total_input": 4096,
        "total_output": 1024,
        "request_throughput": 1.25,
        "output_throughput": 160.0,
        "total_token_throughput": 800.0,
        "median_ttft_ms": 10.0,
        "percentiles_ttft_ms": [[50, 10.0], [95, 20.0], [99, 30.0]],
        "percentiles_tpot_ms": [[50, 4.0], [95, 5.0], [99, 6.0]],
        "percentiles_itl_ms": [[50, 3.0], [95, 4.0], [99, 5.0]],
        "percentiles_e2el_ms": [[50, 100.0], [95, 200.0], [99, 300.0]],
    }

    case = _normalize_bench_serve_case(
        profile={"name": "prefill", "input_length": 512, "output_length": 128},
        raw_payload=raw_payload,
        command=["vllm", "bench", "serve"],
        raw_output_path=tmp_path / "raw.json",
        wall_time_seconds=12.0,
    )

    assert case["request_throughput"] == 1.25
    assert case["output_tokens_per_second"] == 160.0
    assert case["ttft_ms"]["p50"] == 10.0
    assert case["ttft_ms"]["p95"] == 20.0
    assert case["tpot_ms"]["p99"] == 6.0
    assert case["e2e_latency_ms"]["p95"] == 200.0


def test_run_speed_suite_supports_evaluation_backend() -> None:
    import src.speed_runner as speed_runner

    record = CheckpointRecord(
        name="demo",
        model_family="llama3.1",
        variant="base",
        method="dobi_svd",
        source="huggingface",
        repo_id="anonymous/lowrankarena-checkpoints",
        revision="main",
        subpath="llama31_8b/dobi_svd/keep_0_8",
        benchmarks=["speed"],
        enabled=True,
        notes="test row",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        index_path = tmp_root / "index.csv"
        save_checkpoint_index([record], path=index_path)

        raw_root = tmp_root / "raw"
        output_root = tmp_root / "eval_outputs"
        original_runner = speed_runner.run_lm_eval_suite
        original_perf_counter = speed_runner.time.perf_counter
        perf_values = iter([0.0, 10.0, 10.0, 22.0, 22.0, 34.0])

        def fake_perf_counter() -> float:
            return next(perf_values)

        def fake_run_lm_eval_suite(request):  # type: ignore[no-untyped-def]
            suite_id = Path(request.suite_path).with_suffix("").name
            output_path = output_root / f"{suite_id}.json"
            raw_path = raw_root / f"{suite_id}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if suite_id == "ppl":
                raw_payload = {
                    "backend": "contiguous_ppl",
                    "tasks": [
                        {"name": "wikitext2", "token_count": 2048},
                        {"name": "c4_stream", "token_count": 4096},
                    ],
                }
                payload = {
                    "backend": {"name": "contiguous_ppl", "version": "1.0"},
                    "suite": {"name": "ppl"},
                    "metrics": {"primary_metric": "ppl", "mean": 12.3},
                    "validation": {"layout_kind": "fake"},
                }
            else:
                sample_counts = {
                    "mcq": {"boolq": 10, "piqa": 20},
                    "base_math": {"lra_mathqa": 15, "mmlu_stem": 15},
                }[suite_id]
                raw_payload = {
                    "total_evaluation_time_seconds": 5.0,
                    "n-samples": sample_counts,
                }
                payload = {
                    "backend": {"name": "lm_eval_harness", "version": "0.4.11"},
                    "suite": {"name": suite_id},
                    "metrics": {"primary_metric": "acc", "mean": 0.5},
                    "validation": {"layout_kind": "fake"},
                }
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")

            class FakeResult:
                def __init__(self) -> None:
                    self.status = "completed"
                    self.checkpoint_name = request.checkpoint_name
                    self.suite = suite_id
                    self.output_path = str(output_path)
                    self.raw_output_path = str(raw_path)
                    self.metrics = payload["metrics"]

            return FakeResult()

        speed_runner.run_lm_eval_suite = fake_run_lm_eval_suite
        speed_runner.time.perf_counter = fake_perf_counter
        try:
            result = run_speed_suite(
                VllmSpeedRequest(
                    checkpoint_name="demo",
                    suite_path=Path("benchmark/speed/speed.yaml"),
                    index_path=index_path,
                    output_dir=tmp_root / "speed_outputs",
                    run_label="smoke",
                )
            )
        finally:
            speed_runner.run_lm_eval_suite = original_runner
            speed_runner.time.perf_counter = original_perf_counter

        payload = json.loads(Path(result.output_path).read_text(encoding="utf-8"))
        assert payload["backend"]["name"] == "evaluation"
        assert payload["metrics"]["suite_count"] == 3
        assert payload["metrics"]["total_wall_time_seconds"] == 34.0
        assert payload["metrics"]["by_work_unit"]["tokens"]["total_count"] == 6144
        assert payload["metrics"]["by_work_unit"]["examples"]["total_count"] == 60
        assert "cuda_runtime" in payload["runtime"]
        assert "devices" in payload["runtime"]["cuda_runtime"]
        assert [item["suite"] for item in payload["details"]["suites"]] == [
            "ppl",
            "mcq",
            "base/base_math",
        ]


def test_run_speed_suite_supports_vllm_bench_serve_backend() -> None:
    import src.speed_runner as speed_runner

    record = CheckpointRecord(
        name="demo",
        model_family="llama3.1",
        variant="base",
        method="dobi_svd",
        source="local",
        repo_id="",
        revision="main",
        subpath="demo",
        benchmarks=["speed"],
        enabled=True,
        notes="test row",
    )

    class FakePopen:
        def __init__(self, command, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.command = command
            self.kwargs = kwargs
            self.returncode = None

        def poll(self):  # type: ignore[no-untyped-def]
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout=None):  # type: ignore[no-untyped-def]
            self.returncode = 0
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        index_path = tmp_root / "index.csv"
        save_checkpoint_index([record], path=index_path)
        model_root = tmp_root / "model"
        model_root.mkdir()

        loaded = SimpleNamespace(record=record, locator=str(model_root))
        prepared = SimpleNamespace(
            model_path=str(model_root),
            tokenizer_path=str(model_root),
            tokenizer_mode="slow",
            model_impl="transformers",
            preparation_kind="fake_vllm_wrapper",
            source_model_path=str(model_root),
            notes=["fake"],
        )

        original_load_checkpoint = speed_runner.load_checkpoint
        original_prepare_model_for_vllm = speed_runner.prepare_model_for_vllm
        original_validate_checkpoint_layout = speed_runner.validate_checkpoint_layout
        original_popen = speed_runner.subprocess.Popen
        original_run = speed_runner.subprocess.run
        original_sleep = speed_runner.time.sleep
        original_installed_vllm_version = speed_runner.installed_vllm_version

        def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
            result_dir = Path(command[command.index("--result-dir") + 1])
            result_filename = command[command.index("--result-filename") + 1]
            input_len = int(command[command.index("--input-len") + 1])
            output_len = int(command[command.index("--output-len") + 1])
            result_dir.mkdir(parents=True, exist_ok=True)
            raw_payload = {
                "completed": 64,
                "failed": 0,
                "total_input": 64 * input_len,
                "total_output": 64 * output_len,
                "request_throughput": 1.0,
                "request_goodput": 1.0,
                "output_throughput": 128.0,
                "total_token_throughput": 256.0,
                "max_output_tokens_per_s": 200.0,
                "max_concurrent_requests": 4,
                "percentiles_ttft_ms": [[50, 11.0], [95, 22.0], [99, 33.0]],
                "percentiles_tpot_ms": [[50, 4.0], [95, 5.0], [99, 6.0]],
                "percentiles_itl_ms": [[50, 3.0], [95, 4.0], [99, 5.0]],
                "percentiles_e2el_ms": [[50, 100.0], [95, 200.0], [99, 300.0]],
            }
            (result_dir / result_filename).write_text(json.dumps(raw_payload), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        speed_runner.load_checkpoint = lambda *args, **kwargs: loaded
        speed_runner.prepare_model_for_vllm = lambda *args, **kwargs: prepared
        speed_runner.validate_checkpoint_layout = lambda *args, **kwargs: {"layout_kind": "fake"}
        speed_runner.subprocess.Popen = FakePopen
        speed_runner.subprocess.run = fake_run
        speed_runner.time.sleep = lambda *args, **kwargs: None
        speed_runner.installed_vllm_version = lambda: "0.18.1"
        try:
            result = run_speed_suite(
                VllmSpeedRequest(
                    checkpoint_name="demo",
                    suite_path=Path("benchmark/speed/serve_e2e.yaml"),
                    index_path=index_path,
                    output_dir=tmp_root / "speed_outputs",
                    port=8123,
                    run_label="smoke",
                )
            )
        finally:
            speed_runner.load_checkpoint = original_load_checkpoint
            speed_runner.prepare_model_for_vllm = original_prepare_model_for_vllm
            speed_runner.validate_checkpoint_layout = original_validate_checkpoint_layout
            speed_runner.subprocess.Popen = original_popen
            speed_runner.subprocess.run = original_run
            speed_runner.time.sleep = original_sleep
            speed_runner.installed_vllm_version = original_installed_vllm_version

        payload = json.loads(Path(result.output_path).read_text(encoding="utf-8"))
        assert payload["backend"] == {"name": "vllm_bench_serve", "version": "0.18.1"}
        assert payload["config"]["port"] == 8123
        assert payload["config"]["dtype"] == "float16"
        assert payload["metrics"]["profile_count"] == 3
        assert payload["metrics"]["completed_requests"] == 192
        assert payload["metrics"]["mean_output_tokens_per_second"] == 128.0
        assert payload["metrics"]["mean_ttft_ms_p95"] == 22.0
        assert payload["metrics"]["mean_e2e_latency_ms_p99"] == 300.0
        assert payload["runtime"]["model_impl"] == "transformers"
        assert "--model-impl" in payload["runtime"]["server_command"]
        assert [profile["name"] for profile in payload["details"]["profiles"]] == [
            "prefill_heavy_4k_to_32",
            "balanced_2k_to_128",
            "decode_heavy_512_to_512",
        ]
