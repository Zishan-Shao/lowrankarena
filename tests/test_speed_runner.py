from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.registry import CheckpointRecord, save_checkpoint_index
from src.speed_runner import VllmSpeedRequest, _resolve_speed_case_specs, run_speed_suite


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


def test_run_speed_suite_supports_evaluation_backend() -> None:
    import src.speed_runner as speed_runner

    record = CheckpointRecord(
        name="demo",
        model_family="llama3.1",
        variant="base",
        method="dobi_svd",
        source="huggingface",
        repo_id="Duke-CEI-SVD/LowRankArena",
        revision="main",
        subpath="llama31_8b/DobiSVD/cuij_Llama_3_1_8B_DobiSVD_0.8",
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
        perf_values = iter([0.0, 10.0, 10.0, 22.0, 22.0, 34.0, 34.0, 40.0])

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
                    "mmlu_pro": {"mmlu_pro": 15},
                    "gsm8k": {"gsm8k": 25},
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
        assert payload["metrics"]["suite_count"] == 4
        assert payload["metrics"]["total_wall_time_seconds"] == 40.0
        assert payload["metrics"]["by_work_unit"]["tokens"]["total_count"] == 6144
        assert payload["metrics"]["by_work_unit"]["examples"]["total_count"] == 70
        assert [item["suite"] for item in payload["details"]["suites"]] == [
            "accuracy/ppl",
            "accuracy/mcq",
            "accuracy/mmlu_pro",
            "accuracy/gsm8k",
        ]
