from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.benchmarking import resolve_suite_path, suite_output_name
from src.speed_runner import VllmSpeedRequest, run_speed_suite


@dataclass(slots=True)
class SpeedRequest:
    checkpoint_name: str
    suite: str = "speed/serve"
    batch_size: int = 1
    sequence_length: int = 2048
    generation_length: int = 128
    output_dir: str | Path | None = None
    suite_path: str | Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpeedResult:
    checkpoint_name: str
    suite: str
    status: str
    output_path: str
    stats: dict[str, Any]


def run_speed(request: SpeedRequest, index_path: str | None = None) -> SpeedResult:
    if index_path is None:
        raise ValueError("index_path is required for speed runs.")

    requested_suite_path = request.suite_path or request.extra.get("suite_path") or request.extra.get("benchmark_path") or request.suite
    suite_path = resolve_suite_path(requested_suite_path)
    result = run_speed_suite(
        VllmSpeedRequest(
            checkpoint_name=request.checkpoint_name,
            suite_path=suite_path,
            index_path=index_path,
            output_dir=request.output_dir,
            batch_sizes=request.extra.get("batch_sizes", [request.batch_size]),
            prompt_lengths=request.extra.get("prompt_lengths", [request.sequence_length]),
            generation_lengths=request.extra.get("generation_lengths", [request.generation_length]),
            repeat=request.extra.get("repeat"),
            warmup=request.extra.get("warmup"),
            tensor_parallel_size=request.extra.get("tensor_parallel_size"),
            gpu_memory_utilization=request.extra.get("gpu_memory_utilization"),
            dtype=request.extra.get("dtype"),
            enforce_eager=request.extra.get("enforce_eager"),
            lm_eval_bin=request.extra.get("lm_eval_bin"),
            eval_model_backend=request.extra.get("eval_model_backend"),
            eval_device=request.extra.get("eval_device"),
            eval_batch_size=request.extra.get("eval_batch_size"),
            eval_limit=request.extra.get("eval_limit"),
            eval_num_fewshot=request.extra.get("eval_num_fewshot"),
            verbose_backend=bool(request.extra.get("verbose_backend", False)),
            show_progress=bool(request.extra.get("show_progress", False)),
            run_label=str(request.extra.get("run_label", "ad_hoc")),
            strict_validation=bool(request.extra.get("strict_validation", False)),
        )
    )
    return SpeedResult(
        checkpoint_name=request.checkpoint_name,
        suite=suite_output_name(suite_path),
        status=result.status,
        output_path=result.output_path,
        stats=result.stats,
    )


def benchmark_checkpoint(
    checkpoint_name: str,
    suite: str = "speed/serve",
    batch_size: int = 1,
    sequence_length: int = 2048,
    generation_length: int = 128,
    output_dir: str | Path | None = None,
    index_path: str | None = None,
    extra: dict[str, Any] | None = None,
    suite_path: str | Path | None = None,
) -> SpeedResult:
    request = SpeedRequest(
        checkpoint_name=checkpoint_name,
        suite=suite,
        batch_size=batch_size,
        sequence_length=sequence_length,
        generation_length=generation_length,
        output_dir=output_dir,
        suite_path=suite_path,
        extra=extra or {},
    )
    return run_speed(request, index_path=index_path)
