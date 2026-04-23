from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.benchmarking import resolve_suite_path, suite_output_name
from src.lm_eval_runner import LmEvalRequest, run_lm_eval_suite


@dataclass(slots=True)
class EvalRequest:
    checkpoint_name: str
    suite: str = "mcq"
    dataset: str = "lm_eval_harness"
    output_dir: str | Path | None = None
    limit: float | int | None = None
    device: str | None = None
    batch_size: str | int | None = None
    suite_path: str | Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    checkpoint_name: str
    suite: str
    status: str
    output_path: str
    metrics: dict[str, Any]


def run_eval(request: EvalRequest, index_path: str | None = None) -> EvalResult:
    if index_path is None:
        raise ValueError("index_path is required for eval runs.")

    requested_suite_path = request.suite_path or request.extra.get("suite_path") or request.extra.get("benchmark_path") or request.suite
    suite_path = resolve_suite_path(requested_suite_path)
    result = run_lm_eval_suite(
        LmEvalRequest(
            checkpoint_name=request.checkpoint_name,
            suite_path=suite_path,
            index_path=index_path,
            output_dir=request.output_dir,
            model_backend=request.extra.get("model_backend"),
            device=request.device,
            batch_size=request.batch_size,
            limit=request.limit,
            tensor_parallel_size=request.extra.get("tensor_parallel_size"),
            gpu_memory_utilization=request.extra.get("gpu_memory_utilization"),
            max_model_len=request.extra.get("max_model_len"),
            enforce_eager=request.extra.get("enforce_eager"),
            extra_model_args=request.extra.get("model_args", {}),
        )
    )
    return EvalResult(
        checkpoint_name=request.checkpoint_name,
        suite=suite_output_name(suite_path),
        status=result.status,
        output_path=result.output_path,
        metrics=result.metrics,
    )


def evaluate_checkpoint(
    checkpoint_name: str,
    suite: str = "mcq",
    dataset: str = "lm_eval_harness",
    output_dir: str | Path | None = None,
    index_path: str | None = None,
    extra: dict[str, Any] | None = None,
    limit: float | int | None = None,
    device: str | None = None,
    batch_size: str | int | None = None,
    suite_path: str | Path | None = None,
) -> EvalResult:
    request = EvalRequest(
        checkpoint_name=checkpoint_name,
        suite=suite,
        dataset=dataset,
        output_dir=output_dir,
        limit=limit,
        device=device,
        batch_size=batch_size,
        suite_path=suite_path,
        extra=extra or {},
    )
    return run_eval(request, index_path=index_path)
