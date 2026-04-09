from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.load import load_checkpoint
from src.utils import dump_json, ensure_dir, project_path, utc_timestamp


@dataclass(slots=True)
class EvalRequest:
    checkpoint_name: str
    suite: str = "main"
    dataset: str = "placeholder"
    output_dir: str | Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    checkpoint_name: str
    suite: str
    status: str
    output_path: str
    metrics: dict[str, Any]


def result_path_for(request: EvalRequest) -> Path:
    output_root = Path(request.output_dir) if request.output_dir else project_path("results", "eval")
    ensure_dir(output_root)
    filename = f"{request.suite}__{request.checkpoint_name}.json"
    return output_root / filename


def run_eval(request: EvalRequest, index_path: str | None = None) -> EvalResult:
    loaded = load_checkpoint(request.checkpoint_name, index_path=index_path)
    metrics = {
        "placeholder_score": 0.0,
        "ready_for_backend": False,
    }
    payload = {
        "kind": "eval",
        "checkpoint": loaded.record.name,
        "suite": request.suite,
        "dataset": request.dataset,
        "status": "stub",
        "locator": loaded.locator,
        "metrics": metrics,
        "extra": request.extra,
        "generated_at": utc_timestamp(),
        "notes": "This file is a scaffold artifact. Replace src.eval.run_eval with a real benchmark backend.",
    }
    output_path = dump_json(payload, result_path_for(request))
    return EvalResult(
        checkpoint_name=request.checkpoint_name,
        suite=request.suite,
        status="stub",
        output_path=str(output_path),
        metrics=metrics,
    )


def evaluate_checkpoint(
    checkpoint_name: str,
    suite: str = "main",
    dataset: str = "placeholder",
    output_dir: str | Path | None = None,
    index_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> EvalResult:
    request = EvalRequest(
        checkpoint_name=checkpoint_name,
        suite=suite,
        dataset=dataset,
        output_dir=output_dir,
        extra=extra or {},
    )
    return run_eval(request, index_path=index_path)
