from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.load import load_checkpoint
from src.utils import dump_json, ensure_dir, project_path, utc_timestamp


@dataclass(slots=True)
class SpeedRequest:
    checkpoint_name: str
    suite: str = "speed"
    batch_size: int = 1
    sequence_length: int = 2048
    output_dir: str | Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpeedResult:
    checkpoint_name: str
    suite: str
    status: str
    output_path: str
    stats: dict[str, Any]


def result_path_for(request: SpeedRequest) -> Path:
    output_root = Path(request.output_dir) if request.output_dir else project_path("results", "speed")
    ensure_dir(output_root)
    filename = f"{request.suite}__{request.checkpoint_name}.json"
    return output_root / filename


def run_speed(request: SpeedRequest, index_path: str | None = None) -> SpeedResult:
    loaded = load_checkpoint(request.checkpoint_name, index_path=index_path)
    stats = {
        "latency_ms": 0.0,
        "tokens_per_second": 0.0,
        "ready_for_backend": False,
    }
    payload = {
        "kind": "speed",
        "checkpoint": loaded.record.name,
        "suite": request.suite,
        "batch_size": request.batch_size,
        "sequence_length": request.sequence_length,
        "status": "stub",
        "locator": loaded.locator,
        "stats": stats,
        "extra": request.extra,
        "generated_at": utc_timestamp(),
        "notes": "This file is a scaffold artifact. Replace src.speed.run_speed with a real profiler backend.",
    }
    output_path = dump_json(payload, result_path_for(request))
    return SpeedResult(
        checkpoint_name=request.checkpoint_name,
        suite=request.suite,
        status="stub",
        output_path=str(output_path),
        stats=stats,
    )


def benchmark_checkpoint(
    checkpoint_name: str,
    suite: str = "speed",
    batch_size: int = 1,
    sequence_length: int = 2048,
    output_dir: str | Path | None = None,
    index_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> SpeedResult:
    request = SpeedRequest(
        checkpoint_name=checkpoint_name,
        suite=suite,
        batch_size=batch_size,
        sequence_length=sequence_length,
        output_dir=output_dir,
        extra=extra or {},
    )
    return run_speed(request, index_path=index_path)
