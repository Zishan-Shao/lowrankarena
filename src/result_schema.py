from __future__ import annotations

from pathlib import Path
from typing import Any

from src.benchmarking import suite_id, suite_output_name
from src.registry import CheckpointRecord
from src.utils import utc_timestamp


def checkpoint_payload(record: CheckpointRecord, *, locator: str) -> dict[str, Any]:
    return {
        "name": record.name,
        "locator": locator,
        "model_family": record.model_family,
        "variant": record.variant,
        "method": record.method,
        "source": record.source,
        "repo_id": record.repo_id,
        "revision": record.revision,
        "subpath": record.subpath,
        "benchmarks": list(record.benchmarks),
        "enabled": record.enabled,
        "notes": record.notes,
    }


def suite_payload(
    *,
    kind: str,
    suite_path: str | Path | None,
    suite_name: str | None = None,
) -> dict[str, Any]:
    if suite_path is None:
        fallback_name = suite_name or kind
        return {
            "id": fallback_name,
            "path": None,
            "name": fallback_name,
        }

    resolved_path = Path(suite_path)
    return {
        "id": suite_output_name(resolved_path),
        "path": suite_id(resolved_path),
        "name": suite_name or resolved_path.stem,
    }


def backend_payload(*, name: str, version: str | None) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
    }


def build_result_payload(
    *,
    kind: str,
    record: CheckpointRecord,
    locator: str,
    backend_name: str,
    backend_version: str | None,
    suite_path: str | Path | None,
    suite_name: str | None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "kind": kind,
        "status": status,
        "generated_at": utc_timestamp(),
        "checkpoint": checkpoint_payload(record, locator=locator),
        "suite": suite_payload(kind=kind, suite_path=suite_path, suite_name=suite_name),
        "backend": backend_payload(name=backend_name, version=backend_version),
        "config": config or {},
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "runtime": runtime or {},
        "details": details or {},
    }
