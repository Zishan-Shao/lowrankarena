from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compress.common import BaselineHandle, CompressionRequest, shell_join
from src.registry import CheckpointRecord, load_checkpoint_index, save_checkpoint_index
from src.utils import dump_json, ensure_dir, project_path


@dataclass(slots=True)
class CompressionArtifact:
    artifact_id: str
    artifact_dir: str
    manifest_path: str
    log_path: str
    manifest: dict[str, Any]
    baseline_path: str | None = None
    command_path: str | None = None
    execution_log_path: str | None = None
    registered_name: str | None = None


def _relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(project_path().resolve()))


def _write_command_script(
    artifact_dir: Path,
    baseline: BaselineHandle | None,
    command: list[str] | None,
) -> Path | None:
    if not command:
        return None
    script_path = artifact_dir / "planned_command.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    if baseline and baseline.path:
        lines.append(shell_join(["cd", baseline.path]))
    lines.append(shell_join(command))
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def save_artifact(
    request: CompressionRequest,
    *,
    baseline: BaselineHandle | None = None,
    command: list[str] | None = None,
    output_format: str = "transformers",
    status: str = "planned",
    ready_for_load: bool = False,
    notes: str = "",
    register: bool | None = None,
) -> CompressionArtifact:
    artifact_dir = ensure_dir(request.artifact_root / request.artifact_id)
    ensure_dir(artifact_dir / "weights")

    manifest = request.to_manifest_fields()
    manifest.update(
        {
            "output_dir": str(artifact_dir.resolve()),
            "output_format": output_format,
            "status": status,
            "ready_for_load": ready_for_load,
            "baseline": {
                "name": baseline.spec.display_name,
                "path": baseline.path,
                "origin": baseline.origin,
                "git_url": baseline.spec.git_url,
                "git_ref": baseline.spec.git_ref,
                "entrypoint": baseline.spec.entrypoint,
                "notes": baseline.spec.notes,
            }
            if baseline
            else None,
            "command": command,
            "notes": notes or request.notes,
        }
    )

    manifest_path = dump_json(manifest, artifact_dir / "manifest.json")
    log_payload = {
        "status": status,
        "ready_for_load": ready_for_load,
        "artifact_id": request.artifact_id,
        "command": command,
        "baseline_path": baseline.path if baseline else None,
    }
    log_path = dump_json(log_payload, artifact_dir / "compression_log.json")
    command_path = _write_command_script(artifact_dir, baseline, command)

    registered_name: str | None = None
    should_register = register if register is not None else request.register
    if should_register:
        if ready_for_load:
            registered_name = register_artifact(request, artifact_dir, enabled=request.enabled)
        else:
            warning_path = artifact_dir / "register_skipped.txt"
            warning_path.write_text(
                "Registration was skipped because the artifact is not yet marked ready_for_load.\n",
                encoding="utf-8",
            )

    return CompressionArtifact(
        artifact_id=request.artifact_id,
        artifact_dir=str(artifact_dir.resolve()),
        manifest_path=str(manifest_path),
        log_path=str(log_path),
        manifest=manifest,
        baseline_path=baseline.path if baseline else None,
        command_path=str(command_path) if command_path else None,
        registered_name=registered_name,
    )


def _is_loadable_hf_artifact(weights_dir: Path) -> bool:
    if not (weights_dir / "config.json").is_file():
        return False
    weight_candidates = (
        weights_dir / "model.safetensors",
        weights_dir / "model.safetensors.index.json",
        weights_dir / "pytorch_model.bin",
        weights_dir / "pytorch_model.bin.index.json",
    )
    return any(path.is_file() for path in weight_candidates)


def execute_artifact(
    request: CompressionRequest,
    artifact: CompressionArtifact,
    *,
    baseline: BaselineHandle | None = None,
) -> CompressionArtifact:
    """Execute a planned command and atomically update its artifact metadata."""

    if not artifact.command_path:
        raise RuntimeError(f"{request.family}/{request.method} has no executable command")

    artifact_dir = Path(artifact.artifact_dir)
    manifest_path = Path(artifact.manifest_path)
    log_path = Path(artifact.log_path)
    execution_log_path = artifact_dir / "execution.log"
    manifest = dict(artifact.manifest)
    manifest["status"] = "running"
    manifest["ready_for_load"] = False
    dump_json(manifest, manifest_path)

    with execution_log_path.open("w", encoding="utf-8") as execution_log:
        process = subprocess.run(
            ["bash", artifact.command_path],
            cwd=artifact_dir,
            stdout=execution_log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    ready_for_load = process.returncode == 0 and _is_loadable_hf_artifact(
        artifact_dir / "weights"
    )
    manifest["status"] = "completed" if ready_for_load else "failed"
    manifest["ready_for_load"] = ready_for_load
    manifest["returncode"] = process.returncode
    manifest["execution_log"] = str(execution_log_path)
    dump_json(manifest, manifest_path)
    dump_json(
        {
            "status": manifest["status"],
            "ready_for_load": ready_for_load,
            "artifact_id": request.artifact_id,
            "command": manifest.get("command"),
            "baseline_path": baseline.path if baseline else artifact.baseline_path,
            "returncode": process.returncode,
            "execution_log": str(execution_log_path),
        },
        log_path,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Compression command failed with exit code {process.returncode}; "
            f"see {execution_log_path}"
        )
    if not ready_for_load:
        raise RuntimeError(
            f"Compression command completed but did not produce a loadable HF artifact in "
            f"{artifact_dir / 'weights'}; see {execution_log_path}"
        )

    registered_name = artifact.registered_name
    if request.register:
        registered_name = register_artifact(
            request,
            artifact_dir / "weights",
            enabled=request.enabled,
        )

    return CompressionArtifact(
        artifact_id=artifact.artifact_id,
        artifact_dir=artifact.artifact_dir,
        manifest_path=artifact.manifest_path,
        log_path=artifact.log_path,
        manifest=manifest,
        baseline_path=artifact.baseline_path,
        command_path=artifact.command_path,
        execution_log_path=str(execution_log_path),
        registered_name=registered_name,
    )


def register_artifact(
    request: CompressionRequest,
    artifact_dir: Path,
    *,
    enabled: bool = False,
    index_path: str | Path | None = None,
) -> str:
    index_file = Path(index_path) if index_path else project_path("checkpoints", "index.csv")
    records = [record for record in load_checkpoint_index(index_file) if record.name != request.artifact_id]
    relative_subpath = _relative_to_project(artifact_dir)
    records.append(
        CheckpointRecord(
            name=request.artifact_id,
            model_family=request.model_slug,
            variant="generated",
            method=request.method,
            source="local",
            repo_id="",
            revision="",
            subpath=relative_subpath,
            benchmarks=["main", "speed"],
            enabled=enabled,
            notes=f"Generated by compress/{request.family}/{request.method}.",
        )
    )
    save_checkpoint_index(records, index_file)
    return request.artifact_id
