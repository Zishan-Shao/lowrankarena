from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from compress.common import CompressionRequest, prepare_baseline
from compress.save import CompressionArtifact, save_artifact
from src.utils import project_path
from src.validation import validate_checkpoint_layout


_SUPPORTED_QWEN3_MODELS = frozenset({"qwen3-8b", "qwen3-8b-base"})
_DEFAULT_BASIS_GROUP_SIZE = 2
_DEFAULT_DEVICE = "cuda:0"
_DEFAULT_MAX_SHARD_SIZE = "5GB"
_DEFAULT_CALIBRATION_SAMPLES = 64
_DEFAULT_SEQUENCE_LENGTH = 2048


def _normalized_model_token(model: str) -> str:
    return model.split("/")[-1].strip().lower().replace("_", "-")


def _supports_qwen3_basis_sharing(model: str) -> bool:
    return _normalized_model_token(model) in _SUPPORTED_QWEN3_MODELS


def _artifact_output_dir(request: CompressionRequest, *, materialize_at_root: bool) -> Path:
    artifact_dir = Path(request.artifact_root) / request.artifact_id
    if materialize_at_root:
        return artifact_dir
    return artifact_dir / "weights"


def _string_extra(request: CompressionRequest, key: str, default: str) -> str:
    value = request.extra.get(key)
    if value in (None, ""):
        return default
    return str(value)


def _int_extra(request: CompressionRequest, key: str, default: int) -> int:
    value = request.extra.get(key)
    if value in (None, ""):
        return int(default)
    return int(value)


def _bool_extra(request: CompressionRequest, key: str, default: bool = False) -> bool:
    value = request.extra.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_qwen3_basis_sharing_command(
    request: CompressionRequest,
    *,
    output_dir: Path,
) -> list[str]:
    if request.ratio is None:
        raise ValueError("Basis sharing requires --ratio for Qwen3 exports.")

    command = [
        sys.executable,
        str(project_path("scripts", "compress_qwen3_lowrank.py")),
        "--model-id",
        request.model,
        "--method",
        "basis_sharing",
        "--keep-ratio",
        str(request.ratio),
        "--output-dir",
        str(output_dir),
        "--device",
        _string_extra(request, "device", _DEFAULT_DEVICE),
        "--dataset",
        request.calibration,
        "--calibration-samples",
        str(_int_extra(request, "calibration_samples", _DEFAULT_CALIBRATION_SAMPLES)),
        "--sequence-length",
        str(_int_extra(request, "sequence_length", _DEFAULT_SEQUENCE_LENGTH)),
        "--seed",
        str(request.seed),
        "--basis-group-size",
        str(_int_extra(request, "basis_group_size", _DEFAULT_BASIS_GROUP_SIZE)),
        "--max-shard-size",
        _string_extra(request, "max_shard_size", _DEFAULT_MAX_SHARD_SIZE),
    ]
    if _bool_extra(request, "unsafe_overwrite"):
        command.append("--unsafe-overwrite")
    return command


def build(request: CompressionRequest) -> CompressionArtifact:
    baseline = prepare_baseline(request)

    if not _supports_qwen3_basis_sharing(request.model):
        return save_artifact(
            request,
            baseline=baseline,
            output_format="transformers_export",
            status="planned",
            ready_for_load=False,
            notes=(
                "Basis-sharing is now wired up in-tree for Qwen/Qwen3-8B and Qwen/Qwen3-8B-Base. "
                "Other models still use the scaffold path for now."
            ),
        )

    output_dir = _artifact_output_dir(request, materialize_at_root=request.execute)
    command = _build_qwen3_basis_sharing_command(request, output_dir=output_dir)

    if request.execute:
        subprocess.run(command, cwd=str(project_path()), check=True)
        summary = validate_checkpoint_layout(output_dir, strict=True)
        return save_artifact(
            request,
            baseline=baseline,
            command=command,
            output_format="transformers_export",
            status="completed",
            ready_for_load=True,
            notes=(
                "Executed the in-tree Qwen3 basis-sharing exporter and validated the resulting checkpoint "
                f"layout ({summary['layout_kind']})."
            ),
        )

    return save_artifact(
        request,
        baseline=baseline,
        command=command,
        output_format="transformers_export",
        status="planned",
        ready_for_load=False,
        notes=(
            "Planned the in-tree Qwen3 basis-sharing export command. "
            "Run the generated planned_command.sh or pass --execute to materialize the checkpoint."
        ),
    )
