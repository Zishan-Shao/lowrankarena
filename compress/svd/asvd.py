from __future__ import annotations

from compress.common import CompressionRequest, build_method_command, prepare_baseline
from compress.save import CompressionArtifact, save_artifact


def build(request: CompressionRequest) -> CompressionArtifact:
    baseline = prepare_baseline(request)
    command = build_method_command(request, baseline, relative_output_dir="out")
    return save_artifact(
        request,
        baseline=baseline,
        command=command,
        output_format="transformers_export",
        status="planned",
        ready_for_load=False,
        notes="ASVD artifact scaffold. The main benchmark should still use released checkpoints by default.",
    )
