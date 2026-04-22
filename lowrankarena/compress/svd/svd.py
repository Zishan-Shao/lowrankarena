from __future__ import annotations

from compress.common import CompressionRequest, prepare_baseline
from compress.save import CompressionArtifact, save_artifact


def build(request: CompressionRequest) -> CompressionArtifact:
    baseline = prepare_baseline(request)
    return save_artifact(
        request,
        baseline=baseline,
        output_format="transformers_export",
        status="planned",
        ready_for_load=False,
        notes="Plain SVD scaffold. Keep dense-model loading and export metadata uniform with other methods.",
    )
