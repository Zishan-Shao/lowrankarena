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
        notes="Basis-sharing scaffold. Implement the in-tree generation recipe here without changing the benchmark entrypoints.",
    )
