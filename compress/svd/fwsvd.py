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
        notes="FWSVD scaffold. This method is expected to use local project code rather than a third-party repo.",
    )
