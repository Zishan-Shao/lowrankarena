from __future__ import annotations

from compress.common import CompressionRequest, prepare_baseline
from compress.save import CompressionArtifact, save_artifact


def build(request: CompressionRequest) -> CompressionArtifact:
    baseline = prepare_baseline(request)
    return save_artifact(
        request,
        baseline=baseline,
        output_format="rtn_export",
        status="planned",
        ready_for_load=False,
        notes="RTN scaffold. This can likely be implemented in-tree with transformers/bitsandbytes rather than a third-party repo.",
    )
