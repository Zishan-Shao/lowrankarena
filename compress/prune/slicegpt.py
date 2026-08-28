from __future__ import annotations

from compress.common import CompressionRequest, prepare_baseline
from compress.save import CompressionArtifact, save_artifact


def build(request: CompressionRequest) -> CompressionArtifact:
    baseline = prepare_baseline(request)
    return save_artifact(
        request,
        baseline=baseline,
        output_format="custom_then_hf_export",
        status="planned",
        ready_for_load=False,
        notes=(
            "SliceGPT planning adapter. Unified execution is gated until the "
            "sliced architecture and checkpoint are exported together."
        ),
    )
