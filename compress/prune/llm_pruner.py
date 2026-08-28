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
            "LLM-Pruner planning adapter. Unified execution is gated until its "
            "structural-pruning output is materialized as a loadable HF artifact."
        ),
    )
