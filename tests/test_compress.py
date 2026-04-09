from __future__ import annotations

from pathlib import Path

from compress.common import CompressionRequest, get_baseline_spec, prepare_baseline
from compress.svd.asvd import build as build_asvd
from src.utils import load_json


def test_prepare_baseline_resolves_existing_asvd_snapshot() -> None:
    request = CompressionRequest(
        family="svd",
        method="asvd",
        model="meta-llama/Llama-3.1-8B",
        ratio=0.5,
    )
    baseline = prepare_baseline(request)
    assert baseline.origin in {"vendored", "git"}
    assert baseline.path is not None
    assert baseline.path.endswith("compress/svd/ASVD")


def test_build_asvd_writes_manifest(tmp_path: Path) -> None:
    request = CompressionRequest(
        family="svd",
        method="asvd",
        model="meta-llama/Llama-3.1-8B",
        ratio=0.5,
        output_root=tmp_path / "artifacts",
    )
    artifact = build_asvd(request)
    payload = load_json(artifact.manifest_path)
    assert payload["family"] == "svd"
    assert payload["method"] == "asvd"
    assert payload["ratio"] == 0.5
    assert payload["ready_for_load"] is False
    assert payload["baseline"]["name"] == "ASVD"


def test_quant_rtn_has_no_git_repo() -> None:
    spec = get_baseline_spec("quant", "rtn")
    assert spec.git_url is None
