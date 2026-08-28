from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import compress.common as common
from compress.common import (
    CompressionRequest,
    get_baseline_spec,
    preflight_request,
    prepare_baseline,
)
from compress.svd.asvd import build as build_asvd
from compress.svd import aa_svd as aa_svd_module
from compress.svd.aa_svd import build as build_aa_svd
from compress.svd.gfw_svd import load_rank_config
from compress.svd.swift_svd import build as build_swift_svd
from compress.svd.zs_svd import build as build_zs_svd
from scripts.run_compress import MODULE_BY_METHOD, list_methods
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


def test_all_public_methods_have_importable_planning_adapters(tmp_path: Path) -> None:
    for (family, method), module_name in sorted(MODULE_BY_METHOD.items()):
        spec = get_baseline_spec(family, method)
        assert spec.family == family
        assert spec.method == method
        module = importlib.import_module(module_name)
        request = CompressionRequest(
            family=family,
            method=method,
            model="test-org/test-model",
            ratio=0.5,
            output_root=tmp_path / "artifacts",
        )
        artifact = module.build(request)
        payload = json.loads(Path(artifact.manifest_path).read_text(encoding="utf-8"))
        assert payload["status"] == "planned"
        assert payload["ready_for_load"] is False


def test_method_listing_reports_execution_capability() -> None:
    rows = list_methods()
    assert len(rows) == len(MODULE_BY_METHOD)
    executable = {row["method"] for row in rows if row["supports_execute"]}
    assert executable == {"aa_svd", "gfw_svd", "swift_svd", "zs_svd"}


def test_gfw_spec_pins_fisherkronecker_commit() -> None:
    spec = get_baseline_spec("svd", "gfw_svd")
    assert spec.git_url == "https://github.com/sayankotor/FisherKronecker.git"
    assert spec.git_ref == "d009b028c1e73545d8c604bcd29c1e091c8f341c"
    assert spec.supports_execute is True
    assert spec.required_extra == ("kron_factors_dir",)


def test_gfw_rank_config_accepts_ratio_objects_and_weight_suffix(tmp_path: Path) -> None:
    path = tmp_path / "ranks.json"
    path.write_text(
        json.dumps(
            {
                "model.layers.0.self_attn.q_proj.weight": 0.5,
                "model.layers.0.mlp.up_proj": {"keep_ratio": 0.6},
            }
        ),
        encoding="utf-8",
    )
    assert load_rank_config(path) == {
        "model.layers.0.self_attn.q_proj": 0.5,
        "model.layers.0.mlp.up_proj": 0.6,
    }


def test_gfw_execute_preflight_requires_factor_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(common, "_package_available", lambda _package: True)
    request = CompressionRequest(
        family="svd",
        method="gfw_svd",
        model="test-org/test-model",
        ratio=0.5,
        execute=True,
    )
    missing = preflight_request(request, for_execute=True)
    assert missing.ok is False
    assert any("kron_factors_dir" in error for error in missing.errors)

    factors = tmp_path / "factors"
    factors.mkdir()
    request.extra["kron_factors_dir"] = str(factors)
    ready = preflight_request(request, for_execute=True)
    assert ready.ok is True


def test_plan_only_method_is_rejected_before_execute(monkeypatch) -> None:
    monkeypatch.setattr(common, "_package_available", lambda _package: True)
    request = CompressionRequest(
        family="svd",
        method="asvd",
        model="test-org/test-model",
        ratio=0.5,
        execute=True,
    )
    report = preflight_request(request, for_execute=True)
    assert report.ok is False
    assert any("does not yet support --execute" in error for error in report.errors)


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_new_wrappers_preserve_recorded_export_defaults(tmp_path: Path) -> None:
    common_fields = {
        "family": "svd",
        "model": "test-org/test-model",
        "ratio": 0.6,
        "output_root": tmp_path / "artifacts",
    }
    aa = build_aa_svd(CompressionRequest(method="aa_svd", **common_fields))
    swift = build_swift_svd(
        CompressionRequest(
            method="swift_svd",
            extra={"svd_file": str(tmp_path / "svd.pkl")},
            **common_fields,
        )
    )
    zs = build_zs_svd(CompressionRequest(method="zs_svd", **common_fields))

    assert _option_value(aa.manifest["command"], "--model-dtype") == "bfloat16"
    assert _option_value(aa.manifest["command"], "--target-dtype") == "float16"
    assert _option_value(swift.manifest["command"], "--target-dtype") == "float16"
    assert _option_value(zs.manifest["command"], "--seed") == "3"
    assert _option_value(zs.manifest["command"], "--target-dtype") == "float16"


def test_aa_adapter_default_command_matches_recorded_recipe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        aa_svd_module,
        "infer_model_config",
        lambda _model, _revision, _explicit: "llama3-8B",
    )
    monkeypatch.setattr(
        aa_svd_module,
        "run_logged",
        lambda command, **_kwargs: commands.append(command),
    )
    args = SimpleNamespace(
        upstream_root=tmp_path,
        model="/models/llama31-8b",
        revision="main",
        model_config=None,
        keep_ratio=0.6,
        calibration="wikitext2",
        calibration_file=None,
        output_dir=tmp_path / "weights",
        model_dtype="bfloat16",
        target_dtype="float16",
        unsafe_overwrite=False,
    )

    aa_svd_module.execute(args)
    main_command, export_command = commands
    assert main_command[:5] == [
        sys.executable,
        "-u",
        "main.py",
        "model=llama3-8B",
        "model.name=/models/llama31-8b",
    ]
    assert "model.dtype=bfloat16" in main_command
    assert "model.revision=main" not in main_command
    assert "data=wikitext2" not in main_command
    assert _option_value(export_command, "--target-dtype") == "float16"


def test_recorded_reproduction_shell_commands_still_parse() -> None:
    for relative in (
        "scripts/run_aasvd_keep_sweep_20260724.sh",
        "scripts/run_new_method_keep_sweep_20260724.sh",
    ):
        subprocess.run(["bash", "-n", relative], check=True)


def test_legacy_run_compress_invocation_still_works(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_compress.py",
            "--family",
            "svd",
            "--method",
            "asvd",
            "--model",
            "test-org/test-model",
            "--ratio",
            "0.5",
            "--output-root",
            str(tmp_path / "artifacts"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["manifest"]["method"] == "asvd"
    assert payload["manifest"]["status"] == "planned"
