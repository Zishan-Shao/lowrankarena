from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from compress.common import CompressionRequest, get_baseline_spec, prepare_baseline
from compress.svd.asvd import build as build_asvd
from compress.svd.basis_sharing import build as build_basis_sharing
from src.utils import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_asvd_rank_calculation_clamps_to_gqa_kv_matrix_rank() -> None:
    module_path = PROJECT_ROOT / "compress" / "svd" / "ASVD" / "modules" / "svd_linear.py"
    spec = spec_from_file_location("asvd_svd_linear", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.compute_svd_rank(4096, 1024, 2.0, rank_align=256) == 1024
    assert module.compressed_param_count(4096, 1024, 2.0, rank_align=256) == 1024 * (4096 + 1024)


def test_basis_sharing_weight_info_uses_gqa_kv_width() -> None:
    module_path = PROJECT_ROOT / "compress" / "svd" / "Basis_Sharing" / "config.py"
    spec = spec_from_file_location("basis_sharing_config", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    hf_config = SimpleNamespace(
        model_type="llama",
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    weight_info = module.ShareConfig.resolve_weight_info("unlisted-gqa-llama", hf_config)

    assert weight_info["self_attn.q_proj"] == (16, 16)
    assert weight_info["self_attn.k_proj"] == (16, 8)
    assert weight_info["self_attn.v_proj"] == (16, 8)


def test_build_basis_sharing_qwen3_command_for_instruct(tmp_path: Path) -> None:
    request = CompressionRequest(
        family="svd",
        method="basis_sharing",
        model="Qwen/Qwen3-8B",
        ratio=0.8,
        output_root=tmp_path / "artifacts",
        calibration="wikitext2",
        seed=7,
    )

    artifact = build_basis_sharing(request)
    command = artifact.manifest["command"]

    assert command[1].endswith("scripts/compress_qwen3_lowrank.py")
    assert command[command.index("--model-id") + 1] == "Qwen/Qwen3-8B"
    assert command[command.index("--method") + 1] == "basis_sharing"
    assert command[command.index("--keep-ratio") + 1] == "0.8"
    assert command[command.index("--basis-group-size") + 1] == "2"
    assert command[command.index("--output-dir") + 1].endswith("qwen3-8b_basis-sharing_r80/weights")
    assert artifact.manifest["ready_for_load"] is False


def test_build_basis_sharing_qwen3_command_for_base(tmp_path: Path) -> None:
    request = CompressionRequest(
        family="svd",
        method="basis_sharing",
        model="Qwen/Qwen3-8B-Base",
        ratio=0.6,
        output_root=tmp_path / "artifacts",
        extra={"basis_group_size": 4, "device": "cpu"},
    )

    artifact = build_basis_sharing(request)
    command = artifact.manifest["command"]

    assert command[command.index("--model-id") + 1] == "Qwen/Qwen3-8B-Base"
    assert command[command.index("--basis-group-size") + 1] == "4"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--output-dir") + 1].endswith("qwen3-8b-base_basis-sharing_r60/weights")


def test_build_basis_sharing_qwen3_execute_marks_artifact_ready(tmp_path: Path) -> None:
    request = CompressionRequest(
        family="svd",
        method="basis_sharing",
        model="Qwen/Qwen3-8B",
        ratio=0.5,
        output_root=tmp_path / "artifacts",
        execute=True,
    )

    import compress.svd.basis_sharing as basis_sharing_module

    original_run = basis_sharing_module.subprocess.run
    original_validate = basis_sharing_module.validate_checkpoint_layout
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0)

    basis_sharing_module.subprocess.run = fake_run
    basis_sharing_module.validate_checkpoint_layout = lambda *args, **kwargs: {"layout_kind": "basis_sharing_factorized"}
    try:
        artifact = build_basis_sharing(request)
    finally:
        basis_sharing_module.subprocess.run = original_run
        basis_sharing_module.validate_checkpoint_layout = original_validate

    assert artifact.manifest["status"] == "completed"
    assert artifact.manifest["ready_for_load"] is True
    assert captured["cwd"] == str(PROJECT_ROOT)
    executed = captured["command"]
    assert isinstance(executed, list)
    assert executed[executed.index("--output-dir") + 1].endswith("qwen3-8b_basis-sharing_r50")


def test_quant_rtn_has_no_git_repo() -> None:
    spec = get_baseline_spec("quant", "rtn")
    assert spec.git_url is None
