from __future__ import annotations

from pathlib import Path

from src.load import _cached_snapshot_with_subpath, build_hf_kwargs, load_checkpoint
from src.registry import load_checkpoint_index


def write_index(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name,model_family,variant,method,source,repo_id,revision,subpath,benchmarks,enabled,notes",
                "demo,qwen3,base,pending,huggingface,anonymous/lowrankarena-checkpoints,main,Qwen3-8b,base|speed,true,test row",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_checkpoint_index_reads_rows(tmp_path: Path) -> None:
    index_path = write_index(tmp_path / "index.csv")
    records = load_checkpoint_index(index_path)
    assert len(records) == 1
    assert records[0].name == "demo"
    assert records[0].enabled is True


def test_load_checkpoint_builds_hf_locator(tmp_path: Path) -> None:
    index_path = write_index(tmp_path / "index.csv")
    loaded = load_checkpoint("demo", index_path=str(index_path))
    assert loaded.loader == "huggingface"
    assert loaded.locator == "hf://anonymous/lowrankarena-checkpoints@main/Qwen3-8b"
    assert loaded.metadata["status"] == "resolved"


def test_cached_snapshot_with_subpath_falls_back_to_existing_partial_snapshot(tmp_path: Path) -> None:
    snapshots_root = tmp_path / "snapshots"
    current = snapshots_root / "newer"
    older = snapshots_root / "older"
    current.mkdir(parents=True)
    (older / "llama_7b" / "ASVD" / "demo").mkdir(parents=True)

    assert _cached_snapshot_with_subpath(current, "llama_7b/ASVD/demo") == older


def test_build_hf_kwargs_includes_subfolder() -> None:
    records = load_checkpoint_index(Path(__file__).resolve().parents[1] / "checkpoints" / "index.csv")
    record = next(record for record in records if record.name == "llama31-8b-svdllm-0.6")
    kwargs = build_hf_kwargs(record, token="secret", cache_dir="/tmp/hf-cache")
    assert kwargs["subfolder"] == "llama31_8b/svdllm/keep_0_6"
    assert kwargs["revision"] == "main"
    assert kwargs["token"] == "secret"
