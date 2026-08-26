from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from src.load import build_hf_kwargs, load_checkpoint, load_from_record
from src.registry import CheckpointRecord, load_checkpoint_index


def write_index(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name,model_family,variant,method,source,repo_id,revision,subpath,benchmarks,enabled,notes",
                "demo,qwen3,base,pending,huggingface,Duke-CEI-SVD/LowRankArena,main,Qwen3-8b,main|speed,true,test row",
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
    assert loaded.locator == "hf://Duke-CEI-SVD/LowRankArena@main/Qwen3-8b"
    assert loaded.metadata["status"] == "resolved"


def test_build_hf_kwargs_includes_subfolder() -> None:
    records = load_checkpoint_index(Path(__file__).resolve().parents[1] / "checkpoints" / "index.csv")
    record = next(record for record in records if record.name == "llama31-8b-svdllm-0.6")
    kwargs = build_hf_kwargs(record, token="secret", cache_dir="/tmp/hf-cache")
    assert kwargs["subfolder"] == "llama31_8b/SVDLLMv1/hf_whitening_then_update_0.6"
    assert kwargs["revision"] == "main"
    assert kwargs["token"] == "secret"


def test_downloaded_hf_checkpoint_loads_from_self_contained_local_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    subpath = "checkpoints/low_rank/llama31_8b/basis_sharing/default_0.6"
    snapshot_path = tmp_path / "snapshot"
    checkpoint_path = snapshot_path / subpath
    checkpoint_path.mkdir(parents=True)
    calls: list[tuple[str, str, dict]] = []

    def fake_loader(kind: str):
        class Loader:
            @classmethod
            def from_pretrained(cls, source, **kwargs):
                calls.append((kind, str(source), kwargs))
                return kind

        return Loader

    fake_transformers = SimpleNamespace(
        AutoConfig=fake_loader("config"),
        AutoModelForCausalLM=fake_loader("model"),
        AutoTokenizer=fake_loader("tokenizer"),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("src.load.download_hf_snapshot", lambda *args, **kwargs: snapshot_path)

    record = CheckpointRecord(
        name="basis-sharing-demo",
        model_family="llama3.1",
        variant="base",
        method="basis_sharing",
        source="huggingface",
        repo_id="Duke-CEI-SVD/LowRankArena",
        revision="main",
        subpath=subpath,
        benchmarks=["main"],
    )
    loaded = load_from_record(
        record,
        download=True,
        load_config=True,
        load_tokenizer=True,
        load_model=True,
    )

    assert loaded.local_path == str(checkpoint_path.resolve())
    assert loaded.metadata["loading_mode"] == "snapshot_then_local"
    assert {kind for kind, _, _ in calls} == {"config", "model", "tokenizer"}
    for _, source, kwargs in calls:
        assert source == str(checkpoint_path.resolve())
        assert kwargs["local_files_only"] is True
        assert "subfolder" not in kwargs
