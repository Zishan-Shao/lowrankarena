from __future__ import annotations

from pathlib import Path

from src import Arena
from src.registry import load_checkpoint_index
from src.utils import load_json


def write_index(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name,model_family,variant,method,source,repo_id,revision,subpath,benchmarks,enabled,notes",
                "demo,llama3.1,base,pending,huggingface,anonymous/lowrankarena-checkpoints,main,llama31_8b,base|speed,true,test row",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_arena_register_keeps_overlay_metadata(tmp_path: Path) -> None:
    index_path = write_index(tmp_path / "index.csv")
    manifests_root = tmp_path / "manifests"
    arena = Arena(registries=[index_path], manifests_root=manifests_root)

    description = arena.register(
        id="custom-demo",
        source="local",
        path="/tmp/custom-demo",
        model_family="llama3.1",
        variant="base",
        method="custom",
        benchmarks=["base"],
        enabled=False,
        family="svd",
        ratio=0.5,
        format="local_hf",
        eval_compatible=True,
        speed_compatible=True,
        notes="overlay only",
    )

    assert description["id"] == "custom-demo"
    assert description["ratio"] == 0.5
    assert description["format"] == "local_hf"
    assert description["eval_compatible"] is True
    assert any(row["id"] == "custom-demo" for row in arena.list(enabled_only=False))


def test_arena_register_manifest_persists_sidecar_and_index(tmp_path: Path) -> None:
    index_path = write_index(tmp_path / "index.csv")
    manifests_root = tmp_path / "manifests"
    input_manifest = tmp_path / "external.json"
    input_manifest.write_text(
        """
{
  "id": "external-demo",
  "source": "huggingface",
  "repo_id": "example-org/example-lowrank",
  "revision": "main",
  "subpath": "exports/demo",
  "model_family": "llama3.1",
  "variant": "base",
  "method": "custom",
  "family": "svd",
  "ratio": 0.6,
  "format": "custom_pt_bundle",
  "benchmarks": [],
  "enabled": false,
  "eval_compatible": false,
  "speed_compatible": false,
  "notes": "metadata only"
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    arena = Arena(registries=[index_path], manifests_root=manifests_root)
    description = arena.register_manifest(input_manifest, persist=True, index_path=index_path)

    assert description["id"] == "external-demo"
    persisted_manifest = manifests_root / "external-demo.json"
    assert persisted_manifest.exists()
    assert load_json(persisted_manifest)["eval_compatible"] is False

    records = load_checkpoint_index(index_path)
    record = next(record for record in records if record.name == "external-demo")
    assert record.repo_id == "example-org/example-lowrank"
    assert record.enabled is False
