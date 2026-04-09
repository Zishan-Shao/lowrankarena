from __future__ import annotations

from pathlib import Path

from src.registry import load_checkpoint_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_expected_layout_exists() -> None:
    expected_paths = [
        "pyproject.toml",
        "requirements.txt",
        "src/load.py",
        "src/eval.py",
        "src/speed.py",
        "src/report.py",
        "src/registry.py",
        "src/utils.py",
        "benchmark/main.yaml",
        "benchmark/speed.yaml",
        "benchmark/modern.yaml",
        "benchmark/pruning.yaml",
        "benchmark/quant.yaml",
        "scripts/run_eval.py",
        "scripts/run_speed.py",
        "scripts/run_all.py",
        "scripts/make_table.py",
        "scripts/add_checkpoint.py",
        "checkpoints/index.csv",
        "checkpoints/README.md",
        "results/eval/.gitkeep",
        "results/speed/.gitkeep",
        "results/tables/.gitkeep",
        "results/figures/.gitkeep",
    ]
    for relative_path in expected_paths:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_default_checkpoint_manifest_is_seeded() -> None:
    records = load_checkpoint_index(PROJECT_ROOT / "checkpoints" / "index.csv")
    assert len(records) >= 9
    assert all(record.repo_id == "Duke-CEI-SVD/LowRankArena" for record in records)
    assert any(record.name == "llama31-8b-svdllm-0.6" for record in records)
