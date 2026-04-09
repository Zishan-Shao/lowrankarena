from __future__ import annotations

from pathlib import Path

from src.eval import evaluate_checkpoint
from src.utils import load_json


def write_index(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name,model_family,variant,method,source,repo_id,revision,subpath,benchmarks,enabled,notes",
                "demo,llama3.1,base,pending,huggingface,Duke-CEI-SVD/LowRankArena,main,llama31_8b,main|speed,true,test row",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_evaluate_checkpoint_writes_stub_result(tmp_path: Path) -> None:
    index_path = write_index(tmp_path / "index.csv")
    output_dir = tmp_path / "results"
    result = evaluate_checkpoint(
        checkpoint_name="demo",
        suite="main",
        dataset="aggregate",
        output_dir=output_dir,
        index_path=str(index_path),
    )

    payload = load_json(result.output_path)
    assert result.status == "stub"
    assert payload["checkpoint"] == "demo"
    assert payload["suite"] == "main"
    assert payload["metrics"]["placeholder_score"] == 0.0
