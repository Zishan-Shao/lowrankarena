from __future__ import annotations

from pathlib import Path

from src.lm_eval_runner import _build_model_args
from src.load import load_checkpoint


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


def test_build_model_args_for_hf_checkpoint(tmp_path: Path) -> None:
    index_path = write_index(tmp_path / "index.csv")
    loaded = load_checkpoint("demo", index_path=str(index_path))
    model_args = _build_model_args(loaded, extra_model_args={"dtype": "float16"})

    assert model_args == {
        "pretrained": "Duke-CEI-SVD/LowRankArena",
        "revision": "main",
        "subfolder": "llama31_8b",
        "dtype": "float16",
    }
