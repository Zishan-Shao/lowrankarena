from __future__ import annotations

from pathlib import Path

from src.inference_adapter import PreparedInferenceModel
from src.lm_eval_runner import LmEvalRequest, _build_command, _build_model_args
from src.utils import project_path


def test_build_model_args_for_hf_checkpoint(tmp_path: Path) -> None:
    prepared = PreparedInferenceModel(
        model_path=str(tmp_path / "model"),
        tokenizer_path=str(tmp_path / "tokenizer"),
        tokenizer_mode="slow",
        preparation_kind="direct",
        source_model_path=str(tmp_path / "model"),
    )
    model_args = _build_model_args(prepared, extra_model_args={"dtype": "float16"})

    assert model_args == {
        "pretrained": str(tmp_path / "model"),
        "tokenizer": str(tmp_path / "tokenizer"),
        "use_fast_tokenizer": False,
        "dtype": "float16",
    }


def test_build_command_includes_suite_level_lm_eval_contract_flags(tmp_path: Path) -> None:
    suite_path = tmp_path / "custom.yaml"
    suite_path.write_text("name: placeholder\n", encoding="utf-8")
    request = LmEvalRequest(
        checkpoint_name="demo",
        suite_path=suite_path,
        index_path=str(tmp_path / "index.csv"),
        lm_eval_bin="lm-eval",
    )
    prepared = PreparedInferenceModel(
        model_path="/tmp/model",
        tokenizer_path="/tmp/model",
        tokenizer_mode="auto",
        preparation_kind="direct",
        source_model_path="/tmp/model",
    )
    suite_config = {
        "eval": {
            "tasks": ["custom_task"],
            "dtype": "float16",
            "num_fewshot": 5,
            "include_paths": ["benchmark"],
            "gen_kwargs": {"max_gen_toks": 64, "temperature": 0.0},
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
        }
    }

    command = _build_command(request, suite_config, tmp_path / "raw", prepared)

    assert "--include_path" in command
    assert str((project_path() / "benchmark").resolve()) in command
    assert "--gen_kwargs" in command
    assert "max_gen_toks=64" in command
    assert "temperature=0.0" in command
    assert "--fewshot_as_multiturn" in command
    assert "False" in command
    assert "--apply_chat_template" not in command


def test_build_command_clamps_max_gen_toks_to_model_context_window(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"max_position_embeddings": 2048}', encoding="utf-8")
    suite_path = tmp_path / "mmlu_pro.yaml"
    suite_path.write_text("name: mmlu_pro\n", encoding="utf-8")
    request = LmEvalRequest(
        checkpoint_name="demo",
        suite_path=suite_path,
        index_path=str(tmp_path / "index.csv"),
        lm_eval_bin="lm-eval",
    )
    prepared = PreparedInferenceModel(
        model_path=str(model_dir),
        tokenizer_path=str(model_dir),
        tokenizer_mode="auto",
        preparation_kind="direct",
        source_model_path=str(model_dir),
    )
    suite_config = {
        "eval": {
            "tasks": ["mmlu_pro"],
            "gen_kwargs": {"max_gen_toks": 4096, "temperature": 0.0},
        }
    }

    command = _build_command(request, suite_config, tmp_path / "raw", prepared)

    assert "--gen_kwargs" in command
    assert "max_gen_toks=2047" in command
    assert "max_gen_toks=4096" not in command
