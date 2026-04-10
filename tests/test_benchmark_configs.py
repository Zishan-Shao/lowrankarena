from __future__ import annotations

from pathlib import Path

from src.utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_accuracy_suites_use_exact_lm_eval_task_ids() -> None:
    mcq = load_yaml(PROJECT_ROOT / "benchmark" / "accuracy" / "mcq.yaml")
    ppl = load_yaml(PROJECT_ROOT / "benchmark" / "accuracy" / "ppl.yaml")
    mmlu = load_yaml(PROJECT_ROOT / "benchmark" / "accuracy" / "mmlu.yaml")

    assert mcq["eval"]["backend"] == "lm_eval_harness"
    assert mcq["eval"]["version"] == "0.4.11"
    assert mcq["eval"]["metric"] == "acc_norm"
    assert mcq["eval"]["tasks"] == [
        "boolq",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "piqa",
        "hellaswag",
        "openbookqa",
    ]

    assert ppl["eval"]["backend"] == "lm_eval_harness"
    assert ppl["eval"]["version"] == "0.4.11"
    assert ppl["eval"]["metric"] == "word_perplexity"
    assert ppl["eval"]["tasks"] == ["wikitext", "paloma_ptb", "c4"]
    assert "c4_stream" not in ppl["eval"]["tasks"]
    assert "ptb" not in ppl["eval"]["tasks"]

    assert mmlu["eval"]["backend"] == "lm_eval_harness"
    assert mmlu["eval"]["version"] == "0.4.11"
    assert mmlu["eval"]["metric"] == "acc"
    assert mmlu["eval"]["tasks"] == ["mmlu"]
