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
    assert mcq["eval"]["dtype"] == "float16"
    assert mcq["eval"]["tracked_metrics"] == ["acc", "acc_norm"]
    assert "metric" not in mcq["eval"]
    assert "metric_fallbacks" not in mcq["eval"]
    assert mcq["eval"]["metric_aggregation"] == "macro_mean"
    assert mcq["eval"]["limit"] == 200
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
    assert ppl["eval"]["dtype"] == "float16"
    assert ppl["eval"]["metric"] == "word_perplexity"
    assert ppl["eval"]["metric_aggregation"] == "macro_mean"
    assert ppl["eval"]["tasks"] == ["wikitext", "paloma_ptb", "c4"]
    assert "c4_stream" not in ppl["eval"]["tasks"]
    assert "ptb" not in ppl["eval"]["tasks"]

    assert mmlu["eval"]["backend"] == "lm_eval_harness"
    assert mmlu["eval"]["version"] == "0.4.11"
    assert mmlu["eval"]["dtype"] == "float16"
    assert mmlu["eval"]["metric"] == "acc"
    assert mmlu["eval"]["metric_aggregation"] == "macro_mean"
    assert mmlu["eval"]["tasks"] == ["mmlu"]

    frozen_checkpoints = [
        "llama31-8b-asvd-0.4",
        "llama31-8b-dobi-0.8",
        "llama31-8b-svdllm-v1-update-0.6",
        "llama31-8b-svdllm-v2-0.6",
        "llama31-8b-basis-sharing-0.6",
        "llama-7b-svdllm-v1-update-0.5",
        "llama-7b-basis-sharing-0.5",
        "llama-7b-dobi-0.8",
    ]
    assert mcq["selection"]["checkpoints"] == frozen_checkpoints
    assert ppl["selection"]["checkpoints"] == frozen_checkpoints
    assert mmlu["selection"]["checkpoints"] == frozen_checkpoints

    speed = load_yaml(PROJECT_ROOT / "benchmark" / "speed" / "leaderboard.yaml")
    assert speed["speed"]["metric_aggregation"] == "macro_mean"
    assert speed["selection"]["checkpoints"] == frozen_checkpoints
    assert speed["speed"]["batch_sizes"] == [1]
    assert speed["speed"]["prompt_lengths"] == [512]
    assert speed["speed"]["generation_lengths"] == [128]
    assert speed["speed"]["gpu_memory_utilization"] == 0.35
    assert speed["speed"]["dtype"] == "float16"
