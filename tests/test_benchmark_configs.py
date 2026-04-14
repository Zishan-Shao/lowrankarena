from __future__ import annotations

from pathlib import Path

from src.utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_accuracy_suites_keep_expected_backends_and_task_configs() -> None:
    mcq = load_yaml(PROJECT_ROOT / "benchmark" / "accuracy" / "mcq.yaml")
    ppl = load_yaml(PROJECT_ROOT / "benchmark" / "accuracy" / "ppl.yaml")
    mmlu = load_yaml(PROJECT_ROOT / "benchmark" / "accuracy" / "mmlu.yaml")
    mmlu_pro = load_yaml(PROJECT_ROOT / "benchmark" / "accuracy" / "mmlu_pro.yaml")
    gsm8k = load_yaml(PROJECT_ROOT / "benchmark" / "accuracy" / "gsm8k.yaml")
    main = load_yaml(PROJECT_ROOT / "benchmark" / "main.yaml")

    assert mcq["eval"]["backend"] == "lm_eval_harness"
    assert mcq["eval"]["version"] == "0.4.11"
    assert mcq["eval"]["dtype"] == "float16"
    assert mcq["eval"]["metric"] == "acc_norm"
    assert mcq["eval"]["metric_fallbacks"] == ["acc"]
    assert mcq["eval"]["tracked_metrics"] == ["acc", "acc_norm"]
    assert mcq["eval"]["metric_aggregation"] == "macro_mean"
    assert "limit" not in mcq["eval"]
    assert mcq["eval"]["tasks"] == [
        "boolq",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "piqa",
        "hellaswag",
        "openbookqa",
    ]

    assert ppl["eval"]["backend"] == "contiguous_ppl"
    assert ppl["eval"]["version"] == "1.0"
    assert ppl["eval"]["dtype"] == "float16"
    assert ppl["eval"]["metric"] == "ppl"
    assert ppl["eval"]["metric_aggregation"] == "macro_mean"
    assert ppl["eval"]["batch_size"] == 1
    assert ppl["eval"]["max_length"] == 2048
    assert ppl["eval"]["datasets"] == [
        {"name": "wikitext2", "kind": "wikitext2", "split": "test"},
        {
            "name": "c4_stream",
            "kind": "c4_stream",
            "split": "validation",
            "max_eval_tokens": 262144,
        },
    ]

    assert mmlu["eval"]["backend"] == "lm_eval_harness"
    assert mmlu["eval"]["version"] == "0.4.11"
    assert mmlu["eval"]["dtype"] == "float16"
    assert mmlu["eval"]["metric"] == "acc"
    assert mmlu["eval"]["metric_aggregation"] == "macro_mean"
    assert mmlu["eval"]["tasks"] == ["mmlu"]

    assert mmlu_pro["eval"]["backend"] == "lm_eval_harness"
    assert mmlu_pro["eval"]["version"] == "0.4.11"
    assert mmlu_pro["eval"]["dtype"] == "float16"
    assert mmlu_pro["eval"]["metric"] == "exact_match"
    assert mmlu_pro["eval"]["tracked_metrics"] == ["exact_match"]
    assert mmlu_pro["eval"]["num_fewshot"] == 5
    assert mmlu_pro["eval"]["apply_chat_template"] is False
    assert mmlu_pro["eval"]["summary_source"] == "groups"
    assert mmlu_pro["eval"]["summary_entity"] == "mmlu_pro"
    assert mmlu_pro["eval"]["tasks"] == ["mmlu_pro"]

    assert gsm8k["eval"]["backend"] == "lm_eval_harness"
    assert gsm8k["eval"]["version"] == "0.4.11"
    assert gsm8k["eval"]["dtype"] == "float16"
    assert gsm8k["eval"]["metric"] == "exact_match"
    assert gsm8k["eval"]["tracked_metrics"] == ["exact_match"]
    assert gsm8k["eval"]["num_fewshot"] == 5
    assert gsm8k["eval"]["apply_chat_template"] is False
    assert "include_paths" not in gsm8k["eval"]
    assert gsm8k["eval"]["tasks"] == ["gsm8k"]

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
    assert mmlu_pro["selection"]["checkpoints"] == frozen_checkpoints
    assert gsm8k["selection"]["checkpoints"] == frozen_checkpoints
    assert main["includes"] == [
        "accuracy/ppl",
        "accuracy/mcq",
        "accuracy/mmlu_pro",
        "accuracy/gsm8k",
        "memory/leaderboard",
        "speed/serve",
    ]

    serve = load_yaml(PROJECT_ROOT / "benchmark" / "speed" / "serve.yaml")
    assert serve["speed"]["metric_aggregation"] == "macro_mean"
    assert serve["selection"]["checkpoints"] == frozen_checkpoints
    assert [case["name"] for case in serve["speed"]["cases"]] == [
        "interactive_short",
        "interactive_long_decode",
        "interactive_long_context",
        "balanced_short",
        "balanced_long_decode",
        "throughput_short",
        "throughput_long_decode",
    ]
    assert sorted({case["batch_size"] for case in serve["speed"]["cases"]}) == [1, 2, 4]
    assert sorted({case["prompt_length"] for case in serve["speed"]["cases"]}) == [512, 1024, 2048]
    assert sorted({case["generation_length"] for case in serve["speed"]["cases"]}) == [128, 512]
    assert serve["speed"]["repeat"] == 7
    assert serve["speed"]["gpu_memory_utilization"] == 0.35
    assert serve["speed"]["dtype"] == "float16"
    assert serve["speed"]["max_model_len"] == 3072

    edge = load_yaml(PROJECT_ROOT / "benchmark" / "speed" / "edge.yaml")
    assert edge["speed"]["metric_aggregation"] == "macro_mean"
    assert edge["selection"]["checkpoints"] == frozen_checkpoints
    assert [case["name"] for case in edge["speed"]["cases"]] == [
        "edge_prefill_4k_b4",
        "edge_decode_4k_b4",
        "edge_prefill_8k_b2",
        "edge_decode_8k_b2",
        "edge_prefill_near_16k_b1",
        "edge_decode_near_16k_b1",
    ]
    assert sorted({case["batch_size"] for case in edge["speed"]["cases"]}) == [1, 2, 4]
    assert sorted({case["generation_length"] for case in edge["speed"]["cases"]}) == [128, 512]
    assert edge["speed"]["repeat"] == 7
    assert edge["speed"]["gpu_memory_utilization"] == 0.6
    assert edge["speed"]["dtype"] == "float16"
    assert edge["speed"]["max_model_len"] == 16384
