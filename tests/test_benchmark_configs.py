from __future__ import annotations

from pathlib import Path

from src.benchmarking import select_checkpoints_for_suite
from src.registry import CheckpointRecord, save_checkpoint_index
from src.utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_suites_do_not_freeze_checkpoint_names() -> None:
    for path in (PROJECT_ROOT / "benchmark").glob("**/*.yaml"):
        if "tasks" in path.relative_to(PROJECT_ROOT / "benchmark").parts:
            continue
        config = load_yaml(path)
        assert "checkpoints" not in config.get("selection", {}), path


def test_accuracy_suites_keep_expected_backends_and_task_configs() -> None:
    mcq = load_yaml(PROJECT_ROOT / "benchmark" / "mcq.yaml")
    ppl = load_yaml(PROJECT_ROOT / "benchmark" / "ppl.yaml")
    mmlu = load_yaml(PROJECT_ROOT / "benchmark" / "mmlu.yaml")
    base_math = load_yaml(PROJECT_ROOT / "benchmark" / "base" / "base_math.yaml")
    mmlu_pro = load_yaml(PROJECT_ROOT / "benchmark" / "instruct" / "mmlu_pro.yaml")
    gsm8k = load_yaml(PROJECT_ROOT / "benchmark" / "instruct" / "gsm8k.yaml")
    aime = load_yaml(PROJECT_ROOT / "benchmark" / "instruct" / "aime.yaml")
    ifeval = load_yaml(PROJECT_ROOT / "benchmark" / "instruct" / "ifeval.yaml")
    base = load_yaml(PROJECT_ROOT / "benchmark" / "base.yaml")
    instruct = load_yaml(PROJECT_ROOT / "benchmark" / "instruct.yaml")
    instruct_appendix = load_yaml(PROJECT_ROOT / "benchmark" / "instruct_appendix.yaml")
    memory = load_yaml(PROJECT_ROOT / "benchmark" / "memory" / "active.yaml")

    assert mcq["eval"]["backend"] == "lm_eval_harness"
    assert mcq["eval"]["model_backend"] == "vllm"
    assert mcq["eval"]["version"] == "0.4.11"
    assert mcq["eval"]["dtype"] == "auto"
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
    assert ppl["eval"]["dtype"] == "auto"
    assert ppl["eval"]["metric"] == "ppl"
    assert ppl["eval"]["metric_aggregation"] == "macro_mean"
    assert ppl["eval"]["batch_size"] == 1
    assert ppl["eval"]["max_length"] == 2048
    assert ppl["eval"]["datasets"] == [
        {
            "name": "wikitext2",
            "kind": "wikitext2",
            "split": "test",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        },
        {
            "name": "c4_stream",
            "kind": "c4_stream",
            "path": "allenai/c4",
            "config_name": "en",
            "split": "validation",
            "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
            "document_offset": 0,
            "max_eval_tokens": 262144,
        },
    ]

    assert mmlu["eval"]["backend"] == "lm_eval_harness"
    assert mmlu["eval"]["version"] == "0.4.11"
    assert mmlu["eval"]["dtype"] == "auto"
    assert mmlu["eval"]["metric"] == "acc"
    assert mmlu["eval"]["metric_aggregation"] == "macro_mean"
    assert mmlu["eval"]["tasks"] == ["mmlu"]

    assert base_math["eval"]["backend"] == "lm_eval_harness"
    assert base_math["eval"]["model_backend"] == "vllm"
    assert base_math["eval"]["version"] == "0.4.11"
    assert base_math["eval"]["dtype"] == "auto"
    assert base_math["eval"]["metric"] == "acc_norm"
    assert base_math["eval"]["metric_fallbacks"] == ["acc"]
    assert base_math["eval"]["tracked_metrics"] == ["acc", "acc_norm"]
    assert base_math["eval"]["metric_aggregation"] == "macro_mean"
    assert base_math["eval"]["include_paths"] == ["benchmark/base/tasks"]
    assert base_math["eval"]["tasks"] == ["lra_mathqa", "mmlu_stem"]

    assert mmlu_pro["eval"]["backend"] == "lm_eval_harness"
    assert mmlu_pro["eval"]["model_backend"] == "vllm"
    assert mmlu_pro["eval"]["version"] == "0.4.11"
    assert mmlu_pro["eval"]["dtype"] == "auto"
    assert mmlu_pro["eval"]["metric"] == "acc"
    assert mmlu_pro["eval"]["tracked_metrics"] == ["acc"]
    assert mmlu_pro["eval"]["num_fewshot"] == 5
    assert "apply_chat_template" not in mmlu_pro["eval"]
    assert "gen_kwargs" not in mmlu_pro["eval"]
    assert "summary_source" not in mmlu_pro["eval"]
    assert "summary_entity" not in mmlu_pro["eval"]
    assert mmlu_pro["eval"]["tasks"] == ["leaderboard_mmlu_pro"]

    assert gsm8k["eval"]["backend"] == "lm_eval_harness"
    assert gsm8k["eval"]["model_backend"] == "vllm"
    assert gsm8k["eval"]["version"] == "0.4.11"
    assert gsm8k["eval"]["dtype"] == "auto"
    assert gsm8k["eval"]["metric"] == "exact_match"
    assert gsm8k["eval"]["tracked_metrics"] == ["exact_match"]
    assert gsm8k["eval"]["num_fewshot"] == 8
    assert "apply_chat_template" not in gsm8k["eval"]
    assert "gen_kwargs" not in gsm8k["eval"]
    assert "include_paths" not in gsm8k["eval"]
    assert gsm8k["eval"]["tasks"] == ["gsm8k_cot"]

    assert aime["eval"]["backend"] == "lm_eval_harness"
    assert aime["eval"]["model_backend"] == "vllm"
    assert aime["eval"]["version"] == "0.4.11"
    assert aime["eval"]["dtype"] == "auto"
    assert aime["eval"]["metric"] == "exact_match"
    assert aime["eval"]["tracked_metrics"] == ["exact_match"]
    assert aime["eval"]["num_fewshot"] == 0
    assert aime["eval"]["summary_source"] == "results"
    assert aime["eval"]["summary_entity"] == "aime24"
    assert aime["eval"]["solved_count_metric"] == "exact_match"
    assert "apply_chat_template" not in aime["eval"]
    assert "gen_kwargs" not in aime["eval"]
    assert aime["eval"]["tasks"] == ["aime24"]

    assert ifeval["eval"]["backend"] == "lm_eval_harness"
    assert ifeval["eval"]["model_backend"] == "vllm"
    assert ifeval["eval"]["version"] == "0.4.11"
    assert ifeval["eval"]["dtype"] == "auto"
    assert ifeval["eval"]["metric"] == "prompt_level_strict_acc"
    assert ifeval["eval"]["tracked_metrics"] == [
        "prompt_level_strict_acc",
        "inst_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_loose_acc",
    ]
    assert ifeval["eval"]["num_fewshot"] == 0
    assert ifeval["eval"]["apply_chat_template"] is True
    assert "gen_kwargs" not in ifeval["eval"]
    assert ifeval["eval"]["tasks"] == ["ifeval"]

    assert memory["name"] == "active_memory"
    assert memory["kind"] == "memory"
    assert memory["selection"] == {"enabled_only": True}
    assert memory["memory"]["backend"] == "transformers"
    assert memory["memory"]["dtype"] == "auto"
    assert memory["memory"]["batch_size"] == 1
    assert memory["memory"]["prompt_length"] == 512
    assert memory["memory"]["generation_length"] == 128

    for suite in [mcq, ppl, mmlu, base_math, mmlu_pro, gsm8k, aime, ifeval, base, instruct, instruct_appendix, memory]:
        assert "checkpoints" not in suite.get("selection", {})

    assert base["selection"]["variants"] == ["base"]
    assert base_math["selection"]["variants"] == ["base"]
    assert base["includes"] == [
        "ppl",
        "mcq",
        "base/base_math",
    ]
    assert mmlu_pro["selection"]["variants"] == ["instruct"]
    assert gsm8k["selection"]["variants"] == ["instruct"]
    assert instruct["selection"]["variants"] == ["instruct"]
    assert instruct["includes"] == [
        "instruct/mmlu_pro",
        "instruct/gsm8k",
    ]
    assert aime["selection"]["variants"] == ["instruct"]
    assert ifeval["selection"]["variants"] == ["instruct"]
    assert instruct_appendix["selection"]["variants"] == ["instruct"]
    assert instruct_appendix["includes"] == [
        "instruct/aime",
        "instruct/ifeval",
    ]

    serve = load_yaml(PROJECT_ROOT / "benchmark" / "speed" / "serve.yaml")
    assert serve["speed"]["metric_aggregation"] == "macro_mean"
    assert serve["selection"] == {"enabled_only": True}
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
    assert serve["speed"]["dtype"] == "auto"
    assert serve["speed"]["max_model_len"] == 3072

    edge = load_yaml(PROJECT_ROOT / "benchmark" / "speed" / "edge.yaml")
    assert edge["speed"]["metric_aggregation"] == "macro_mean"
    assert edge["selection"] == {"enabled_only": True}
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
    assert edge["speed"]["dtype"] == "auto"
    assert edge["speed"]["max_model_len"] == 16384

    eval_speed = load_yaml(PROJECT_ROOT / "benchmark" / "speed" / "speed.yaml")
    assert eval_speed["speed"]["backend"] == "evaluation"
    assert eval_speed["speed"]["model_backend"] == "vllm"
    assert eval_speed["speed"]["metric_aggregation"] == "macro_mean"
    assert eval_speed["selection"] == {"enabled_only": True}
    assert eval_speed["speed"]["eval_suites"] == [
        "ppl",
        "mcq",
        "base/base_math",
    ]
    assert eval_speed["speed"]["metrics"] == [
        "total_wall_time_seconds",
        "mean_suite_wall_time_seconds",
    ]


def test_aggregate_selection_override_supports_variant_lanes(tmp_path: Path) -> None:
    records = [
        CheckpointRecord(
            name="base-demo",
            model_family="llama",
            variant="base",
            method="demo",
            source="local",
            repo_id="",
            revision="main",
            subpath="base",
            benchmarks=["base"],
        ),
        CheckpointRecord(
            name="instruct-demo",
            model_family="llama",
            variant="instruct",
            method="demo",
            source="local",
            repo_id="",
            revision="main",
            subpath="instruct",
            benchmarks=["instruct"],
        ),
    ]
    index_path = tmp_path / "index.csv"
    save_checkpoint_index(records, path=index_path)

    shared_suite_config = {
        "selection": {
            "checkpoints": ["base-demo"],
            "enabled_only": True,
        }
    }

    selected = select_checkpoints_for_suite(
        shared_suite_config,
        index_path=index_path,
        selection_override={"variants": ["instruct"], "enabled_only": True},
    )

    assert [record.name for record in selected] == ["instruct-demo"]
