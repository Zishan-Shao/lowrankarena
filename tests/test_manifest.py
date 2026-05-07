from __future__ import annotations

from pathlib import Path

from src.registry import load_checkpoint_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_expected_benchmark_only_layout_exists() -> None:
    expected_paths = [
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        "benchmark/README.md",
        "benchmark/base.yaml",
        "benchmark/instruct.yaml",
        "benchmark/instruct_appendix.yaml",
        "benchmark/mcq.yaml",
        "benchmark/mmlu.yaml",
        "benchmark/ppl.yaml",
        "benchmark/ppl_smoke.yaml",
        "benchmark/base/base_math.yaml",
        "benchmark/base/tasks/lra_mathqa.yaml",
        "benchmark/base/tasks/mmlu_math.yaml",
        "benchmark/instruct/mmlu_pro.yaml",
        "benchmark/instruct/gsm8k.yaml",
        "benchmark/instruct/aime.yaml",
        "benchmark/instruct/ifeval.yaml",
        "benchmark/memory/active.yaml",
        "benchmark/speed/serve.yaml",
        "benchmark/speed/serve_e2e.yaml",
        "benchmark/speed/edge.yaml",
        "benchmark/speed/speed.yaml",
        "checkpoints/index.csv",
        "checkpoints/README.md",
        "checkpoints/manifests/README.md",
        "checkpoints/vllm/README.md",
        "scripts/README.md",
        "scripts/add_checkpoint.py",
        "scripts/make_table.py",
        "scripts/measure_peak_memory.py",
        "scripts/run_all.py",
        "scripts/run_eval.py",
        "scripts/run_main.py",
        "scripts/run_memory.py",
        "scripts/run_speed.py",
        "src/__init__.py",
        "src/arena.py",
        "src/benchmarking.py",
        "src/dtype_utils.py",
        "src/lm_eval_runner.py",
        "src/load.py",
        "src/memory_runner.py",
        "src/modeling/common.py",
        "src/modeling/llama/modeling_lowrank_llama.py",
        "src/modeling/mistral/modeling_lowrank_mistral.py",
        "src/modeling/qwen/modeling_lowrank_qwen2.py",
        "src/modeling/qwen/modeling_lowrank_qwen3.py",
        "src/ppl_runner.py",
        "src/registry.py",
        "src/result_schema.py",
        "src/scoring.py",
        "src/speed_runner.py",
        "src/validation.py",
        "src/vllm/vllm_adapter.py",
        "results/README.md",
        "results/eval/README.md",
        "results/memory/README.md",
        "results/speed/README.md",
        "results/tables/README.md",
        "results/figures/README.md",
        "tests/README.md",
        "tests/test_arena.py",
        "tests/test_benchmark_configs.py",
        "tests/test_eval.py",
        "tests/test_load.py",
        "tests/test_manifest.py",
        "tests/test_memory.py",
        "tests/test_modeling.py",
        "tests/test_ppl_runner.py",
        "tests/test_result_schema.py",
        "tests/test_scoring.py",
        "tests/test_speed_runner.py",
        "tests/test_validation.py",
        "tests/test_vllm_adapter.py",
    ]
    for relative_path in expected_paths:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_default_checkpoint_manifest_is_anonymized() -> None:
    records = load_checkpoint_index(PROJECT_ROOT / "checkpoints" / "index.csv")
    assert len(records) >= 8
    assert any(record.name == "llama31-8b-svdllm-0.6" for record in records)
    assert all(record.repo_id != "private/hosted-checkpoints" for record in records)
