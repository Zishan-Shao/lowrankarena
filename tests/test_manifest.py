from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from src.registry import load_checkpoint_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_exporter_module():
    exporter = PROJECT_ROOT / "compress" / "svd" / "SVD-LLM" / "huggingface_repos" / "export_svdllm_lowrank.py"
    spec = spec_from_file_location("export_svdllm_lowrank", exporter)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expected_layout_exists() -> None:
    expected_paths = [
        "pyproject.toml",
        "requirements.txt",
        "benchmark/README.md",
        "src/load.py",
        "src/__init__.py",
        "src/arena.py",
        "src/README.md",
        "src/modeling/README.md",
        "src/modeling/__init__.py",
        "src/modeling/common.py",
        "src/modeling/llama/README.md",
        "src/modeling/llama/__init__.py",
        "src/modeling/llama/configuration_lowrank_llama.py",
        "src/modeling/llama/modeling_lowrank_llama.py",
        "src/modeling/mistral/README.md",
        "src/modeling/mistral/__init__.py",
        "src/modeling/mistral/configuration_lowrank_mistral.py",
        "src/modeling/mistral/modeling_lowrank_mistral.py",
        "src/modeling/qwen/README.md",
        "src/modeling/qwen/__init__.py",
        "src/modeling/qwen/configuration_lowrank_qwen2.py",
        "src/modeling/qwen/modeling_lowrank_qwen2.py",
        "src/modeling/qwen/configuration_lowrank_qwen3.py",
        "src/modeling/qwen/modeling_lowrank_qwen3.py",
        "src/loader.py",
        "src/benchmarking.py",
        "src/dtype_utils.py",
        "src/lm_eval_runner.py",
        "src/ppl_runner.py",
        "src/memory_runner.py",
        "src/speed_runner.py",
        "src/scoring.py",
        "src/eval.py",
        "src/memory.py",
        "src/result_schema.py",
        "src/validation.py",
        "src/speed.py",
        "src/report.py",
        "src/registry.py",
        "src/utils.py",
        "benchmark/base.yaml",
        "benchmark/instruct.yaml",
        "benchmark/mcq.yaml",
        "benchmark/mmlu.yaml",
        "benchmark/ppl.yaml",
        "benchmark/base/README.md",
        "benchmark/base/base_math.yaml",
        "benchmark/base/tasks/lra_mathqa.yaml",
        "benchmark/base/tasks/utils.py",
        "benchmark/instruct/README.md",
        "benchmark/instruct/mmlu_pro.yaml",
        "benchmark/instruct/gsm8k.yaml",
        "benchmark/memory/active.yaml",
        "benchmark/speed/README.md",
        "benchmark/speed/serve.yaml",
        "benchmark/speed/edge.yaml",
        "benchmark/speed/speed.yaml",
        "scripts/README.md",
        "scripts/run_eval.py",
        "scripts/run_memory.py",
        "scripts/run_speed.py",
        "scripts/run_all.py",
        "scripts/run_main.py",
        "scripts/run_compress.py",
        "scripts/make_table.py",
        "scripts/add_checkpoint.py",
        "checkpoints/index.csv",
        "checkpoints/README.md",
        "checkpoints/vllm/README.md",
        "checkpoints/manifests/README.md",
        "compress/README.md",
        "compress/artifacts/README.md",
        "compress/common.py",
        "compress/save.py",
        "compress/svd/README.md",
        "compress/svd/asvd.py",
        "compress/svd/basis_sharing.py",
        "compress/svd/dobi_svd.py",
        "compress/svd/fwsvd.py",
        "compress/svd/svd.py",
        "compress/svd/svd_llm.py",
        "compress/prune/README.md",
        "compress/prune/bonsai.py",
        "compress/prune/llm_pruner.py",
        "compress/prune/slicegpt.py",
        "compress/prune/wanda_sp.py",
        "compress/quant/README.md",
        "compress/quant/awq.py",
        "compress/quant/gptq.py",
        "compress/quant/rtn.py",
        "compress/artifacts/.gitkeep",
        "results/README.md",
        "results/eval/README.md",
        "results/eval/.gitkeep",
        "results/memory/README.md",
        "results/memory/.gitkeep",
        "results/speed/README.md",
        "results/speed/.gitkeep",
        "results/tables/README.md",
        "results/tables/.gitkeep",
        "results/figures/README.md",
        "results/figures/.gitkeep",
        "tests/README.md",
        "tests/test_arena.py",
        "tests/test_benchmark_configs.py",
        "tests/test_memory.py",
        "tests/test_modeling.py",
        "tests/test_ppl_runner.py",
        "tests/test_result_schema.py",
        "tests/test_scoring.py",
        "tests/test_validation.py",
        "tests/test_vllm_adapter.py",
    ]
    for relative_path in expected_paths:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_default_checkpoint_manifest_is_seeded() -> None:
    records = load_checkpoint_index(PROJECT_ROOT / "checkpoints" / "index.csv")
    assert len(records) >= 9
    assert any(record.name == "llama31-8b-svdllm-0.6" for record in records)


def test_exporter_uses_repo_modeling_root() -> None:
    module = _load_exporter_module()
    assert module.MODELING_ROOT == PROJECT_ROOT / "src" / "modeling"


def test_exporter_supports_mistral_family() -> None:
    module = _load_exporter_module()
    spec = module._select_model_spec(SimpleNamespace(config=SimpleNamespace(model_type="mistral")))

    assert spec["architectures"] == ["LowRankMistralForCausalLM"]
    assert spec["source_dir"] == PROJECT_ROOT / "src" / "modeling" / "mistral"
    assert spec["copy_files"] == (
        "../common.py",
        "configuration_lowrank_mistral.py",
        "modeling_lowrank_mistral.py",
    )
