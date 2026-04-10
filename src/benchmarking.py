from __future__ import annotations

from pathlib import Path

from src.registry import CheckpointRecord, filter_checkpoints, load_checkpoint_index
from src.utils import load_yaml, project_path


BENCHMARK_ROOT = project_path("benchmark")


def discover_suite_paths(root: str | Path = BENCHMARK_ROOT) -> dict[str, Path]:
    benchmark_root = Path(root)
    configs: dict[str, Path] = {}
    for path in sorted(benchmark_root.glob("**/*.yaml")):
        relative = path.relative_to(benchmark_root)
        stem_key = path.stem
        relative_key = str(relative.with_suffix(""))
        configs[relative_key] = path
        configs.setdefault(stem_key, path)
    return configs


def resolve_suite_path(suite_name: str | Path, root: str | Path = BENCHMARK_ROOT) -> Path:
    candidate = Path(suite_name)
    if candidate.exists():
        return candidate.resolve()

    configs = discover_suite_paths(root=root)
    key = str(suite_name)
    if key not in configs:
        known = ", ".join(sorted(configs))
        raise FileNotFoundError(f"Unknown benchmark suite '{suite_name}'. Known suites: {known}")
    return configs[key]


def load_suite_config(suite_name: str | Path, root: str | Path = BENCHMARK_ROOT) -> tuple[Path, dict]:
    path = resolve_suite_path(suite_name, root=root)
    return path, load_yaml(path)


def suite_id(config_path: str | Path, root: str | Path = BENCHMARK_ROOT) -> str:
    path = Path(config_path).resolve()
    benchmark_root = Path(root).resolve()
    return str(path.relative_to(benchmark_root).with_suffix(""))


def suite_output_name(config_path: str | Path, root: str | Path = BENCHMARK_ROOT) -> str:
    relative_parts = list(Path(config_path).resolve().relative_to(Path(root).resolve()).with_suffix("").parts)
    if len(relative_parts) >= 2 and relative_parts[-1] == relative_parts[-2]:
        relative_parts.pop()
    return "__".join(relative_parts)


def select_checkpoints_for_suite(config: dict, index_path: str | Path) -> list[CheckpointRecord]:
    records = load_checkpoint_index(index_path)
    selection = config.get("selection", {})
    enabled_only = bool(selection.get("enabled_only", True))
    benchmarks = selection.get("benchmarks", [])

    if not benchmarks:
        return [record for record in records if record.enabled or not enabled_only]

    selected: list[CheckpointRecord] = []
    for benchmark in benchmarks:
        selected.extend(filter_checkpoints(records, benchmark=benchmark, enabled_only=enabled_only))

    deduped: dict[str, CheckpointRecord] = {}
    for record in selected:
        deduped[record.name] = record
    return list(deduped.values())
