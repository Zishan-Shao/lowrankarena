from __future__ import annotations

from pathlib import Path

from src.registry import CheckpointRecord, load_checkpoint_index
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


def _selection_items(selection: dict, key: str) -> list[str]:
    return [str(item) for item in selection.get(key, []) if str(item).strip()]


def _matches_selection(record: CheckpointRecord, selection: dict, *, enabled_only: bool) -> bool:
    if enabled_only and not record.enabled:
        return False

    benchmarks = _selection_items(selection, "benchmarks")
    if benchmarks and not any(benchmark in record.benchmarks for benchmark in benchmarks):
        return False

    variants = _selection_items(selection, "variants")
    if variants and record.variant not in variants:
        return False

    model_families = _selection_items(selection, "model_families")
    if model_families and record.model_family not in model_families:
        return False

    methods = _selection_items(selection, "methods")
    if methods and record.method not in methods:
        return False

    return True


def select_checkpoints_for_suite(
    config: dict,
    index_path: str | Path,
    *,
    selection_override: dict | None = None,
) -> list[CheckpointRecord]:
    records = load_checkpoint_index(index_path)
    selection = selection_override if selection_override is not None else config.get("selection", {})
    enabled_only = bool(selection.get("enabled_only", True))
    explicit_names = _selection_items(selection, "checkpoints")

    if explicit_names:
        by_name = {record.name: record for record in records}
        selected: list[CheckpointRecord] = []
        missing: list[str] = []
        for name in explicit_names:
            record = by_name.get(name)
            if record is None:
                missing.append(name)
                continue
            if enabled_only and not record.enabled:
                continue
            selected.append(record)
        if missing:
            raise KeyError(f"Unknown checkpoints in suite selection: {', '.join(missing)}")
        return selected

    return [
        record
        for record in records
        if _matches_selection(record, selection, enabled_only=enabled_only)
    ]
