from __future__ import annotations

import json
import math
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - the project runtime depends on PyYAML.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read audit config files.")
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Audit config must contain a top-level mapping: {path}")
    return data


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain a top-level mapping: {path}")
    return data


def dump_json(data: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def slugify(value: Any) -> str:
    text = str(value).replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-_.").lower()
    return text or "item"


def ratio_tag(value: float | int | str | None) -> str:
    if value is None:
        return "na"
    numeric = float(value)
    scaled = numeric * 100.0 if numeric <= 1.0 else numeric
    return f"r{int(round(scaled)):02d}"


def model_slug(model: str) -> str:
    return slugify(model.split("/")[-1])


def suite_output_name(suite: str) -> str:
    parts = [part for part in suite.replace("\\", "/").split("/") if part]
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        parts.pop()
    return "__".join(parts)


def eval_result_path(output_root: str | Path, suite: str, checkpoint: str) -> Path:
    return Path(output_root) / f"{suite_output_name(suite)}__{checkpoint}.json"


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def format_template(template: str, values: dict[str, Any]) -> str:
    normalized = {key: "" if value is None else value for key, value in values.items()}
    return template.format(**normalized)


def numeric_metric_from_eval_payload(payload: dict[str, Any]) -> float | None:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    for key in ("mean", "accuracy", "score", "value"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    primary_metric = metrics.get("primary_metric")
    if isinstance(primary_metric, str):
        by_metric = metrics.get("by_metric", {})
        if isinstance(by_metric, dict):
            primary_payload = by_metric.get(primary_metric, {})
            if isinstance(primary_payload, dict):
                value = primary_payload.get("mean")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / float(len(values))
    variance = sum((value - avg) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(variance)


def ci95(values: list[float]) -> float | None:
    std = sample_std(values)
    if std is None:
        return None
    return 1.96 * std / math.sqrt(float(len(values)))
