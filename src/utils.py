from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional until dependencies are installed.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def user_cache_path(*parts: str) -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
    return cache_root.joinpath("lowrankarena", *parts)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read benchmark config files.")
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at the top level: {path}")
    return data


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain a mapping at the top level: {path}")
    return data


def dump_json(data: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_tags(raw: str) -> list[str]:
    separators_normalized = raw.replace(",", "|")
    return [part.strip() for part in separators_normalized.split("|") if part.strip()]


def bool_from_text(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
