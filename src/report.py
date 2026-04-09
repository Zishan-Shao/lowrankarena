from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.utils import ensure_dir, load_json, project_path


def discover_results(kind: str, result_dir: str | Path | None = None) -> list[Path]:
    root = Path(result_dir) if result_dir else project_path("results", kind)
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))


def flatten_result(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint": payload.get("checkpoint", ""),
        "suite": payload.get("suite", ""),
        "status": payload.get("status", ""),
    }
    for key, value in payload.get("metrics", {}).items():
        row[f"metric_{key}"] = value
    for key, value in payload.get("stats", {}).items():
        row[f"stat_{key}"] = value
    return row


def load_result_rows(kind: str, result_dir: str | Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover_results(kind, result_dir=result_dir):
        rows.append(flatten_result(load_json(path)))
    return rows


def _stringify(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def make_markdown_table(
    rows: Iterable[dict[str, Any]],
    columns: list[str] | None = None,
) -> str:
    materialized_rows = list(rows)
    if not materialized_rows:
        return "| status |\n| --- |\n| no data |\n"

    resolved_columns = columns or list(materialized_rows[0].keys())
    header = "| " + " | ".join(resolved_columns) + " |"
    separator = "| " + " | ".join("---" for _ in resolved_columns) + " |"
    body = [
        "| " + " | ".join(_stringify(row.get(column, "")) for column in resolved_columns) + " |"
        for row in materialized_rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def write_table(
    name: str,
    markdown: str,
    output_dir: str | Path | None = None,
) -> Path:
    root = Path(output_dir) if output_dir else project_path("results", "tables")
    ensure_dir(root)
    output_path = root / name
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def build_table(
    kind: str,
    columns: list[str] | None = None,
    result_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> Path:
    rows = load_result_rows(kind, result_dir=result_dir)
    markdown = make_markdown_table(rows, columns=columns)
    table_name = filename or f"{kind}.md"
    return write_table(table_name, markdown, output_dir=output_dir)
