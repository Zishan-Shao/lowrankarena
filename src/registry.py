from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.utils import bool_from_text, ensure_dir, project_path, split_tags


DEFAULT_INDEX_PATH = project_path("checkpoints", "index.csv")
FIELDNAMES = [
    "name",
    "model_family",
    "variant",
    "method",
    "source",
    "repo_id",
    "revision",
    "subpath",
    "benchmarks",
    "enabled",
    "notes",
]


@dataclass(slots=True)
class CheckpointRecord:
    name: str
    model_family: str
    variant: str
    method: str
    source: str
    repo_id: str
    revision: str
    subpath: str
    benchmarks: list[str]
    enabled: bool = True
    notes: str = ""

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "CheckpointRecord":
        return cls(
            name=row["name"].strip(),
            model_family=row.get("model_family", "").strip(),
            variant=row.get("variant", "").strip(),
            method=row.get("method", "").strip(),
            source=row.get("source", "huggingface").strip(),
            repo_id=row.get("repo_id", "").strip(),
            revision=row.get("revision", "main").strip() or "main",
            subpath=row.get("subpath", "").strip(),
            benchmarks=split_tags(row.get("benchmarks", "")),
            enabled=bool_from_text(row.get("enabled", "true")),
            notes=row.get("notes", "").strip(),
        )

    def to_row(self) -> dict[str, str]:
        return {
            "name": self.name,
            "model_family": self.model_family,
            "variant": self.variant,
            "method": self.method,
            "source": self.source,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "subpath": self.subpath,
            "benchmarks": "|".join(self.benchmarks),
            "enabled": "true" if self.enabled else "false",
            "notes": self.notes,
        }


def load_checkpoint_index(path: str | Path = DEFAULT_INDEX_PATH) -> list[CheckpointRecord]:
    index_path = Path(path)
    if not index_path.exists():
        return []
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [CheckpointRecord.from_row(row) for row in reader]


def save_checkpoint_index(
    records: Iterable[CheckpointRecord],
    path: str | Path = DEFAULT_INDEX_PATH,
) -> Path:
    index_path = Path(path)
    ensure_dir(index_path.parent)
    ordered_records = sorted(records, key=lambda record: record.name.lower())
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in ordered_records:
            writer.writerow(record.to_row())
    return index_path


def merge_checkpoint_indexes(paths: Iterable[str | Path]) -> list[CheckpointRecord]:
    merged: dict[str, CheckpointRecord] = {}
    for path in paths:
        for record in load_checkpoint_index(path):
            merged[record.name] = record
    return list(merged.values())


def upsert_checkpoint(
    record: CheckpointRecord,
    path: str | Path = DEFAULT_INDEX_PATH,
) -> Path:
    records = {item.name: item for item in load_checkpoint_index(path)}
    records[record.name] = record
    return save_checkpoint_index(records.values(), path=path)


def get_checkpoint(name: str, path: str | Path = DEFAULT_INDEX_PATH) -> CheckpointRecord:
    for record in load_checkpoint_index(path):
        if record.name == name:
            return record
    raise KeyError(f"Checkpoint '{name}' was not found in {path}.")


def filter_checkpoints(
    records: Iterable[CheckpointRecord],
    benchmark: str | None = None,
    method: str | None = None,
    enabled_only: bool = True,
) -> list[CheckpointRecord]:
    selected: list[CheckpointRecord] = []
    for record in records:
        if enabled_only and not record.enabled:
            continue
        if benchmark and benchmark not in record.benchmarks:
            continue
        if method and record.method != method:
            continue
        selected.append(record)
    return selected
