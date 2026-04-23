from __future__ import annotations

import ast
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lowrankarena" / "datasets" / "mathqa"
SPLIT_FILES = {
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json",
}
CHOICE_LABELS = ["a", "b", "c", "d", "e"]


def _download_once(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    tmp_path.replace(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_zip(url: str, sha256: str, cache_dir: str | None = None) -> Path:
    root = Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR
    archive_path = root / "MathQA.zip"
    if not archive_path.exists():
        _download_once(url, archive_path)

    actual_sha256 = _sha256(archive_path)
    if actual_sha256 != sha256:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MathQA archive checksum mismatch: expected {sha256}, got {actual_sha256}."
        )
    return archive_path


def load_mathqa_dataset(
    url: str,
    sha256: str,
    cache_dir: str | None = None,
    **_: Any,
) -> DatasetDict:
    archive_path = _resolve_zip(url=url, sha256=sha256, cache_dir=cache_dir)
    datasets = {}
    with zipfile.ZipFile(archive_path) as archive:
        for split, filename in SPLIT_FILES.items():
            rows = json.loads(archive.read(filename).decode("utf-8"))
            datasets[split] = Dataset.from_list(rows)
    return DatasetDict(datasets)


def doc_to_choice(doc: dict[str, Any]) -> list[str]:
    raw_options = str(doc["options"])
    option_items: list[str]
    try:
        parsed_options = ast.literal_eval(raw_options)
    except (SyntaxError, ValueError):
        parsed_options = None
    if isinstance(parsed_options, list):
        option_items = [str(item) for item in parsed_options]
    else:
        option_items = [raw_options]

    matches = []
    for item in option_items:
        matches.extend(re.findall(r"([a-e])\s*\)\s*(.*?)(?=,\s*[a-e]\s*\)|$)", item))
    by_label = {label: value.strip().rstrip(" ,") for label, value in matches}
    choices = [by_label[label] for label in CHOICE_LABELS if label in by_label]
    if len(choices) != len(CHOICE_LABELS):
        raise ValueError(f"Expected five MathQA choices, got {len(choices)} from {raw_options!r}.")
    return choices


def doc_to_target(doc: dict[str, Any]) -> int:
    correct = str(doc["correct"]).strip().lower()
    if correct not in CHOICE_LABELS:
        raise ValueError(f"Unexpected MathQA answer label: {correct!r}.")
    return CHOICE_LABELS.index(correct)
