#!/usr/bin/env python3
"""Extract and verify the Llama-3.1-8B MoDeGPT BoolQ floor results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RESULT_REVISION = "db12fe2017e9075d5c0c46f80b6ce2c9ccb431dd"
RANDOM_BASELINE = 0.5
BOOLQ_N = 3270
MAJORITY_BASELINE = 2033 / BOOLQ_N
NUMERIC_FIELDS = (
    "keep_ratio",
    "mcq_macro",
    "boolq_acc",
    "boolq_stderr",
    "random_baseline",
    "majority_baseline",
)


def keep_ratio(checkpoint: dict) -> float:
    method = checkpoint.get("method")
    if method == "dense":
        return 1.0
    name = str(checkpoint.get("name", ""))
    try:
        return float(name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"cannot infer keep ratio from {name}") from error


def load_rows(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob("llama31-8b-*.json"))
    wanted_names = {"llama31-8b-dense", *(f"llama31-8b-modegpt-0.{i}" for i in range(4, 9))}
    rows = []
    found_names = set()
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = document.get("checkpoint", {})
        name = str(checkpoint.get("name", ""))
        if name not in wanted_names:
            continue
        found_names.add(name)
        if document.get("status") not in {"ok", "completed"}:
            raise ValueError(f"result is not marked complete: {path.name}")
        if document.get("backend", {}).get("version") != "0.4.11":
            raise ValueError(f"unexpected backend version: {path.name}")
        config = document.get("config", {})
        if config.get("limit") is not None:
            raise ValueError(f"result is not a full-split evaluation: {path.name}")
        if str(config.get("batch_size")) != "1" or config.get("dtype") != "float16":
            raise ValueError(f"unexpected batch size or dtype: {path.name}")
        try:
            boolq = document["details"]["tasks"]["boolq"]
            n_samples = document["runtime"]["n_samples"]["boolq"]["effective"]
            row = {
                "model": "Llama-3.1-8B",
                "method": "dense" if checkpoint.get("method") == "dense" else "MoDeGPT",
                "keep_ratio": keep_ratio(checkpoint),
                "mcq_macro": float(document["metrics"]["mean"]),
                "boolq_acc": float(boolq["value"]),
                "boolq_stderr": float(boolq["stderr"]),
                "boolq_n": int(n_samples),
                "random_baseline": RANDOM_BASELINE,
                "majority_baseline": MAJORITY_BASELINE,
                "result_revision": RESULT_REVISION,
                "result_path": f"results/evaluation/llama31_8b/mcq/{path.name}",
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed result: {path.name}") from error
        if row["boolq_n"] != BOOLQ_N:
            raise ValueError(f"expected full BoolQ split ({BOOLQ_N}): {path.name}")
        rows.append(row)

    if found_names != wanted_names:
        raise ValueError(f"missing results: {sorted(wanted_names - found_names)}")
    return sorted(rows, key=lambda row: row["keep_ratio"] if row["method"] != "dense" else -1)


def check_expected(rows: list[dict], expected_path: Path) -> None:
    with expected_path.open(newline="", encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    if len(rows) != len(expected):
        raise ValueError("row count does not match expected.csv")
    for actual, reference in zip(rows, expected):
        for field in NUMERIC_FIELDS:
            if not math.isclose(float(actual[field]), float(reference[field]), abs_tol=1e-9):
                raise ValueError(f"{actual['method']} keep={actual['keep_ratio']} {field} mismatch")
        for field in ("model", "method", "result_revision", "result_path"):
            if str(actual[field]) != reference[field]:
                raise ValueError(f"{actual['method']} keep={actual['keep_ratio']} {field} mismatch")
        if int(actual["boolq_n"]) != int(reference["boolq_n"]):
            raise ValueError("BoolQ sample count mismatch")


def interpretation(value: float) -> str:
    if math.isclose(value, MAJORITY_BASELINE, abs_tol=1e-10):
        return "at majority"
    if value < RANDOM_BASELINE:
        return "below random"
    if value < MAJORITY_BASELINE:
        return "between random and majority"
    return "above majority"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="downloaded normalized MCQ directory")
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path(__file__).with_name("expected.csv"),
        help="reference CSV to check (default: expected.csv beside this script)",
    )
    args = parser.parse_args()
    rows = load_rows(args.results_dir)
    check_expected(rows, args.expected)

    print("method | keep | MCQ macro | BoolQ | stderr | n | interpretation")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---")
    for row in rows:
        print(
            f"{row['method']} | {row['keep_ratio']:.1f} | {row['mcq_macro']:.6f} | "
            f"{row['boolq_acc']:.6f} | {row['boolq_stderr']:.6f} | "
            f"{row['boolq_n']} | {interpretation(row['boolq_acc'])}"
        )
    print(
        f"\nreferences: random={RANDOM_BASELINE:.10f}, "
        f"majority={MAJORITY_BASELINE:.10f} ({2033}/{BOOLQ_N})"
    )
    print("PASS: all six published results match expected.csv")


if __name__ == "__main__":
    main()
