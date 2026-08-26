#!/usr/bin/env python3
"""Recompute the calibration-draw math-retention rankings from HF results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


EXPECTED_BACKEND = "0.4.11"
EXPECTED_DRAWS = {"c4_primary", "wikitext_primary", "wikitext_s0", "wikitext_s1"}
EXPECTED_METHODS = {"asvd", "basis_sharing", "modegpt", "svdllm_v1"}
NUMERIC_FIELDS = ("mathqa", "mmlu_math", "math_retention")


def draw_from_name(name: str) -> str:
    if "-c4-" in name:
        return "c4_primary"
    if "-wikitext-s0-" in name:
        return "wikitext_s0"
    if "-wikitext-s1-" in name:
        return "wikitext_s1"
    if "-wikitext-" in name:
        return "wikitext_primary"
    raise ValueError(f"cannot infer calibration draw from checkpoint name: {name}")


def task_value(document: dict, task: str) -> float:
    try:
        value = float(document["details"]["tasks"][task]["value"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing numeric task value for {task}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite task value for {task}: {value}")
    return value


def load_rows(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob("*.json"))
    if len(paths) != 17:
        raise ValueError(f"expected 17 result JSON files, found {len(paths)} in {results_dir}")

    compressed: list[dict] = []
    dense: tuple[float, float] | None = None
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("status") not in {"ok", "completed"}:
            raise ValueError(f"result is not marked complete: {path.name}")
        if document.get("backend", {}).get("version") != EXPECTED_BACKEND:
            raise ValueError(f"unexpected backend version in {path.name}")
        config = document.get("config", {})
        if config.get("limit") is not None:
            raise ValueError(f"result is not a full-split evaluation: {path.name}")
        if str(config.get("batch_size")) != "1" or config.get("dtype") != "float16":
            raise ValueError(f"unexpected batch size or dtype in {path.name}")

        checkpoint = document.get("checkpoint", {})
        name = str(checkpoint.get("name", ""))
        method = str(checkpoint.get("method", ""))
        mathqa = task_value(document, "lra_mathqa")
        mmlu_math = task_value(document, "mmlu_math")

        if method == "dense":
            if dense is not None:
                raise ValueError("found more than one dense reference")
            dense = (mathqa, mmlu_math)
            continue

        compressed.append(
            {
                "draw": draw_from_name(name),
                "method": method,
                "mathqa": mathqa,
                "mmlu_math": mmlu_math,
            }
        )

    if dense is None or min(dense) <= 0:
        raise ValueError("missing or invalid dense reference")

    found_pairs = {(row["draw"], row["method"]) for row in compressed}
    expected_pairs = {(draw, method) for draw in EXPECTED_DRAWS for method in EXPECTED_METHODS}
    if found_pairs != expected_pairs:
        missing = sorted(expected_pairs - found_pairs)
        extra = sorted(found_pairs - expected_pairs)
        raise ValueError(f"unexpected result matrix; missing={missing}, extra={extra}")

    for row in compressed:
        row["math_retention"] = 0.5 * (
            row["mathqa"] / dense[0] + row["mmlu_math"] / dense[1]
        )

    by_draw: dict[str, list[dict]] = defaultdict(list)
    for row in compressed:
        by_draw[row["draw"]].append(row)
    for draw_rows in by_draw.values():
        draw_rows.sort(key=lambda row: (-row["math_retention"], row["method"]))
        for rank, row in enumerate(draw_rows, start=1):
            row["rank"] = rank

    return sorted(compressed, key=lambda row: (row["draw"], row["rank"]))


def check_expected(rows: list[dict], expected_path: Path) -> None:
    with expected_path.open(newline="", encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    actual_by_key = {(row["draw"], row["method"]): row for row in rows}
    expected_by_key = {(row["draw"], row["method"]): row for row in expected}
    if actual_by_key.keys() != expected_by_key.keys():
        raise ValueError("computed row keys do not match expected.csv")

    for key, actual in actual_by_key.items():
        reference = expected_by_key[key]
        for field in NUMERIC_FIELDS:
            if not math.isclose(actual[field], float(reference[field]), abs_tol=1e-9):
                raise ValueError(
                    f"{key} {field}: computed {actual[field]:.10f}, "
                    f"expected {float(reference[field]):.10f}"
                )
        if actual["rank"] != int(reference["rank"]):
            raise ValueError(f"{key} rank mismatch")


def print_summary(rows: list[dict]) -> None:
    print("draw | rank 1 | rank 2 | rank 3 | rank 4")
    print("--- | --- | --- | --- | ---")
    for draw in sorted(EXPECTED_DRAWS):
        names = [row["method"] for row in rows if row["draw"] == draw]
        print(f"{draw} | " + " | ".join(names))

    print("\nmethod | min | max | range | ranks | max displacement")
    print("--- | ---: | ---: | ---: | --- | ---:")
    max_displacement = 0
    for method in sorted(EXPECTED_METHODS):
        method_rows = [row for row in rows if row["method"] == method]
        values = [row["math_retention"] for row in method_rows]
        ranks = [row["rank"] for row in method_rows]
        displacement = max(ranks) - min(ranks)
        max_displacement = max(max_displacement, displacement)
        print(
            f"{method} | {min(values):.6f} | {max(values):.6f} | "
            f"{max(values) - min(values):.6f} | "
            f"{','.join(map(str, ranks))} | {displacement}"
        )
    print(f"\nmaximum rank displacement: {max_displacement}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="downloaded HF results directory")
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path(__file__).with_name("expected.csv"),
        help="reference CSV to check (default: expected.csv beside this script)",
    )
    args = parser.parse_args()
    rows = load_rows(args.results_dir)
    check_expected(rows, args.expected)
    print_summary(rows)
    print("\nPASS: all 16 compressed results match expected.csv")


if __name__ == "__main__":
    main()
