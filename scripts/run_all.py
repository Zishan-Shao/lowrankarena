from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import evaluate_checkpoint
from src.registry import filter_checkpoints, load_checkpoint_index
from src.speed import benchmark_checkpoint
from src.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaffold benchmark suites from benchmark/*.yaml.")
    parser.add_argument(
        "--suites",
        nargs="*",
        default=["main", "speed", "modern", "pruning", "quant"],
        help="Benchmark suite names without the .yaml suffix",
    )
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    return parser.parse_args()


def run_suite(suite_name: str, index_path: str) -> list[dict[str, str]]:
    config_path = ROOT / "benchmark" / f"{suite_name}.yaml"
    config = load_yaml(config_path)
    records = load_checkpoint_index(index_path)
    selection = config.get("selection", {})
    enabled_only = bool(selection.get("enabled_only", True))
    benchmarks = selection.get("benchmarks", [suite_name])

    selected = [
        record
        for benchmark in benchmarks
        for record in filter_checkpoints(records, benchmark=benchmark, enabled_only=enabled_only)
    ]
    deduped = {record.name: record for record in selected}

    outputs: list[dict[str, str]] = []
    kind = config.get("kind", "eval")
    for record in deduped.values():
        if kind == "speed":
            speed_config = config.get("speed", {})
            result = benchmark_checkpoint(
                checkpoint_name=record.name,
                suite=suite_name,
                batch_size=int(speed_config.get("batch_sizes", [1])[0]),
                sequence_length=int(speed_config.get("sequence_lengths", [2048])[0]),
                index_path=index_path,
            )
        else:
            eval_config = config.get("eval", {})
            result = evaluate_checkpoint(
                checkpoint_name=record.name,
                suite=suite_name,
                dataset=str(eval_config.get("dataset", "placeholder")),
                index_path=index_path,
            )
        outputs.append(
            {
                "suite": suite_name,
                "checkpoint": record.name,
                "status": result.status,
                "output_path": result.output_path,
            }
        )
    return outputs


def main() -> None:
    args = parse_args()
    summary: list[dict[str, str]] = []
    for suite_name in args.suites:
        summary.extend(run_suite(suite_name, index_path=args.index))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
