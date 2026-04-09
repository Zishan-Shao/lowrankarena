from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scaffold eval job for one checkpoint.")
    parser.add_argument("checkpoint", help="Checkpoint name from checkpoints/index.csv")
    parser.add_argument("--suite", default="main", help="Benchmark suite name")
    parser.add_argument("--dataset", default="placeholder", help="Dataset or task-set identifier")
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        checkpoint_name=args.checkpoint,
        suite=args.suite,
        dataset=args.dataset,
        output_dir=args.output_dir,
        index_path=args.index,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
