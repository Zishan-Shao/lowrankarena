from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.speed import benchmark_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scaffold speed job for one checkpoint.")
    parser.add_argument("checkpoint", help="Checkpoint name from checkpoints/index.csv")
    parser.add_argument("--suite", default="speed", help="Speed suite name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "speed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark_checkpoint(
        checkpoint_name=args.checkpoint,
        suite=args.suite,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir,
        index_path=args.index,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
