from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry import CheckpointRecord, load_checkpoint_index, save_checkpoint_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append or replace a checkpoint row in checkpoints/index.csv.")
    parser.add_argument("name")
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--variant", default="unknown")
    parser.add_argument("--method", default="pending")
    parser.add_argument("--source", default="huggingface")
    parser.add_argument("--repo-id", default="Duke-CEI-SVD/LowRankArena")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--subpath", required=True)
    parser.add_argument("--benchmarks", nargs="*", default=["main"])
    parser.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [record for record in load_checkpoint_index(args.index) if record.name != args.name]
    records.append(
        CheckpointRecord(
            name=args.name,
            model_family=args.model_family,
            variant=args.variant,
            method=args.method,
            source=args.source,
            repo_id=args.repo_id,
            revision=args.revision,
            subpath=args.subpath,
            benchmarks=list(args.benchmarks),
            enabled=args.enabled,
            notes=args.notes,
        )
    )
    save_checkpoint_index(records, args.index)
    print(args.index)


if __name__ == "__main__":
    main()
