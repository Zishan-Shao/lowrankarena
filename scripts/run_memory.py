from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory_runner import MemoryRequest, run_memory_measurement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Transformers memory measurement for one checkpoint."
    )
    parser.add_argument("checkpoint", help="Checkpoint name from checkpoints/index.csv")
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--generation-length", type=int, default=8)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verbose-backend", action="store_true")
    parser.add_argument("--run-label", default="ad_hoc")
    parser.add_argument("--strict-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_memory_measurement(
        MemoryRequest(
            checkpoint_name=args.checkpoint,
            index_path=args.index,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            generation_length=args.generation_length,
            attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
            verbose_backend=args.verbose_backend,
            run_label=args.run_label,
            strict_validation=args.strict_validation,
        )
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
