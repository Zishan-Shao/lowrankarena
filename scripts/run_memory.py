from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import resolve_suite_path
from src.memory_runner import MemoryRequest, run_memory_measurement
from src.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Transformers memory measurement for one checkpoint."
    )
    parser.add_argument("checkpoint", help="Checkpoint name from checkpoints/index.csv")
    parser.add_argument("--suite", default="memory/active", help="Memory suite name or YAML path")
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prompt-length", type=int, default=None)
    parser.add_argument("--generation-length", type=int, default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verbose-backend", action="store_true")
    parser.add_argument("--run-label", default="ad_hoc")
    parser.add_argument("--strict-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = resolve_suite_path(args.suite)
    suite_config = load_yaml(suite_path)
    if suite_config.get("kind") != "memory":
        raise SystemExit(f"Suite {suite_path} is not a memory suite.")
    memory_config = suite_config.get("memory", {})
    batch_size = args.batch_size if args.batch_size is not None else int(memory_config.get("batch_size", 1))
    prompt_length = args.prompt_length if args.prompt_length is not None else int(memory_config.get("prompt_length", 512))
    generation_length = (
        args.generation_length
        if args.generation_length is not None
        else int(memory_config.get("generation_length", 128))
    )
    result = run_memory_measurement(
        MemoryRequest(
            checkpoint_name=args.checkpoint,
            index_path=args.index,
            suite_path=suite_path,
            suite_name=str(suite_config.get("name", "memory")),
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype or memory_config.get("dtype", "auto"),
            batch_size=batch_size,
            prompt_length=prompt_length,
            generation_length=generation_length,
            attn_implementation=args.attn_implementation or memory_config.get("attn_implementation"),
            local_files_only=args.local_files_only,
            verbose_backend=args.verbose_backend,
            run_label=args.run_label,
            strict_validation=args.strict_validation,
        )
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
