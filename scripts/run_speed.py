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
from src.speed_runner import VllmSpeedRequest, run_vllm_speed_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one vLLM speed suite for one checkpoint.")
    parser.add_argument("checkpoint", help="Checkpoint name from checkpoints/index.csv")
    parser.add_argument("--suite", default="speed/speed", help="Speed suite name or YAML path")
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, action="append", default=None)
    parser.add_argument("--prompt-length", type=int, action="append", default=None)
    parser.add_argument("--generation-length", type=int, action="append", default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--verbose-backend", action="store_true", help="Show raw vLLM logs.")
    parser.add_argument("--run-label", default="ad_hoc")
    parser.add_argument("--strict-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = resolve_suite_path(args.suite)
    result = run_vllm_speed_suite(
        VllmSpeedRequest(
            checkpoint_name=args.checkpoint,
            suite_path=suite_path,
            index_path=args.index,
            output_dir=args.output_dir,
            batch_sizes=args.batch_size,
            prompt_lengths=args.prompt_length,
            generation_lengths=args.generation_length,
            repeat=args.repeat,
            warmup=args.warmup,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            enforce_eager=True if args.enforce_eager else None,
            verbose_backend=args.verbose_backend,
            show_progress=True,
            run_label=args.run_label,
            strict_validation=args.strict_validation,
        )
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
