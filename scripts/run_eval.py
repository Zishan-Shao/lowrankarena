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
from src.lm_eval_runner import LmEvalRequest, run_lm_eval_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one eval suite for one checkpoint.")
    parser.add_argument("checkpoint", help="Checkpoint name from checkpoints/index.csv")
    parser.add_argument("--suite", default="mcq", help="Benchmark suite name or YAML path")
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--raw-output-dir", default=None)
    parser.add_argument("--lm-eval-bin", default="lm-eval", help="lm-eval executable path for suites that use the lm_eval_harness backend")
    parser.add_argument(
        "--model-backend",
        default=None,
        help="Override suite model backend, e.g. hf or vllm. If omitted, uses eval.model_backend from YAML or hf.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None, help="Override suite dtype, e.g. auto, float16, fp16, bfloat16, or bf16.")
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="vLLM tensor_parallel_size override.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM gpu_memory_utilization override.")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM max_model_len override.")
    parser.add_argument("--enforce-eager", action="store_true", help="Pass enforce_eager=True to vLLM.")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--run-label", default="ad_hoc")
    parser.add_argument("--strict-validation", action="store_true")
    parser.add_argument("--model-arg", action="append", default=[], help="Extra lm-eval model_arg entries like key=value")
    return parser.parse_args()


def parse_model_args(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid --model-arg value: {item}")
        parsed[key] = value
    return parsed


def main() -> None:
    args = parse_args()
    suite_path = resolve_suite_path(args.suite)
    extra_model_args = parse_model_args(args.model_arg)
    if args.dtype is not None:
        extra_model_args["dtype"] = args.dtype
    result = run_lm_eval_suite(
        LmEvalRequest(
            checkpoint_name=args.checkpoint,
            suite_path=suite_path,
            index_path=args.index,
            output_dir=args.output_dir,
            raw_output_root=args.raw_output_dir,
            lm_eval_bin=args.lm_eval_bin,
            model_backend=args.model_backend,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            num_fewshot=args.num_fewshot,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=True if args.enforce_eager else None,
            log_samples=args.log_samples,
            extra_model_args=extra_model_args,
            run_label=args.run_label,
            strict_validation=args.strict_validation,
        )
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
