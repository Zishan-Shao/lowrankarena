from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import resolve_suite_path
from src.inference_adapter import prepare_model_for_inference
from src.lm_eval_runner import LmEvalRequest, run_lm_eval_suite
from src.load import load_checkpoint
from src.memory_runner import MemoryRequest, run_memory_measurement
from src.speed_runner import VllmSpeedRequest, run_vllm_speed_suite


@dataclass(slots=True)
class SmokeStepResult:
    name: str
    status: str
    duration_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SmokeRunResult:
    checkpoint_name: str
    status: str
    steps: list[SmokeStepResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small end-to-end smoke pass over checkpoint loading plus "
            "the eval, memory, and speed runners."
        )
    )
    parser.add_argument("checkpoint", help="Checkpoint name from checkpoints/index.csv")
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--device", default="cuda:0", help="CUDA device for eval and memory.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verbose-backend", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed step when possible.")

    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")

    parser.add_argument("--eval-suite", default="accuracy/mcq")
    parser.add_argument("--eval-limit", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", default="1")
    parser.add_argument("--eval-num-fewshot", type=int, default=0)
    parser.add_argument("--lm-eval-bin", default="lm-eval")

    parser.add_argument("--memory-dtype", default="float16")
    parser.add_argument("--memory-batch-size", type=int, default=1)
    parser.add_argument("--memory-prompt-length", type=int, default=32)
    parser.add_argument("--memory-generation-length", type=int, default=8)
    parser.add_argument("--memory-attn-implementation", default=None)

    parser.add_argument("--speed-suite", default="speed/speed")
    parser.add_argument("--speed-batch-size", type=int, default=1)
    parser.add_argument("--speed-prompt-length", type=int, default=32)
    parser.add_argument("--speed-generation-length", type=int, default=8)
    parser.add_argument("--speed-repeat", type=int, default=1)
    parser.add_argument("--speed-warmup", type=int, default=0)
    parser.add_argument("--speed-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--speed-gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--speed-dtype", default="auto")
    parser.add_argument("--speed-max-model-len", type=int, default=2048)
    parser.add_argument("--no-enforce-eager", action="store_true")
    return parser.parse_args()


def _print(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


def _cleanup_cuda() -> None:
    try:
        import torch
    except ImportError:
        return

    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()


def _configure_runtime_environment(*, verbose_backend: bool) -> None:
    if verbose_backend:
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "CRITICAL")
    os.environ.setdefault("LMEVAL_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TQDM_DISABLE", "1")


def _run_load_smoke(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_checkpoint(
        args.checkpoint,
        index_path=args.index,
        download=True,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    prepared = prepare_model_for_inference(loaded)
    model_path = prepared.model_path
    if model_path.startswith("hf://"):
        raise RuntimeError("Expected a local checkpoint path after materialization.")

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer_mode = "auto"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if isinstance(tokenizer, bool):
            raise TypeError("AutoTokenizer returned a boolean sentinel instead of a tokenizer instance.")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        tokenizer_mode = "slow"

    return {
        "model_path": model_path,
        "tokenizer_path": prepared.tokenizer_path,
        "tokenizer_mode_prepared": prepared.tokenizer_mode,
        "preparation_kind": prepared.preparation_kind,
        "source_model_path": prepared.source_model_path,
        "locator": loaded.locator,
        "loader": loaded.loader,
        "model_type": getattr(config, "model_type", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_mode": tokenizer_mode,
        "snapshot_path": loaded.metadata.get("snapshot_path"),
    }


def _run_eval_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.eval_limit <= 1:
        _print(
            "Eval smoke is running with --eval-limit=1. "
            "Treat the reported task scores as smoke-only, not as benchmark-quality metrics."
        )
    suite_path = resolve_suite_path(args.eval_suite)
    result = run_lm_eval_suite(
        LmEvalRequest(
            checkpoint_name=args.checkpoint,
            suite_path=suite_path,
            index_path=args.index,
            output_dir=str(ROOT / "results" / "eval"),
            raw_output_root=str(ROOT / "results" / "eval" / "raw"),
            lm_eval_bin=args.lm_eval_bin,
            model_backend="hf",
            device=args.device,
            batch_size=args.eval_batch_size,
            limit=args.eval_limit,
            num_fewshot=args.eval_num_fewshot,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
    )
    return {
        "suite": result.suite,
        "output_path": result.output_path,
        "raw_output_path": result.raw_output_path,
        "metrics": result.metrics,
        "limit": args.eval_limit,
    }


def _run_memory_smoke(args: argparse.Namespace) -> dict[str, Any]:
    result = run_memory_measurement(
        MemoryRequest(
            checkpoint_name=args.checkpoint,
            index_path=args.index,
            output_dir=str(ROOT / "results" / "memory"),
            device=args.device,
            dtype=args.memory_dtype,
            batch_size=args.memory_batch_size,
            prompt_length=args.memory_prompt_length,
            generation_length=args.memory_generation_length,
            attn_implementation=args.memory_attn_implementation,
            local_files_only=args.local_files_only,
            verbose_backend=args.verbose_backend,
        )
    )
    return {
        "suite": result.suite,
        "output_path": result.output_path,
        "metrics": result.metrics,
    }


def _run_speed_smoke(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = resolve_suite_path(args.speed_suite)
    result = run_vllm_speed_suite(
        VllmSpeedRequest(
            checkpoint_name=args.checkpoint,
            suite_path=suite_path,
            index_path=args.index,
            output_dir=str(ROOT / "results" / "speed"),
            batch_sizes=[args.speed_batch_size],
            prompt_lengths=[args.speed_prompt_length],
            generation_lengths=[args.speed_generation_length],
            repeat=args.speed_repeat,
            warmup=args.speed_warmup,
            tensor_parallel_size=args.speed_tensor_parallel_size,
            gpu_memory_utilization=args.speed_gpu_memory_utilization,
            dtype=args.speed_dtype,
            max_model_len=args.speed_max_model_len,
            enforce_eager=not args.no_enforce_eager,
            local_files_only=args.local_files_only,
            verbose_backend=args.verbose_backend,
            show_progress=not args.verbose_backend,
        )
    )
    return {
        "suite": result.suite,
        "output_path": result.output_path,
        "metrics": result.stats,
    }


def _run_step(
    name: str,
    fn,
    *,
    keep_going: bool,
) -> SmokeStepResult:
    _print(f"Starting {name}")
    started = time.perf_counter()
    try:
        details = fn()
    except Exception as exc:
        duration = time.perf_counter() - started
        _print(f"{name} failed after {duration:.2f}s: {type(exc).__name__}: {exc}")
        if not keep_going:
            raise
        return SmokeStepResult(
            name=name,
            status="failed",
            duration_seconds=duration,
            details={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )

    duration = time.perf_counter() - started
    _print(f"{name} completed in {duration:.2f}s")
    return SmokeStepResult(
        name=name,
        status="completed",
        duration_seconds=duration,
        details=details,
    )


def main() -> None:
    args = parse_args()
    _configure_runtime_environment(verbose_backend=args.verbose_backend)
    steps: list[SmokeStepResult] = []

    load_step = _run_step(
        "checkpoint load smoke",
        lambda: _run_load_smoke(args),
        keep_going=False,
    )
    steps.append(load_step)
    _cleanup_cuda()

    runnable_steps: list[tuple[str, Any]] = []
    if not args.skip_eval:
        runnable_steps.append(("eval smoke", lambda: _run_eval_smoke(args)))
    if not args.skip_memory:
        runnable_steps.append(("memory smoke", lambda: _run_memory_smoke(args)))
    if not args.skip_speed:
        runnable_steps.append(("speed smoke", lambda: _run_speed_smoke(args)))

    for name, fn in runnable_steps:
        step = _run_step(name, fn, keep_going=args.keep_going)
        steps.append(step)
        _cleanup_cuda()
        if step.status != "completed" and not args.keep_going:
            break

    overall_status = "completed" if all(step.status == "completed" for step in steps) else "failed"
    summary = SmokeRunResult(
        checkpoint_name=args.checkpoint,
        status=overall_status,
        steps=steps,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    if overall_status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
