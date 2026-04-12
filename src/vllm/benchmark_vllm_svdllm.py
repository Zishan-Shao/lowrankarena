from __future__ import annotations

import argparse
import json
import math
import statistics
import time

from external_vllm import import_installed_vllm
from terminal_ui import ProgressPrinter, configure_runtime_environment, print_failure, use_safe_vllm_cwd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple vLLM speed benchmark for an SVD-LLM wrapper model.")
    parser.add_argument("--model", required=True, help="Path to the local wrapper model directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--prompt-length", type=int, default=512, help="Prompt length in tokens.")
    parser.add_argument("--generation-length", type=int, default=128, help="Generated length in tokens.")
    parser.add_argument("--repeat", type=int, default=5, help="Number of timed runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs.")
    parser.add_argument("--max-model-len", type=int, default=2048, help="vLLM max model len.")
    parser.add_argument("--dtype", default="float16", help="vLLM dtype.")
    parser.add_argument("--tokenizer-mode", default="slow", help="Tokenizer mode.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.4,
        help="vLLM gpu memory utilization.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Disable compile/cudagraph.")
    parser.add_argument(
        "--verbose-vllm",
        action="store_true",
        help="Show raw vLLM logs instead of the compact wrapper progress output.",
    )
    return parser.parse_args()


def build_prompt_token_ids(tokenizer, prompt_length: int) -> list[int]:
    seed_text = "The quick brown fox jumps over the lazy dog. "
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("Tokenizer produced no seed token ids.")
    repeats = math.ceil(prompt_length / len(seed_ids))
    return (seed_ids * repeats)[:prompt_length]


def main() -> None:
    args = parse_args()
    configure_runtime_environment(verbose_vllm=args.verbose_vllm)

    from transformers import AutoTokenizer

    installed_vllm = import_installed_vllm()
    LLM = installed_vllm.LLM
    SamplingParams = installed_vllm.SamplingParams

    progress = ProgressPrinter(total_steps=4, enabled=not args.verbose_vllm)

    progress.step(1, "Preparing benchmark")
    progress.detail(f"model={args.model}")
    progress.detail(
        f"batch={args.batch_size}, prompt={args.prompt_length}, generation={args.generation_length}, repeat={args.repeat}, warmup={args.warmup}"
    )

    progress.step(2, "Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )
    try:
        with use_safe_vllm_cwd():
            with progress.waiting(3, "Initializing vLLM engine"):
                llm = LLM(
                    model=args.model,
                    trust_remote_code=True,
                    model_impl="transformers",
                    tokenizer_mode=args.tokenizer_mode,
                    tensor_parallel_size=args.tensor_parallel_size,
                    dtype=args.dtype,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    enforce_eager=args.enforce_eager,
                    use_tqdm_on_load=False,
                )
    except Exception as exc:
        text = str(exc)
        if "Free memory on device" in text or "GPU memory utilization" in text:
            print_failure(
                "vLLM init failed because the selected GPU does not have enough free memory. "
                "Try a less busy GPU or pass a smaller --gpu-memory-utilization such as 0.3."
            )
            raise SystemExit(1) from None
        if args.verbose_vllm:
            raise
        print_failure(f"vLLM init failed: {exc}")
        raise SystemExit(1) from None

    prompt_token_ids = build_prompt_token_ids(tokenizer, args.prompt_length)
    prompts = [prompt_token_ids] * args.batch_size
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.generation_length,
        ignore_eos=True,
    )

    progress.step(4, "Running warmup and timed benchmark")
    for warmup_index in range(args.warmup):
        progress.detail(f"warmup {warmup_index + 1}/{args.warmup}")
        try:
            llm.generate(prompts, sampling_params, use_tqdm=False)
        except Exception as exc:
            if args.verbose_vllm:
                raise
            print_failure(f"warmup failed: {exc}")
            raise SystemExit(1) from None

    runs: list[dict[str, float]] = []
    for run_index in range(args.repeat):
        progress.detail(f"timed run {run_index + 1}/{args.repeat}")
        started = time.perf_counter()
        try:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        except Exception as exc:
            if args.verbose_vllm:
                raise
            print_failure(f"timed run failed: {exc}")
            raise SystemExit(1) from None
        elapsed = time.perf_counter() - started
        prompt_tokens = sum(len(output.prompt_token_ids or []) for output in outputs)
        generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        runs.append(
            {
                "latency_seconds": elapsed,
                "prompt_tokens": float(prompt_tokens),
                "generated_tokens": float(generated_tokens),
                "prefill_tokens_per_second": float(prompt_tokens) / elapsed if elapsed else 0.0,
                "decode_tokens_per_second": float(generated_tokens) / elapsed if elapsed else 0.0,
                "end_to_end_tokens_per_second": float(prompt_tokens + generated_tokens) / elapsed if elapsed else 0.0,
            }
        )

    summary = {
        "model": args.model,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "generation_length": args.generation_length,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "mean_latency_seconds": statistics.mean(item["latency_seconds"] for item in runs),
        "mean_prefill_tokens_per_second": statistics.mean(item["prefill_tokens_per_second"] for item in runs),
        "mean_decode_tokens_per_second": statistics.mean(item["decode_tokens_per_second"] for item in runs),
        "mean_end_to_end_tokens_per_second": statistics.mean(item["end_to_end_tokens_per_second"] for item in runs),
        "runs": runs,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
