from __future__ import annotations

import argparse
import time

from external_vllm import import_installed_vllm
from terminal_ui import ProgressPrinter, configure_runtime_environment, print_failure, use_safe_vllm_cwd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test vLLM on an SVD-LLM wrapper model.")
    parser.add_argument("--model", required=True, help="Path to the local wrapper model directory.")
    parser.add_argument("--prompt", default="Hello, my name is", help="Prompt text.")
    parser.add_argument("--max-tokens", type=int, default=8, help="Number of generated tokens.")
    parser.add_argument("--max-model-len", type=int, default=2048, help="vLLM max model len.")
    parser.add_argument("--dtype", default="auto", help="vLLM dtype.")
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
    parser.add_argument(
        "--tokenizer-mode",
        default="slow",
        help="Tokenizer mode. The original checkpoint needs 'slow'.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable cudagraph/compile for easier debugging.",
    )
    parser.add_argument(
        "--verbose-vllm",
        action="store_true",
        help="Show raw vLLM logs instead of the compact wrapper progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime_environment(verbose_vllm=args.verbose_vllm)

    installed_vllm = import_installed_vllm()
    LLM = installed_vllm.LLM
    SamplingParams = installed_vllm.SamplingParams

    started = time.time()
    progress = ProgressPrinter(total_steps=4, enabled=not args.verbose_vllm)

    progress.step(1, "Preparing vLLM smoke test")
    progress.detail(f"model={args.model}")
    progress.detail(f"tokenizer_mode={args.tokenizer_mode}")
    progress.detail(f"dtype={args.dtype}")

    try:
        with use_safe_vllm_cwd():
            with progress.waiting(2, "Initializing vLLM engine"):
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
    print(f"[ok] llm initialized in {time.time() - started:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    progress.step(3, "Running generation")
    try:
        outputs = llm.generate([args.prompt], sampling_params, use_tqdm=False)
    except Exception as exc:
        if args.verbose_vllm:
            raise
        print_failure(f"generation failed: {exc}")
        raise SystemExit(1) from None
    progress.step(4, "Generation completed")
    print(f"[ok] generation finished in {time.time() - started:.2f}s")
    print("prompt:", outputs[0].prompt)
    print("text:", outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
