from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.benchmarking import suite_id, suite_output_name
from src.load import load_checkpoint
from src.scoring import summarize_speed_cases
from src.utils import dump_json, ensure_dir, load_yaml, utc_timestamp
from vllm.terminal_ui import ProgressPrinter, configure_runtime_environment, print_failure
from vllm.vllm_adapter import DEFAULT_WRAPPER_CACHE_ROOT, PreparedVllmModel, prepare_model_for_vllm


VLLM_TRY_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = VLLM_TRY_ROOT / "checkpoints" / "index.csv"
DEFAULT_OUTPUT_DIR = VLLM_TRY_ROOT / "results" / "speed"
DEFAULT_SUITE_PATH = REPO_ROOT / "benchmark" / "speed" / "speed.yaml"


@dataclass(slots=True)
class VllmSpeedRequest:
    checkpoint_name: str
    suite_path: str | Path = DEFAULT_SUITE_PATH
    index_path: str | Path = DEFAULT_INDEX_PATH
    output_dir: str | Path | None = None
    wrapper_cache_root: str | Path | None = None
    batch_sizes: list[int] | None = None
    prompt_lengths: list[int] | None = None
    generation_lengths: list[int] | None = None
    repeat: int | None = None
    warmup: int | None = None
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    max_model_len: int | None = None
    enforce_eager: bool | None = None
    trust_remote_code: bool = True
    local_files_only: bool = False
    verbose_vllm: bool = False


@dataclass(slots=True)
class VllmSpeedResult:
    checkpoint_name: str
    suite: str
    status: str
    output_path: str
    stats: dict[str, Any]
    prepared_model: PreparedVllmModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype speed runner that prepares a checkpoint for vLLM before constructing LLM(...)."
    )
    parser.add_argument("--checkpoint-name", required=True, help="Checkpoint name in the provided index CSV.")
    parser.add_argument(
        "--suite-path",
        default=str(DEFAULT_SUITE_PATH),
        help="Speed suite YAML. Defaults to the repo benchmark/speed/speed.yaml.",
    )
    parser.add_argument(
        "--index-path",
        default=str(DEFAULT_INDEX_PATH),
        help="Checkpoint index CSV. Defaults to the vllm_try demo index.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write the speed result JSON.",
    )
    parser.add_argument(
        "--wrapper-cache-root",
        default=str(DEFAULT_WRAPPER_CACHE_ROOT),
        help="Where to materialize cached vLLM wrapper models.",
    )
    parser.add_argument("--batch-size", dest="batch_sizes", action="append", type=int, help="Override suite batch size.")
    parser.add_argument(
        "--prompt-length",
        dest="prompt_lengths",
        action="append",
        type=int,
        help="Override suite prompt length.",
    )
    parser.add_argument(
        "--generation-length",
        dest="generation_lengths",
        action="append",
        type=int,
        help="Override suite generation length.",
    )
    parser.add_argument("--repeat", type=int, help="Override suite repeat count.")
    parser.add_argument("--warmup", type=int, help="Override suite warmup count.")
    parser.add_argument("--tensor-parallel-size", type=int, help="Override tensor parallel size.")
    parser.add_argument("--gpu-memory-utilization", type=float, help="Override GPU memory utilization.")
    parser.add_argument("--dtype", help="Override vLLM dtype.")
    parser.add_argument("--max-model-len", type=int, help="Override max model len.")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=None,
        help="Disable compile/cudagraph for easier debugging.",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Do not fetch missing HF files from the network.")
    parser.add_argument(
        "--verbose-vllm",
        action="store_true",
        help="Show raw vLLM logs instead of the compact prototype progress output.",
    )
    return parser.parse_args()


def _build_prompt_token_ids(tokenizer: Any, prompt_length: int) -> list[int]:
    seed_text = "The quick brown fox jumps over the lazy dog. "
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("Tokenizer produced no seed token ids.")
    repeats = math.ceil(prompt_length / len(seed_ids))
    return (seed_ids * repeats)[:prompt_length]


def _safe_suite_output_name(config_path: str | Path) -> str:
    try:
        return suite_output_name(config_path)
    except Exception:
        return Path(config_path).resolve().with_suffix("").name


def _safe_suite_id(config_path: str | Path) -> str:
    try:
        return suite_id(config_path)
    except Exception:
        return str(Path(config_path).resolve())


def _result_path_for(request: VllmSpeedRequest) -> Path:
    output_root = Path(request.output_dir) if request.output_dir else DEFAULT_OUTPUT_DIR
    ensure_dir(output_root)
    suite_name = _safe_suite_output_name(request.suite_path)
    return output_root / f"{suite_name}__{request.checkpoint_name}.json"


def _bool_from_optional_flag(value: bool | None, fallback: bool) -> bool:
    return fallback if value is None else bool(value)


def run_vllm_speed_suite(request: VllmSpeedRequest) -> VllmSpeedResult:
    configure_runtime_environment(verbose_vllm=request.verbose_vllm)
    progress = ProgressPrinter(total_steps=6, enabled=not request.verbose_vllm)

    suite_path = Path(request.suite_path).expanduser().resolve()
    suite_config = load_yaml(suite_path)
    if suite_config.get("kind") != "speed":
        raise ValueError(f"Suite {suite_path} is not a speed suite.")

    speed_config = suite_config.get("speed", {})
    batch_sizes = request.batch_sizes or [int(item) for item in speed_config.get("batch_sizes", [1])]
    prompt_lengths = request.prompt_lengths or [int(item) for item in speed_config.get("prompt_lengths", [512])]
    generation_lengths = request.generation_lengths or [int(item) for item in speed_config.get("generation_lengths", [128])]
    repeat = int(request.repeat if request.repeat is not None else speed_config.get("repeat", 5))
    warmup = int(request.warmup if request.warmup is not None else speed_config.get("warmup", 1))
    if repeat < 1:
        raise ValueError("repeat must be at least 1 for speed benchmarking.")
    tensor_parallel_size = int(
        request.tensor_parallel_size if request.tensor_parallel_size is not None else speed_config.get("tensor_parallel_size", 1)
    )
    gpu_memory_utilization = float(
        request.gpu_memory_utilization
        if request.gpu_memory_utilization is not None
        else speed_config.get("gpu_memory_utilization", 0.9)
    )
    dtype = request.dtype or speed_config.get("dtype", "auto")
    max_model_len = (
        int(request.max_model_len)
        if request.max_model_len is not None
        else (int(speed_config["max_model_len"]) if speed_config.get("max_model_len") is not None else None)
    )
    enforce_eager = _bool_from_optional_flag(request.enforce_eager, bool(speed_config.get("enforce_eager", False)))

    progress.step(1, "Resolving suite and checkpoint")
    progress.detail(f"suite={suite_path}")
    progress.detail(f"checkpoint={request.checkpoint_name}")

    with progress.waiting(2, "Materializing checkpoint"):
        loaded = load_checkpoint(
            request.checkpoint_name,
            index_path=str(request.index_path),
            download=True,
            local_files_only=request.local_files_only,
            trust_remote_code=request.trust_remote_code,
        )
    progress.detail(f"downloaded_path={loaded.local_path or loaded.locator}")

    progress.step(3, "Preparing model for vLLM")
    prepared = prepare_model_for_vllm(
        loaded,
        wrapper_cache_root=request.wrapper_cache_root,
    )
    progress.detail(f"preparation={prepared.preparation_kind}")
    progress.detail(f"model_path={prepared.model_path}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    progress.step(4, "Loading tokenizer")
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": request.trust_remote_code}
    if prepared.tokenizer_mode == "slow":
        tokenizer_kwargs["use_fast"] = False
    tokenizer = AutoTokenizer.from_pretrained(prepared.tokenizer_path, **tokenizer_kwargs)

    llm_kwargs: dict[str, Any] = {
        "model": prepared.model_path,
        "tokenizer": prepared.tokenizer_path,
        "tokenizer_mode": prepared.tokenizer_mode,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": request.trust_remote_code,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "enforce_eager": enforce_eager,
        "use_tqdm_on_load": False,
    }
    if prepared.model_impl is not None:
        llm_kwargs["model_impl"] = prepared.model_impl
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm_kwargs.update(prepared.extra_llm_kwargs)

    try:
        with progress.waiting(5, "Initializing vLLM engine"):
            llm = LLM(**llm_kwargs)
    except Exception as exc:
        text = str(exc)
        if "Free memory on device" in text or "GPU memory utilization" in text:
            print_failure(
                "vLLM init failed because the selected GPU does not have enough free memory. "
                "Try a less busy GPU or pass a smaller --gpu-memory-utilization such as 0.3."
            )
            raise SystemExit(1) from None
        if request.verbose_vllm:
            raise
        print_failure(f"vLLM init failed: {exc}")
        raise SystemExit(1) from None

    total_cases = len(prompt_lengths) * len(generation_lengths) * len(batch_sizes)
    progress.step(6, f"Running benchmark cases ({total_cases} total)")
    cases: list[dict[str, Any]] = []
    case_index = 0
    for prompt_length in prompt_lengths:
        prompt_token_ids = _build_prompt_token_ids(tokenizer, prompt_length)
        for generation_length in generation_lengths:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=int(generation_length),
                ignore_eos=True,
            )
            for batch_size in batch_sizes:
                case_index += 1
                progress.detail(
                    f"case {case_index}/{total_cases}: batch={batch_size}, prompt={prompt_length}, generation={generation_length}"
                )
                prompts = [prompt_token_ids] * int(batch_size)
                for warmup_index in range(warmup):
                    progress.detail(f"warmup {warmup_index + 1}/{warmup}")
                    llm.generate(prompts, sampling_params, use_tqdm=False)

                run_stats: list[dict[str, float]] = []
                for run_index in range(repeat):
                    progress.detail(f"timed run {run_index + 1}/{repeat}")
                    started = time.perf_counter()
                    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                    elapsed = time.perf_counter() - started
                    prompt_tokens = sum(len(output.prompt_token_ids or []) for output in outputs)
                    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
                    run_stats.append(
                        {
                            "latency_seconds": elapsed,
                            "prompt_tokens": float(prompt_tokens),
                            "generated_tokens": float(generated_tokens),
                            "prefill_tokens_per_second": float(prompt_tokens) / elapsed if elapsed else 0.0,
                            "decode_tokens_per_second": float(generated_tokens) / elapsed if elapsed else 0.0,
                            "end_to_end_tokens_per_second": float(prompt_tokens + generated_tokens) / elapsed if elapsed else 0.0,
                        }
                    )

                cases.append(
                    {
                        "batch_size": int(batch_size),
                        "prompt_length": int(prompt_length),
                        "generation_length": int(generation_length),
                        "repeat": repeat,
                        "warmup": warmup,
                        "latency_seconds": statistics.mean(item["latency_seconds"] for item in run_stats),
                        "prefill_tokens_per_second": statistics.mean(item["prefill_tokens_per_second"] for item in run_stats),
                        "decode_tokens_per_second": statistics.mean(item["decode_tokens_per_second"] for item in run_stats),
                        "end_to_end_tokens_per_second": statistics.mean(item["end_to_end_tokens_per_second"] for item in run_stats),
                        "runs": run_stats,
                    }
                )

    stats = summarize_speed_cases(cases)
    payload = {
        "kind": "speed",
        "backend": speed_config.get("backend", "vllm"),
        "backend_version": None,
        "checkpoint": loaded.record.name,
        "suite": _safe_suite_output_name(suite_path),
        "suite_name": suite_config.get("name", suite_path.stem),
        "suite_path": _safe_suite_id(suite_path),
        "status": "completed",
        "locator": loaded.locator,
        "stats": stats,
        "cases": cases,
        "meta": {
            "model_family": loaded.record.model_family,
            "variant": loaded.record.variant,
            "method": loaded.record.method,
            "source": loaded.record.source,
            "repo_id": loaded.record.repo_id,
            "revision": loaded.record.revision,
            "subpath": loaded.record.subpath,
            "benchmarks": loaded.record.benchmarks,
            "notes": loaded.record.notes,
        },
        "prepared_model": {
            "model_path": prepared.model_path,
            "tokenizer_path": prepared.tokenizer_path,
            "tokenizer_mode": prepared.tokenizer_mode,
            "model_impl": prepared.model_impl,
            "preparation_kind": prepared.preparation_kind,
            "source_model_path": prepared.source_model_path,
            "notes": prepared.notes,
        },
        "runtime": {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
        },
        "generated_at": utc_timestamp(),
    }
    try:
        import vllm

        payload["backend_version"] = getattr(vllm, "__version__", None)
    except Exception:
        payload["backend_version"] = None

    output_path = dump_json(payload, _result_path_for(request))
    return VllmSpeedResult(
        checkpoint_name=request.checkpoint_name,
        suite=_safe_suite_output_name(suite_path),
        status="completed",
        output_path=str(output_path),
        stats=stats,
        prepared_model=prepared,
    )


def main() -> None:
    args = parse_args()
    result = run_vllm_speed_suite(
        VllmSpeedRequest(
            checkpoint_name=args.checkpoint_name,
            suite_path=args.suite_path,
            index_path=args.index_path,
            output_dir=args.output_dir,
            wrapper_cache_root=args.wrapper_cache_root,
            batch_sizes=args.batch_sizes,
            prompt_lengths=args.prompt_lengths,
            generation_lengths=args.generation_lengths,
            repeat=args.repeat,
            warmup=args.warmup,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
            local_files_only=args.local_files_only,
            verbose_vllm=args.verbose_vllm,
        )
    )
    print(f"[ok] speed suite completed: {result.output_path}")
    print(f"[ok] preparation kind: {result.prepared_model.preparation_kind}")
    print(f"[ok] mean prefill tok/s: {result.stats['mean_prefill_tokens_per_second']}")
    print(f"[ok] mean decode tok/s: {result.stats['mean_decode_tokens_per_second']}")


if __name__ == "__main__":
    main()
