from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.benchmarking import suite_output_name
from src.load import load_checkpoint
from src.result_schema import build_result_payload
from src.scoring import summarize_speed_cases
from src.utils import dump_json, ensure_dir, load_yaml, run_results_root
from src.validation import validate_checkpoint_layout
from src.vllm.external_vllm import import_installed_vllm, installed_vllm_version
from src.vllm.terminal_ui import ProgressPrinter, configure_runtime_environment, use_safe_vllm_cwd
from src.vllm.vllm_adapter import prepare_model_for_vllm


@dataclass(slots=True)
class VllmSpeedRequest:
    checkpoint_name: str
    suite_path: str | Path
    index_path: str | Path
    output_dir: str | Path | None = None
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
    verbose_backend: bool = False
    show_progress: bool = False
    run_label: str = "ad_hoc"
    strict_validation: bool = False


@dataclass(slots=True)
class VllmSpeedResult:
    checkpoint_name: str
    suite: str
    status: str
    output_path: str
    stats: dict[str, Any]


def _build_prompt_token_ids(tokenizer: Any, prompt_length: int) -> list[int]:
    seed_text = "The quick brown fox jumps over the lazy dog. "
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("Tokenizer produced no seed token ids.")
    repeats = math.ceil(prompt_length / len(seed_ids))
    return (seed_ids * repeats)[:prompt_length]


def _result_path_for(request: VllmSpeedRequest) -> Path:
    output_root = Path(request.output_dir) if request.output_dir else run_results_root("speed", request.run_label)
    ensure_dir(output_root)
    suite_name = suite_output_name(request.suite_path)
    return output_root / f"{suite_name}__{request.checkpoint_name}.json"


def _cartesian_speed_case_specs(
    *,
    batch_sizes: list[int],
    prompt_lengths: list[int],
    generation_lengths: list[int],
) -> list[dict[str, int | str]]:
    cases: list[dict[str, int | str]] = []
    for prompt_length in prompt_lengths:
        for generation_length in generation_lengths:
            for batch_size in batch_sizes:
                cases.append(
                    {
                        "name": f"batch{batch_size}_prompt{prompt_length}_gen{generation_length}",
                        "batch_size": int(batch_size),
                        "prompt_length": int(prompt_length),
                        "generation_length": int(generation_length),
                    }
                )
    return cases


def _normalize_speed_case_spec(case: dict[str, Any], *, index: int) -> dict[str, int | str]:
    required_fields = ("batch_size", "prompt_length", "generation_length")
    missing = [field for field in required_fields if case.get(field) is None]
    if missing:
        raise ValueError(f"Speed case #{index} is missing required fields: {', '.join(missing)}")
    return {
        "name": str(case.get("name", f"case_{index}")),
        "batch_size": int(case["batch_size"]),
        "prompt_length": int(case["prompt_length"]),
        "generation_length": int(case["generation_length"]),
    }


def _resolve_speed_case_specs(speed_config: dict[str, Any], request: VllmSpeedRequest) -> list[dict[str, int | str]]:
    speed_batch_sizes = [int(item) for item in speed_config.get("batch_sizes", [1])]
    speed_prompt_lengths = [int(item) for item in speed_config.get("prompt_lengths", [512])]
    speed_generation_lengths = [int(item) for item in speed_config.get("generation_lengths", [128])]
    has_axis_override = any(
        value is not None
        for value in (
            request.batch_sizes,
            request.prompt_lengths,
            request.generation_lengths,
        )
    )
    if has_axis_override:
        return _cartesian_speed_case_specs(
            batch_sizes=request.batch_sizes or speed_batch_sizes,
            prompt_lengths=request.prompt_lengths or speed_prompt_lengths,
            generation_lengths=request.generation_lengths or speed_generation_lengths,
        )

    configured_cases = speed_config.get("cases")
    if configured_cases is not None:
        if not isinstance(configured_cases, list) or not configured_cases:
            raise ValueError("Speed suites that define `speed.cases` must provide a non-empty list.")
        return [_normalize_speed_case_spec(case, index=index) for index, case in enumerate(configured_cases, start=1)]

    return _cartesian_speed_case_specs(
        batch_sizes=speed_batch_sizes,
        prompt_lengths=speed_prompt_lengths,
        generation_lengths=speed_generation_lengths,
    )


def run_vllm_speed_suite(request: VllmSpeedRequest) -> VllmSpeedResult:
    configure_runtime_environment(verbose_vllm=request.verbose_backend)
    progress = ProgressPrinter(
        total_steps=4,
        enabled=request.show_progress and not request.verbose_backend,
    )
    progress.step(1, "Loading speed suite")
    suite_path = Path(request.suite_path)
    suite_config = load_yaml(suite_path)
    if suite_config.get("kind") != "speed":
        raise ValueError(f"Suite {suite_path} is not a speed suite.")

    speed_config = suite_config.get("speed", {})
    case_specs = _resolve_speed_case_specs(speed_config, request)
    batch_sizes = sorted({int(case["batch_size"]) for case in case_specs})
    prompt_lengths = sorted({int(case["prompt_length"]) for case in case_specs})
    generation_lengths = sorted({int(case["generation_length"]) for case in case_specs})
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
    enforce_eager = bool(request.enforce_eager if request.enforce_eager is not None else speed_config.get("enforce_eager", False))

    progress.step(2, "Preparing checkpoint for vLLM")
    loaded = load_checkpoint(
        request.checkpoint_name,
        index_path=str(request.index_path),
        download=True,
        local_files_only=request.local_files_only,
        trust_remote_code=request.trust_remote_code,
    )
    prepared = prepare_model_for_vllm(
        loaded,
    )
    validation_summary = validate_checkpoint_layout(
        prepared.model_path,
        strict=request.strict_validation,
    )

    from transformers import AutoTokenizer
    installed_vllm = import_installed_vllm()
    LLM = installed_vllm.LLM
    SamplingParams = installed_vllm.SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        prepared.tokenizer_path,
        **prepared.build_tokenizer_kwargs(trust_remote_code=request.trust_remote_code),
    )

    llm_kwargs = prepared.build_llm_kwargs(
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=request.trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        disable_log_stats=not request.verbose_backend,
    )

    with progress.waiting(3, "Initializing vLLM engine"):
        with use_safe_vllm_cwd():
            llm = LLM(**llm_kwargs)

    progress.step(4, f"Running speed cases ({len(case_specs)} total)")
    cases: list[dict[str, Any]] = []
    prompt_token_ids_by_length: dict[int, list[int]] = {}
    for case_spec in case_specs:
        prompt_length = int(case_spec["prompt_length"])
        generation_length = int(case_spec["generation_length"])
        batch_size = int(case_spec["batch_size"])
        case_name = str(case_spec.get("name", f"batch{batch_size}_prompt{prompt_length}_gen{generation_length}"))
        prompt_token_ids = prompt_token_ids_by_length.get(prompt_length)
        if prompt_token_ids is None:
            prompt_token_ids = _build_prompt_token_ids(tokenizer, prompt_length)
            prompt_token_ids_by_length[prompt_length] = prompt_token_ids
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=generation_length,
            ignore_eos=True,
        )
        prompts = [prompt_token_ids] * batch_size
        for _ in range(warmup):
            llm.generate(prompts, sampling_params, use_tqdm=False)

        run_stats: list[dict[str, float]] = []
        for _ in range(repeat):
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
                "name": case_name,
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "generation_length": generation_length,
                "repeat": repeat,
                "warmup": warmup,
                "latency_seconds": statistics.mean(item["latency_seconds"] for item in run_stats),
                "prefill_tokens_per_second": statistics.mean(item["prefill_tokens_per_second"] for item in run_stats),
                "decode_tokens_per_second": statistics.mean(item["decode_tokens_per_second"] for item in run_stats),
                "end_to_end_tokens_per_second": statistics.mean(item["end_to_end_tokens_per_second"] for item in run_stats),
                "runs": run_stats,
            }
        )

    aggregation = str(speed_config.get("metric_aggregation", "macro_mean"))
    stats = summarize_speed_cases(cases, aggregation=aggregation)
    payload = build_result_payload(
        kind="speed",
        record=loaded.record,
        locator=loaded.locator,
        backend_name=speed_config.get("backend", "vllm"),
        backend_version=installed_vllm_version(),
        suite_path=suite_path,
        suite_name=suite_config.get("name", suite_path.stem),
        config={
            "cases": case_specs,
            "batch_sizes": batch_sizes,
            "prompt_lengths": prompt_lengths,
            "generation_lengths": generation_lengths,
            "repeat": repeat,
            "warmup": warmup,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            "metric_aggregation": aggregation,
            "trust_remote_code": request.trust_remote_code,
            "local_files_only": request.local_files_only,
        },
        metrics=stats,
        artifacts={},
        runtime={
            "model_path": prepared.model_path,
            "tokenizer_path": prepared.tokenizer_path,
            "tokenizer_mode": prepared.tokenizer_mode,
            "model_impl": prepared.model_impl,
            "preparation_kind": prepared.preparation_kind,
            "source_model_path": prepared.source_model_path,
            "preparation_notes": prepared.notes,
        },
        validation=validation_summary,
        details={"cases": cases},
        run_label=request.run_label,
        strict_validation=request.strict_validation,
    )

    output_path = dump_json(payload, _result_path_for(request))
    return VllmSpeedResult(
        checkpoint_name=request.checkpoint_name,
        suite=suite_output_name(suite_path),
        status="completed",
        output_path=str(output_path),
        stats=stats,
    )
