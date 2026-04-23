from __future__ import annotations

import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.benchmarking import resolve_suite_path, suite_id, suite_output_name
from src.dtype_utils import normalize_dtype_name
from src.hardware import describe_cuda_runtime
from src.load import load_checkpoint
from src.lm_eval_runner import DEFAULT_LM_EVAL_BIN, LmEvalRequest, run_lm_eval_suite
from src.result_schema import build_result_payload
from src.scoring import aggregate_values, summarize_speed_cases
from src.utils import dump_json, ensure_dir, load_json, load_yaml, run_results_root
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
    lm_eval_bin: str | None = None
    eval_model_backend: str | None = None
    eval_device: str | None = None
    eval_batch_size: str | int | None = None
    eval_limit: float | int | None = None
    eval_num_fewshot: int | None = None
    host: str | None = None
    port: int | None = None
    request_rate: float | str | None = None
    num_prompts: int | None = None
    max_concurrency: int | None = None
    ready_check_timeout_seconds: int | None = None
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


def _nested_eval_roots(request: VllmSpeedRequest) -> tuple[Path, Path]:
    speed_root = Path(request.output_dir) if request.output_dir else run_results_root("speed", request.run_label)
    eval_root = ensure_dir(speed_root / "eval_artifacts")
    raw_root = ensure_dir(eval_root / "raw")
    return eval_root, raw_root


def _coerce_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _summarize_nested_eval_run(
    *,
    eval_suite_path: Path,
    wall_time_seconds: float,
    result_output_path: str,
    raw_output_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_json(result_output_path)
    raw_payload = load_json(raw_output_path)
    backend_name = str(payload.get("backend", {}).get("name", "eval"))
    work_unit_kind: str | None = None
    work_unit_count: int | None = None
    backend_reported_time_seconds: float | None = None

    if backend_name == "lm_eval_harness":
        n_samples = raw_payload.get("n-samples", {})
        if isinstance(n_samples, dict):
            work_unit_kind = "examples"
            work_unit_count = sum(_coerce_count(value) or 0 for value in n_samples.values())
        backend_time = raw_payload.get("total_evaluation_time_seconds")
        if isinstance(backend_time, (int, float)):
            backend_reported_time_seconds = float(backend_time)
    elif backend_name == "contiguous_ppl":
        tasks = raw_payload.get("tasks", [])
        if isinstance(tasks, list):
            work_unit_kind = "tokens"
            work_unit_count = sum(_coerce_count(item.get("token_count")) or 0 for item in tasks if isinstance(item, dict))
        backend_time = raw_payload.get("total_evaluation_time_seconds")
        if isinstance(backend_time, (int, float)):
            backend_reported_time_seconds = float(backend_time)

    throughput = None
    if work_unit_count is not None and wall_time_seconds > 0:
        throughput = float(work_unit_count) / float(wall_time_seconds)

    suite_detail = {
        "suite": suite_id(eval_suite_path),
        "suite_name": payload.get("suite", {}).get("name", eval_suite_path.stem),
        "backend": backend_name,
        "wall_time_seconds": wall_time_seconds,
        "backend_reported_time_seconds": backend_reported_time_seconds,
        "work_unit_kind": work_unit_kind,
        "work_unit_count": work_unit_count,
        "work_units_per_second": throughput,
        "primary_metric": payload.get("metrics", {}).get("primary_metric"),
        "metric_mean": payload.get("metrics", {}).get("mean"),
        "output_path": result_output_path,
        "raw_output_path": raw_output_path,
    }
    return suite_detail, payload.get("validation", {})


def _summarize_evaluation_speed_runs(suite_runs: list[dict[str, Any]], *, aggregation: str) -> dict[str, Any]:
    wall_times = [float(item["wall_time_seconds"]) for item in suite_runs if item.get("wall_time_seconds") is not None]
    grouped_units: dict[str, list[dict[str, Any]]] = {}
    for item in suite_runs:
        unit_kind = item.get("work_unit_kind")
        if not unit_kind:
            continue
        grouped_units.setdefault(str(unit_kind), []).append(item)

    by_work_unit: dict[str, dict[str, Any]] = {}
    for unit_kind, items in grouped_units.items():
        counts = [int(item["work_unit_count"]) for item in items if item.get("work_unit_count") is not None]
        throughputs = [float(item["work_units_per_second"]) for item in items if item.get("work_units_per_second") is not None]
        total_time = sum(float(item["wall_time_seconds"]) for item in items if item.get("wall_time_seconds") is not None)
        total_count = sum(counts)
        by_work_unit[unit_kind] = {
            "suite_count": len(items),
            "total_count": total_count,
            "mean_throughput_per_second": aggregate_values(throughputs, aggregation=aggregation),
            "total_throughput_per_second": (float(total_count) / total_time) if total_time > 0 and total_count > 0 else None,
        }

    return {
        "aggregation": aggregation,
        "suite_count": len(suite_runs),
        "completed_suite_count": len(suite_runs),
        "total_wall_time_seconds": sum(wall_times),
        "mean_suite_wall_time_seconds": aggregate_values(wall_times, aggregation=aggregation),
        "max_suite_wall_time_seconds": max(wall_times) if wall_times else None,
        "by_work_unit": by_work_unit,
    }


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


def _normalize_bench_serve_profile(profile: dict[str, Any], *, index: int) -> dict[str, int | str]:
    input_length = profile.get("input_length", profile.get("prompt_length"))
    output_length = profile.get("output_length", profile.get("generation_length"))
    missing = []
    if input_length is None:
        missing.append("input_length")
    if output_length is None:
        missing.append("output_length")
    if missing:
        raise ValueError(f"vLLM bench serve profile #{index} is missing required fields: {', '.join(missing)}")
    return {
        "name": str(profile.get("name", f"profile_{index}")),
        "input_length": int(input_length),
        "output_length": int(output_length),
    }


def _resolve_bench_serve_profiles(speed_config: dict[str, Any], request: VllmSpeedRequest) -> list[dict[str, int | str]]:
    has_axis_override = request.prompt_lengths is not None or request.generation_lengths is not None
    if has_axis_override:
        prompt_lengths = request.prompt_lengths or [int(item) for item in speed_config.get("prompt_lengths", [512])]
        generation_lengths = request.generation_lengths or [
            int(item) for item in speed_config.get("generation_lengths", [128])
        ]
        profiles: list[dict[str, int | str]] = []
        for input_length in prompt_lengths:
            for output_length in generation_lengths:
                profiles.append(
                    {
                        "name": f"input{int(input_length)}_output{int(output_length)}",
                        "input_length": int(input_length),
                        "output_length": int(output_length),
                    }
                )
        return profiles

    configured_profiles = speed_config.get("profiles", speed_config.get("cases"))
    if configured_profiles is None:
        return [
            {
                "name": "input512_output128",
                "input_length": 512,
                "output_length": 128,
            }
        ]
    if not isinstance(configured_profiles, list) or not configured_profiles:
        raise ValueError("vLLM bench serve suites must provide a non-empty `speed.profiles` list.")
    return [
        _normalize_bench_serve_profile(profile, index=index)
        for index, profile in enumerate(configured_profiles, start=1)
    ]


def _format_cli_value(value: Any) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile_value(raw: dict[str, Any], metric: str, percentile: int | float) -> float | None:
    if float(percentile) == 50.0:
        median_value = _float_or_none(raw.get(f"median_{metric}_ms"))
        if median_value is not None:
            return median_value

    percentile_text = str(int(percentile)) if float(percentile).is_integer() else str(percentile)
    for key in (f"p{percentile_text}_{metric}_ms", f"percentile_{percentile_text}_{metric}_ms"):
        value = _float_or_none(raw.get(key))
        if value is not None:
            return value

    percentile_payload = raw.get(f"percentiles_{metric}_ms")
    if isinstance(percentile_payload, dict):
        for key, value in percentile_payload.items():
            numeric_key = _float_or_none(key)
            if numeric_key is not None and numeric_key == float(percentile):
                return _float_or_none(value)
    if isinstance(percentile_payload, list):
        for item in percentile_payload:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                numeric_key = _float_or_none(item[0])
                if numeric_key is not None and numeric_key == float(percentile):
                    return _float_or_none(item[1])
    return None


def _latency_metric_summary(raw: dict[str, Any], metric: str) -> dict[str, float | None]:
    return {
        "mean": _float_or_none(raw.get(f"mean_{metric}_ms")),
        "p50": _percentile_value(raw, metric, 50),
        "p95": _percentile_value(raw, metric, 95),
        "p99": _percentile_value(raw, metric, 99),
    }


def _normalize_bench_serve_case(
    *,
    profile: dict[str, int | str],
    raw_payload: dict[str, Any],
    command: list[str],
    raw_output_path: Path,
    wall_time_seconds: float,
) -> dict[str, Any]:
    return {
        "name": str(profile["name"]),
        "input_length": int(profile["input_length"]),
        "output_length": int(profile["output_length"]),
        "completed": raw_payload.get("completed"),
        "failed": raw_payload.get("failed"),
        "total_input_tokens": raw_payload.get("total_input"),
        "total_output_tokens": raw_payload.get("total_output"),
        "request_throughput": _float_or_none(raw_payload.get("request_throughput")),
        "request_goodput": _float_or_none(raw_payload.get("request_goodput")),
        "output_tokens_per_second": _float_or_none(raw_payload.get("output_throughput")),
        "total_tokens_per_second": _float_or_none(raw_payload.get("total_token_throughput")),
        "max_output_tokens_per_second": _float_or_none(raw_payload.get("max_output_tokens_per_s")),
        "max_concurrent_requests_observed": raw_payload.get("max_concurrent_requests"),
        "ttft_ms": _latency_metric_summary(raw_payload, "ttft"),
        "tpot_ms": _latency_metric_summary(raw_payload, "tpot"),
        "itl_ms": _latency_metric_summary(raw_payload, "itl"),
        "e2e_latency_ms": _latency_metric_summary(raw_payload, "e2el"),
        "wall_time_seconds": wall_time_seconds,
        "raw_output_path": str(raw_output_path),
        "bench_command": command,
    }


def _summarize_bench_serve_cases(cases: list[dict[str, Any]], *, aggregation: str) -> dict[str, Any]:
    def collect(name: str) -> list[float]:
        return [float(case[name]) for case in cases if case.get(name) is not None]

    def collect_latency(metric_name: str, percentile_name: str) -> list[float]:
        values = []
        for case in cases:
            metric_payload = case.get(metric_name)
            if isinstance(metric_payload, dict) and metric_payload.get(percentile_name) is not None:
                values.append(float(metric_payload[percentile_name]))
        return values

    metrics: dict[str, Any] = {
        "profile_count": len(cases),
        "aggregation": aggregation,
        "completed_requests": sum(int(case.get("completed") or 0) for case in cases),
        "failed_requests": sum(int(case.get("failed") or 0) for case in cases),
        "mean_request_throughput": aggregate_values(collect("request_throughput"), aggregation=aggregation),
        "mean_output_tokens_per_second": aggregate_values(collect("output_tokens_per_second"), aggregation=aggregation),
        "mean_total_tokens_per_second": aggregate_values(collect("total_tokens_per_second"), aggregation=aggregation),
    }
    for normalized_name in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_latency_ms"):
        for percentile_name in ("p50", "p95", "p99"):
            metrics[f"mean_{normalized_name}_{percentile_name}"] = aggregate_values(
                collect_latency(normalized_name, percentile_name),
                aggregation=aggregation,
            )
    return metrics


def _bench_serve_artifact_root(request: VllmSpeedRequest, suite_path: Path) -> Path:
    speed_root = Path(request.output_dir) if request.output_dir else run_results_root("speed", request.run_label)
    return ensure_dir(speed_root / "bench_serve_artifacts" / suite_output_name(suite_path) / request.checkpoint_name)


def _build_vllm_serve_command(
    *,
    vllm_bin: str,
    prepared: Any,
    served_model_name: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_model_len: int | None,
    enforce_eager: bool,
    trust_remote_code: bool,
    verbose_backend: bool,
    server_log_level: str,
) -> list[str]:
    command = [
        vllm_bin,
        "serve",
        prepared.model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--tokenizer",
        prepared.tokenizer_path,
        "--tokenizer-mode",
        prepared.tokenizer_mode,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
        "--uvicorn-log-level",
        server_log_level,
    ]
    model_impl = getattr(prepared, "model_impl", None)
    if model_impl is not None:
        command.extend(["--model-impl", str(model_impl)])
    if max_model_len is not None:
        command.extend(["--max-model-len", str(max_model_len)])
    if enforce_eager:
        command.append("--enforce-eager")
    if trust_remote_code:
        command.append("--trust-remote-code")
    if not verbose_backend:
        command.extend(["--disable-log-stats", "--disable-uvicorn-access-log"])
    return command


def _build_vllm_bench_serve_command(
    *,
    vllm_bin: str,
    profile: dict[str, int | str],
    served_model_name: str,
    tokenizer_path: str,
    tokenizer_mode: str,
    backend: str,
    host: str,
    port: int,
    endpoint: str,
    request_rate: float | str,
    num_prompts: int,
    max_concurrency: int,
    num_warmups: int,
    seed: int,
    random_range_ratio: float,
    percentiles: list[int],
    result_dir: Path,
    result_filename: str,
    trust_remote_code: bool,
    ignore_eos: bool,
    ready_check_timeout_seconds: int,
) -> list[str]:
    command = [
        vllm_bin,
        "bench",
        "serve",
        "--backend",
        backend,
        "--host",
        host,
        "--port",
        str(port),
        "--endpoint",
        endpoint,
        "--model",
        served_model_name,
        "--served-model-name",
        served_model_name,
        "--tokenizer",
        tokenizer_path,
        "--tokenizer-mode",
        tokenizer_mode,
        "--dataset-name",
        "random",
        "--input-len",
        str(profile["input_length"]),
        "--output-len",
        str(profile["output_length"]),
        "--random-range-ratio",
        str(random_range_ratio),
        "--request-rate",
        _format_cli_value(request_rate),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(max_concurrency),
        "--num-warmups",
        str(num_warmups),
        "--seed",
        str(seed),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        ",".join(str(item) for item in percentiles),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
        "--ready-check-timeout-sec",
        str(ready_check_timeout_seconds),
        "--disable-tqdm",
        "--disable-shuffle",
        "--temperature",
        "0.0",
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    if ignore_eos:
        command.append("--ignore-eos")
    return command


def _terminate_process(process: subprocess.Popen[Any], *, timeout_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)


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
    backend_name = str(speed_config.get("backend", "vllm")).strip().lower()
    if backend_name != "vllm":
        raise ValueError(f"Suite {suite_path} is configured for backend {backend_name!r}, not vllm.")
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
    dtype = normalize_dtype_name(request.dtype or speed_config.get("dtype", "auto"))
    max_model_len = (
        int(request.max_model_len)
        if request.max_model_len is not None
        else (int(speed_config["max_model_len"]) if speed_config.get("max_model_len") is not None else None)
    )
    enforce_eager = bool(request.enforce_eager if request.enforce_eager is not None else speed_config.get("enforce_eager", False))
    cuda_runtime = describe_cuda_runtime(limit=tensor_parallel_size)

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
            "cuda_runtime": cuda_runtime,
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


def run_vllm_bench_serve_suite(request: VllmSpeedRequest) -> VllmSpeedResult:
    configure_runtime_environment(verbose_vllm=request.verbose_backend)
    progress = ProgressPrinter(
        total_steps=5,
        enabled=request.show_progress and not request.verbose_backend,
    )
    progress.step(1, "Loading online serving speed suite")
    suite_path = Path(request.suite_path)
    suite_config = load_yaml(suite_path)
    if suite_config.get("kind") != "speed":
        raise ValueError(f"Suite {suite_path} is not a speed suite.")

    speed_config = suite_config.get("speed", {})
    backend_name = str(speed_config.get("backend", "")).strip().lower()
    if backend_name != "vllm_bench_serve":
        raise ValueError(f"Suite {suite_path} is configured for backend {backend_name!r}, not vllm_bench_serve.")

    profiles = _resolve_bench_serve_profiles(speed_config, request)
    aggregation = str(speed_config.get("metric_aggregation", "macro_mean"))
    tensor_parallel_size = int(
        request.tensor_parallel_size if request.tensor_parallel_size is not None else speed_config.get("tensor_parallel_size", 1)
    )
    gpu_memory_utilization = float(
        request.gpu_memory_utilization
        if request.gpu_memory_utilization is not None
        else speed_config.get("gpu_memory_utilization", 0.9)
    )
    dtype = normalize_dtype_name(request.dtype or speed_config.get("dtype", "auto"))
    max_model_len = (
        int(request.max_model_len)
        if request.max_model_len is not None
        else (int(speed_config["max_model_len"]) if speed_config.get("max_model_len") is not None else None)
    )
    enforce_eager = bool(request.enforce_eager if request.enforce_eager is not None else speed_config.get("enforce_eager", False))
    host = str(request.host if request.host is not None else speed_config.get("host", "127.0.0.1"))
    port = int(request.port if request.port is not None else speed_config.get("port", 8000))
    request_rate = request.request_rate if request.request_rate is not None else speed_config.get("request_rate", 1.0)
    num_prompts = int(request.num_prompts if request.num_prompts is not None else speed_config.get("num_prompts", 64))
    max_concurrency = int(
        request.max_concurrency if request.max_concurrency is not None else speed_config.get("max_concurrency", 16)
    )
    num_warmups = int(speed_config.get("num_warmups", speed_config.get("warmup", 1)))
    seed = int(speed_config.get("seed", 0))
    random_range_ratio = float(speed_config.get("random_range_ratio", 0.0))
    percentiles = [int(item) for item in speed_config.get("metric_percentiles", [50, 95, 99])]
    for required_percentile in (50, 95, 99):
        if required_percentile not in percentiles:
            percentiles.append(required_percentile)
    percentiles = sorted(percentiles)
    endpoint = str(speed_config.get("endpoint", "/v1/completions"))
    client_backend = str(speed_config.get("client_backend", "openai"))
    vllm_bin = str(speed_config.get("vllm_bin", "vllm"))
    server_log_level = str(speed_config.get("server_log_level", "error" if not request.verbose_backend else "info"))
    ready_check_timeout_seconds = int(
        request.ready_check_timeout_seconds
        if request.ready_check_timeout_seconds is not None
        else speed_config.get("ready_check_timeout_seconds", 600)
    )
    server_startup_grace_seconds = float(speed_config.get("server_startup_grace_seconds", 2.0))
    ignore_eos = bool(speed_config.get("ignore_eos", True))
    cuda_runtime = describe_cuda_runtime(limit=tensor_parallel_size)

    progress.step(2, "Preparing checkpoint for vLLM serving")
    loaded = load_checkpoint(
        request.checkpoint_name,
        index_path=str(request.index_path),
        download=True,
        local_files_only=request.local_files_only,
        trust_remote_code=request.trust_remote_code,
    )
    prepared = prepare_model_for_vllm(loaded)
    validation_summary = validate_checkpoint_layout(
        prepared.model_path,
        strict=request.strict_validation,
    )
    served_model_name = str(speed_config.get("served_model_name", request.checkpoint_name))
    raw_root = _bench_serve_artifact_root(request, suite_path)
    server_stdout_path = raw_root / "server.stdout.log"
    server_stderr_path = raw_root / "server.stderr.log"
    server_command = _build_vllm_serve_command(
        vllm_bin=vllm_bin,
        prepared=prepared,
        served_model_name=served_model_name,
        host=host,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        trust_remote_code=request.trust_remote_code,
        verbose_backend=request.verbose_backend,
        server_log_level=server_log_level,
    )

    progress.step(3, "Starting vLLM OpenAI-compatible server")
    safe_cwd = Path(__file__).resolve().parent
    env = os.environ.copy()
    server_stdout = server_stdout_path.open("w", encoding="utf-8")
    server_stderr = server_stderr_path.open("w", encoding="utf-8")
    try:
        server_process = subprocess.Popen(
            server_command,
            cwd=str(safe_cwd),
            stdout=server_stdout,
            stderr=server_stderr,
            env=env,
        )
    except Exception:
        server_stdout.close()
        server_stderr.close()
        raise
    server_returncode: int | None = None
    cases: list[dict[str, Any]] = []
    try:
        if server_startup_grace_seconds > 0:
            time.sleep(server_startup_grace_seconds)
        server_returncode = server_process.poll()
        if server_returncode is not None:
            raise RuntimeError(
                "vLLM server exited before benchmark clients started. "
                f"returncode={server_returncode}; stderr={server_stderr_path}"
            )

        progress.step(4, f"Running vLLM bench serve profiles ({len(profiles)} total)")
        for profile in profiles:
            profile_name = str(profile["name"])
            safe_profile_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in profile_name)
            raw_result_path = raw_root / f"{safe_profile_name}.json"
            bench_stdout_path = raw_root / f"{safe_profile_name}.stdout.log"
            bench_stderr_path = raw_root / f"{safe_profile_name}.stderr.log"
            if raw_result_path.exists():
                raw_result_path.unlink()
            bench_command = _build_vllm_bench_serve_command(
                vllm_bin=vllm_bin,
                profile=profile,
                served_model_name=served_model_name,
                tokenizer_path=prepared.tokenizer_path,
                tokenizer_mode=prepared.tokenizer_mode,
                backend=client_backend,
                host=host,
                port=port,
                endpoint=endpoint,
                request_rate=request_rate,
                num_prompts=num_prompts,
                max_concurrency=max_concurrency,
                num_warmups=num_warmups,
                seed=seed,
                random_range_ratio=random_range_ratio,
                percentiles=percentiles,
                result_dir=raw_root,
                result_filename=raw_result_path.name,
                trust_remote_code=request.trust_remote_code,
                ignore_eos=ignore_eos,
                ready_check_timeout_seconds=ready_check_timeout_seconds,
            )
            started = time.perf_counter()
            with (
                bench_stdout_path.open("w", encoding="utf-8") as bench_stdout,
                bench_stderr_path.open("w", encoding="utf-8") as bench_stderr,
            ):
                completed = subprocess.run(
                    bench_command,
                    cwd=str(safe_cwd),
                    stdout=bench_stdout,
                    stderr=bench_stderr,
                    env=env,
                )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                raise RuntimeError(
                    "vLLM bench serve failed for profile "
                    f"{profile_name!r} with returncode={completed.returncode}; stderr={bench_stderr_path}"
                )
            if not raw_result_path.exists():
                raise FileNotFoundError(
                    f"vLLM bench serve did not write expected result file: {raw_result_path}"
                )
            raw_payload = load_json(raw_result_path)
            if isinstance(raw_payload, list):
                if not raw_payload:
                    raise ValueError(f"Empty vLLM bench serve result list in {raw_result_path}")
                raw_payload = raw_payload[-1]
            if not isinstance(raw_payload, dict):
                raise ValueError(f"Expected vLLM bench serve JSON object in {raw_result_path}")
            case = _normalize_bench_serve_case(
                profile=profile,
                raw_payload=raw_payload,
                command=bench_command,
                raw_output_path=raw_result_path,
                wall_time_seconds=elapsed,
            )
            case["stdout_path"] = str(bench_stdout_path)
            case["stderr_path"] = str(bench_stderr_path)
            cases.append(case)
    finally:
        _terminate_process(server_process)
        server_returncode = server_process.returncode
        server_stdout.close()
        server_stderr.close()

    progress.step(5, "Writing normalized online serving result")
    stats = _summarize_bench_serve_cases(cases, aggregation=aggregation)
    payload = build_result_payload(
        kind="speed",
        record=loaded.record,
        locator=loaded.locator,
        backend_name=str(speed_config.get("backend", "vllm_bench_serve")),
        backend_version=installed_vllm_version(),
        suite_path=suite_path,
        suite_name=str(suite_config.get("name", suite_path.stem)),
        config={
            "profiles": profiles,
            "metric_aggregation": aggregation,
            "metric_percentiles": percentiles,
            "vllm_bin": vllm_bin,
            "client_backend": client_backend,
            "host": host,
            "port": port,
            "endpoint": endpoint,
            "request_rate": request_rate,
            "num_prompts": num_prompts,
            "max_concurrency": max_concurrency,
            "num_warmups": num_warmups,
            "seed": seed,
            "random_range_ratio": random_range_ratio,
            "ignore_eos": ignore_eos,
            "ready_check_timeout_seconds": ready_check_timeout_seconds,
            "server_startup_grace_seconds": server_startup_grace_seconds,
            "server_log_level": server_log_level,
            "served_model_name": served_model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            "trust_remote_code": request.trust_remote_code,
            "local_files_only": request.local_files_only,
        },
        metrics=stats,
        artifacts={
            "raw_root": str(raw_root),
            "server_stdout_path": str(server_stdout_path),
            "server_stderr_path": str(server_stderr_path),
        },
        runtime={
            "model_path": prepared.model_path,
            "tokenizer_path": prepared.tokenizer_path,
            "tokenizer_mode": prepared.tokenizer_mode,
            "cuda_runtime": cuda_runtime,
            "model_impl": prepared.model_impl,
            "preparation_kind": prepared.preparation_kind,
            "source_model_path": prepared.source_model_path,
            "preparation_notes": prepared.notes,
            "server_command": server_command,
            "server_returncode_after_shutdown": server_returncode,
        },
        validation=validation_summary,
        details={"profiles": cases},
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


def run_evaluation_speed_suite(request: VllmSpeedRequest) -> VllmSpeedResult:
    suite_path = Path(request.suite_path)
    suite_config = load_yaml(suite_path)
    if suite_config.get("kind") != "speed":
        raise ValueError(f"Suite {suite_path} is not a speed suite.")

    speed_config = suite_config.get("speed", {})
    backend_name = str(speed_config.get("backend", "")).strip().lower()
    if backend_name not in {"evaluation", "eval"}:
        raise ValueError(f"Suite {suite_path} is not configured for the evaluation speed backend.")

    eval_suites_raw = list(speed_config.get("eval_suites") or [])
    if not eval_suites_raw:
        raise ValueError(f"Suite {suite_path} did not configure any speed.eval_suites.")
    eval_suite_paths = [resolve_suite_path(item) for item in eval_suites_raw]
    aggregation = str(speed_config.get("metric_aggregation", "macro_mean"))
    eval_output_root, eval_raw_output_root = _nested_eval_roots(request)
    lm_eval_bin = request.lm_eval_bin or str(speed_config.get("lm_eval_bin", DEFAULT_LM_EVAL_BIN))
    eval_model_backend = request.eval_model_backend if request.eval_model_backend is not None else speed_config.get("model_backend")
    eval_device = request.eval_device if request.eval_device is not None else speed_config.get("device")
    eval_batch_size = request.eval_batch_size if request.eval_batch_size is not None else speed_config.get("batch_size")
    eval_limit = request.eval_limit if request.eval_limit is not None else speed_config.get("limit")
    eval_num_fewshot = request.eval_num_fewshot if request.eval_num_fewshot is not None else speed_config.get("num_fewshot")
    eval_dtype = (
        normalize_dtype_name(request.dtype if request.dtype is not None else speed_config.get("dtype"))
        if request.dtype is not None or speed_config.get("dtype") is not None
        else None
    )
    eval_tensor_parallel_size = (
        request.tensor_parallel_size if request.tensor_parallel_size is not None else speed_config.get("tensor_parallel_size")
    )
    eval_gpu_memory_utilization = (
        request.gpu_memory_utilization
        if request.gpu_memory_utilization is not None
        else speed_config.get("gpu_memory_utilization")
    )
    eval_max_model_len = request.max_model_len if request.max_model_len is not None else speed_config.get("max_model_len")
    eval_enforce_eager = request.enforce_eager if request.enforce_eager is not None else speed_config.get("enforce_eager")
    cuda_runtime = describe_cuda_runtime(limit=eval_tensor_parallel_size)

    loaded = load_checkpoint(
        request.checkpoint_name,
        index_path=str(request.index_path),
        download=False,
        local_files_only=request.local_files_only,
        trust_remote_code=request.trust_remote_code,
    )

    suite_runs: list[dict[str, Any]] = []
    validations: dict[str, Any] = {}
    for eval_suite_path in eval_suite_paths:
        started = time.perf_counter()
        eval_result = run_lm_eval_suite(
            LmEvalRequest(
                checkpoint_name=request.checkpoint_name,
                suite_path=eval_suite_path,
                index_path=request.index_path,
                output_dir=eval_output_root,
                raw_output_root=eval_raw_output_root,
                lm_eval_bin=lm_eval_bin,
                model_backend=eval_model_backend,
                device=eval_device,
                batch_size=eval_batch_size,
                limit=eval_limit,
                num_fewshot=eval_num_fewshot,
                tensor_parallel_size=eval_tensor_parallel_size,
                gpu_memory_utilization=eval_gpu_memory_utilization,
                max_model_len=eval_max_model_len,
                enforce_eager=eval_enforce_eager,
                extra_model_args={"dtype": eval_dtype} if eval_dtype is not None else {},
                trust_remote_code=request.trust_remote_code,
                local_files_only=request.local_files_only,
                run_label=request.run_label,
                strict_validation=request.strict_validation,
            )
        )
        elapsed = time.perf_counter() - started
        suite_detail, validation = _summarize_nested_eval_run(
            eval_suite_path=eval_suite_path,
            wall_time_seconds=elapsed,
            result_output_path=eval_result.output_path,
            raw_output_path=eval_result.raw_output_path,
        )
        suite_runs.append(suite_detail)
        validations[suite_detail["suite"]] = validation

    metrics = _summarize_evaluation_speed_runs(suite_runs, aggregation=aggregation)
    payload = build_result_payload(
        kind="speed",
        record=loaded.record,
        locator=loaded.locator,
        backend_name=str(speed_config.get("backend", "evaluation")),
        backend_version=None,
        suite_path=suite_path,
        suite_name=str(suite_config.get("name", suite_path.stem)),
        config={
            "eval_suites": [suite_id(path) for path in eval_suite_paths],
            "metric_aggregation": aggregation,
            "lm_eval_bin": lm_eval_bin,
            "model_backend": eval_model_backend,
            "device": eval_device,
            "batch_size": eval_batch_size,
            "limit": eval_limit,
            "num_fewshot": eval_num_fewshot,
            "dtype": eval_dtype,
            "tensor_parallel_size": eval_tensor_parallel_size,
            "gpu_memory_utilization": eval_gpu_memory_utilization,
            "max_model_len": eval_max_model_len,
            "enforce_eager": eval_enforce_eager,
            "trust_remote_code": request.trust_remote_code,
            "local_files_only": request.local_files_only,
        },
        metrics=metrics,
        artifacts={
            "nested_output_root": str(eval_output_root),
            "nested_raw_output_root": str(eval_raw_output_root),
        },
        runtime={
            "suite_count": len(eval_suite_paths),
            "cuda_runtime": cuda_runtime,
        },
        validation={
            "suite_count": len(validations),
            "per_suite": validations,
        },
        details={
            "suites": suite_runs,
        },
        run_label=request.run_label,
        strict_validation=request.strict_validation,
    )
    output_path = dump_json(payload, _result_path_for(request))
    return VllmSpeedResult(
        checkpoint_name=request.checkpoint_name,
        suite=suite_output_name(suite_path),
        status="completed",
        output_path=str(output_path),
        stats=metrics,
    )


def run_speed_suite(request: VllmSpeedRequest) -> VllmSpeedResult:
    suite_path = Path(request.suite_path)
    suite_config = load_yaml(suite_path)
    if suite_config.get("kind") != "speed":
        raise ValueError(f"Suite {suite_path} is not a speed suite.")
    backend_name = str((suite_config.get("speed") or {}).get("backend", "vllm")).strip().lower()
    if backend_name == "vllm":
        return run_vllm_speed_suite(request)
    if backend_name == "vllm_bench_serve":
        return run_vllm_bench_serve_suite(request)
    if backend_name in {"evaluation", "eval"}:
        return run_evaluation_speed_suite(request)
    raise ValueError(f"Unsupported speed backend {backend_name!r} in {suite_path}.")
