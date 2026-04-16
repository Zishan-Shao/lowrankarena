from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.benchmarking import suite_output_name
from src.inference_adapter import PreparedInferenceModel, prepare_model_for_inference
from src.load import load_checkpoint
from src.result_schema import build_result_payload
from src.scoring import normalize_lm_eval_tasks
from src.utils import dump_json, ensure_dir, load_json, load_yaml, project_path, run_results_root
from src.validation import validate_checkpoint_layout


DEFAULT_LM_EVAL_BIN = os.environ.get("LRA_LM_EVAL_BIN", "lm-eval")
TOKENIZER_INFINITY_THRESHOLD = 10**12
eval_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LmEvalRequest:
    checkpoint_name: str
    suite_path: str | Path
    index_path: str | Path
    output_dir: str | Path | None = None
    raw_output_root: str | Path | None = None
    lm_eval_bin: str = DEFAULT_LM_EVAL_BIN
    model_backend: str = "hf"
    device: str | None = None
    batch_size: str | int | None = None
    limit: float | int | None = None
    num_fewshot: int | None = None
    log_samples: bool = False
    use_cache: str | None = None
    trust_remote_code: bool = True
    local_files_only: bool = False
    extra_model_args: dict[str, Any] = field(default_factory=dict)
    run_label: str = "ad_hoc"
    strict_validation: bool = False


@dataclass(slots=True)
class LmEvalResult:
    checkpoint_name: str
    suite: str
    status: str
    output_path: str
    raw_output_path: str
    metrics: dict[str, Any]


def _serialize_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _serialize_cli_pairs(payload: dict[str, Any]) -> list[str]:
    return [f"{key}={_serialize_cli_value(value)}" for key, value in payload.items()]


def _build_model_args(
    prepared: PreparedInferenceModel,
    extra_model_args: dict[str, Any],
    *,
    default_dtype: str = "auto",
) -> dict[str, Any]:
    model_args: dict[str, Any] = {"pretrained": prepared.model_path}
    model_args.setdefault("dtype", default_dtype)
    if prepared.tokenizer_path != prepared.model_path:
        model_args["tokenizer"] = prepared.tokenizer_path
    if prepared.tokenizer_mode == "slow":
        model_args["use_fast_tokenizer"] = False
    model_args.update(extra_model_args)
    return model_args


def _resolve_include_paths(suite_path: str | Path, raw_paths: Any) -> list[Path]:
    if raw_paths is None:
        return []
    if isinstance(raw_paths, (str, Path)):
        items = [raw_paths]
    else:
        items = list(raw_paths)

    resolved: list[Path] = []
    for item in items:
        candidate = Path(str(item))
        if not candidate.is_absolute():
            candidate = (project_path() / candidate).resolve()
        resolved.append(candidate)
    return resolved


def _coerce_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    coerced = int(value)
    if coerced <= 0:
        raise ValueError(f"Expected a positive integer, got {value!r}.")
    return coerced


def _infer_model_max_length(prepared: PreparedInferenceModel) -> int | None:
    search_paths = [Path(prepared.model_path)]
    tokenizer_path = Path(prepared.tokenizer_path)
    if tokenizer_path not in search_paths:
        search_paths.append(tokenizer_path)

    config_attrs = (
        "n_positions",
        "max_position_embeddings",
        "n_ctx",
        "model_max_length",
        "max_sequence_length",
        "max_seq_len",
        "seq_length",
    )
    for base_path in search_paths:
        config_path = base_path / "config.json"
        if not config_path.exists():
            continue
        config = load_json(config_path)
        for attr in config_attrs:
            value = _coerce_positive_int(config.get(attr))
            if value is not None:
                return value

    for base_path in search_paths:
        tokenizer_config_path = base_path / "tokenizer_config.json"
        if not tokenizer_config_path.exists():
            continue
        tokenizer_config = load_json(tokenizer_config_path)
        value = _coerce_positive_int(tokenizer_config.get("model_max_length"))
        if value is not None and value < TOKENIZER_INFINITY_THRESHOLD:
            return value

    return None


def _effective_gen_kwargs(
    eval_config: dict[str, Any],
    *,
    prepared: PreparedInferenceModel,
    suite_path: str | Path,
) -> dict[str, Any]:
    raw_gen_kwargs = eval_config.get("gen_kwargs") or {}
    if not raw_gen_kwargs:
        return {}
    if not isinstance(raw_gen_kwargs, dict):
        raise TypeError(f"eval.gen_kwargs must be a mapping in {suite_path}.")

    gen_kwargs = dict(raw_gen_kwargs)
    max_gen_toks = _coerce_positive_int(gen_kwargs.get("max_gen_toks"))
    model_max_length = _infer_model_max_length(prepared)
    if (
        max_gen_toks is not None
        and model_max_length is not None
        and max_gen_toks >= model_max_length
    ):
        clamped = model_max_length - 1
        eval_logger.warning(
            "Clamping eval.gen_kwargs.max_gen_toks from %s to %s for %s because model_max_length is %s.",
            max_gen_toks,
            clamped,
            suite_path,
            model_max_length,
        )
        gen_kwargs["max_gen_toks"] = clamped
    return gen_kwargs


def _normalize_summary_entries(
    raw_entries: dict[str, dict[str, Any]],
    *,
    preferred_metrics: list[str],
    aggregation: str,
    primary_metric: str | None,
    entity: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    selected = raw_entries
    if entity:
        if entity not in raw_entries:
            known = ", ".join(sorted(raw_entries))
            raise KeyError(f"Unknown lm-eval summary entity {entity!r}. Known entities: {known}")
        selected = {entity: raw_entries[entity]}
    return normalize_lm_eval_tasks(
        selected,
        preferred_metrics=preferred_metrics,
        aggregation=aggregation,
        primary_metric=primary_metric,
    )


def _build_command(
    request: LmEvalRequest,
    suite_config: dict[str, Any],
    raw_output_dir: Path,
    prepared: PreparedInferenceModel,
) -> list[str]:
    eval_config = suite_config.get("eval", {})
    tasks = [str(task) for task in eval_config.get("tasks", [])]
    if not tasks:
        raise ValueError(f"No lm-eval tasks configured in {request.suite_path}.")

    model_args = _build_model_args(
        prepared,
        request.extra_model_args,
        default_dtype=str(eval_config.get("dtype", "auto")),
    )
    command: list[str] = [
        request.lm_eval_bin,
        "run",
        "--model",
        request.model_backend,
        "--model_args",
        *_serialize_cli_pairs(model_args),
        "--tasks",
        *tasks,
        "--output_path",
        str(raw_output_dir),
    ]

    batch_size = request.batch_size if request.batch_size is not None else eval_config.get("batch_size")
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])

    device = request.device if request.device is not None else eval_config.get("device")
    if device:
        command.extend(["--device", str(device)])

    limit = request.limit if request.limit is not None else eval_config.get("limit")
    if limit is not None:
        command.extend(["--limit", str(limit)])

    fewshot = request.num_fewshot if request.num_fewshot is not None else eval_config.get("num_fewshot")
    if fewshot is not None:
        command.extend(["--num_fewshot", str(fewshot)])

    if request.use_cache:
        command.extend(["--use_cache", request.use_cache])
    if request.log_samples:
        command.append("--log_samples")
    if request.trust_remote_code:
        command.append("--trust_remote_code")

    include_paths = _resolve_include_paths(request.suite_path, eval_config.get("include_paths"))
    for include_path in include_paths:
        command.extend(["--include_path", str(include_path)])

    gen_kwargs = _effective_gen_kwargs(
        eval_config,
        prepared=prepared,
        suite_path=request.suite_path,
    )
    if gen_kwargs:
        command.extend(["--gen_kwargs", *_serialize_cli_pairs(gen_kwargs)])

    apply_chat_template = eval_config.get("apply_chat_template")
    if isinstance(apply_chat_template, str) and apply_chat_template.strip():
        command.extend(["--apply_chat_template", apply_chat_template.strip()])
    elif apply_chat_template:
        command.append("--apply_chat_template")

    if "fewshot_as_multiturn" in eval_config:
        command.extend(["--fewshot_as_multiturn", _serialize_cli_value(eval_config.get("fewshot_as_multiturn"))])

    system_instruction = eval_config.get("system_instruction")
    if system_instruction:
        command.extend(["--system_instruction", str(system_instruction)])

    return command


def _find_raw_result_path(raw_output_dir: Path) -> Path:
    results = sorted(raw_output_dir.rglob("results_*.json"))
    if not results:
        raise FileNotFoundError(f"No lm-eval results JSON found under {raw_output_dir}.")
    return results[-1]


def _result_path_for(request: LmEvalRequest) -> Path:
    output_root = Path(request.output_dir) if request.output_dir else run_results_root("eval", request.run_label)
    ensure_dir(output_root)
    suite_name = suite_output_name(request.suite_path)
    return output_root / f"{suite_name}__{request.checkpoint_name}.json"


def _raw_output_root(request: LmEvalRequest) -> Path:
    root = (
        Path(request.raw_output_root)
        if request.raw_output_root
        else run_results_root("eval", request.run_label) / "raw"
    )
    return ensure_dir(root)


def run_lm_eval_suite(request: LmEvalRequest) -> LmEvalResult:
    suite_path = Path(request.suite_path)
    suite_config = load_yaml(suite_path)
    if suite_config.get("kind", "eval") != "eval":
        raise ValueError(f"Suite {suite_path} is not an eval suite.")
    backend_name = str((suite_config.get("eval") or {}).get("backend", "lm_eval_harness"))
    if backend_name == "contiguous_ppl":
        from src.ppl_runner import run_contiguous_ppl_suite

        return run_contiguous_ppl_suite(request, suite_path=suite_path, suite_config=suite_config)
    if backend_name != "lm_eval_harness":
        raise ValueError(f"Unsupported eval backend {backend_name!r} in {suite_path}.")

    loaded = load_checkpoint(
        request.checkpoint_name,
        index_path=str(request.index_path),
        download=True,
        local_files_only=request.local_files_only,
        trust_remote_code=request.trust_remote_code,
    )
    prepared = prepare_model_for_inference(loaded)
    validation_summary = validate_checkpoint_layout(
        prepared.model_path,
        strict=request.strict_validation,
    )
    raw_output_dir = _raw_output_root(request) / suite_output_name(suite_path) / request.checkpoint_name
    ensure_dir(raw_output_dir)

    command = _build_command(request, suite_config, raw_output_dir, prepared)
    subprocess.run(command, check=True, cwd=project_path())

    raw_result_path = _find_raw_result_path(raw_output_dir)
    raw_payload = load_json(raw_result_path)
    eval_config = suite_config.get("eval", {})
    effective_gen_kwargs = _effective_gen_kwargs(
        eval_config,
        prepared=prepared,
        suite_path=suite_path,
    )
    tracked_metrics = [str(item) for item in eval_config.get("tracked_metrics", []) if str(item).strip()]
    preferred_metrics = [str(eval_config.get("metric"))] if eval_config.get("metric") else []
    preferred_metrics.extend([str(item) for item in eval_config.get("metric_fallbacks", [])])
    if not preferred_metrics:
        preferred_metrics = list(tracked_metrics)
    for metric_name in tracked_metrics:
        if metric_name not in preferred_metrics:
            preferred_metrics.append(metric_name)
    aggregation = str(eval_config.get("metric_aggregation", "macro_mean"))
    primary_metric = str(eval_config.get("metric")) if eval_config.get("metric") else None
    raw_results = raw_payload.get("results", {})
    raw_groups = raw_payload.get("groups", {})
    tasks, task_summary = _normalize_summary_entries(
        raw_results,
        preferred_metrics=preferred_metrics,
        aggregation=aggregation,
        primary_metric=primary_metric,
    )
    groups: dict[str, dict[str, Any]] = {}
    group_summary: dict[str, Any] | None = None
    if raw_groups:
        groups, group_summary = _normalize_summary_entries(
            raw_groups,
            preferred_metrics=preferred_metrics,
            aggregation=aggregation,
            primary_metric=primary_metric,
        )

    summary_source = str(eval_config.get("summary_source", "results")).strip().lower()
    summary_entity = str(eval_config.get("summary_entity", "")).strip() or None
    if summary_source == "results":
        _, summary = _normalize_summary_entries(
            raw_results,
            preferred_metrics=preferred_metrics,
            aggregation=aggregation,
            primary_metric=primary_metric,
            entity=summary_entity,
        )
    elif summary_source == "groups":
        if not raw_groups:
            raise ValueError(f"Suite {suite_path} requested groups summary_source but lm-eval returned no groups.")
        _, summary = _normalize_summary_entries(
            raw_groups,
            preferred_metrics=preferred_metrics,
            aggregation=aggregation,
            primary_metric=primary_metric,
            entity=summary_entity,
        )
    else:
        raise ValueError(f"Unsupported eval.summary_source={summary_source!r} in {suite_path}.")

    metrics = {
        "primary_metric": summary["primary_metric"],
        "mean": summary["mean"],
        "task_count": summary["task_count"],
        "scored_task_count": summary["scored_task_count"],
        "aggregation": summary["aggregation"],
        "tracked_metrics": summary["tracked_metrics"],
        "by_metric": summary["by_metric"],
        "summary_source": summary_source,
        "summary_entity": summary_entity,
    }

    payload = build_result_payload(
        kind="eval",
        record=loaded.record,
        locator=loaded.locator,
        backend_name=eval_config.get("backend", "lm_eval_harness"),
        backend_version=raw_payload.get("lm_eval_version"),
        suite_path=suite_path,
        suite_name=suite_config.get("name", suite_path.stem),
        config={
            "tasks": [str(task) for task in eval_config.get("tasks", [])],
            "metric": eval_config.get("metric"),
            "metric_fallbacks": [str(item) for item in eval_config.get("metric_fallbacks", [])],
            "tracked_metrics": tracked_metrics,
            "metric_aggregation": aggregation,
            "dtype": str(eval_config.get("dtype", "auto")),
            "device": request.device if request.device is not None else eval_config.get("device"),
            "batch_size": request.batch_size if request.batch_size is not None else eval_config.get("batch_size"),
            "limit": request.limit if request.limit is not None else eval_config.get("limit"),
            "num_fewshot": request.num_fewshot if request.num_fewshot is not None else eval_config.get("num_fewshot"),
            "model_backend": request.model_backend,
            "gen_kwargs": effective_gen_kwargs,
            "include_paths": [str(path) for path in _resolve_include_paths(suite_path, eval_config.get("include_paths"))],
            "apply_chat_template": eval_config.get("apply_chat_template", False),
            "fewshot_as_multiturn": eval_config.get("fewshot_as_multiturn"),
            "system_instruction": eval_config.get("system_instruction"),
            "summary_source": summary_source,
            "summary_entity": summary_entity,
        },
        metrics=metrics,
        artifacts={
            "command": shlex.join(command),
            "result_path": str(raw_result_path),
        },
        runtime={
            "config": raw_payload.get("config", {}),
            "n_samples": raw_payload.get("n-samples", {}),
            "model_path": prepared.model_path,
            "tokenizer_path": prepared.tokenizer_path,
            "tokenizer_mode": prepared.tokenizer_mode,
            "preparation_kind": prepared.preparation_kind,
            "source_model_path": prepared.source_model_path,
            "preparation_notes": prepared.notes,
        },
        validation=validation_summary,
        details={
            "summary": summary,
            "task_summary": task_summary,
            "group_summary": group_summary,
            "groups": groups,
            "tasks": tasks,
        },
        run_label=request.run_label,
        strict_validation=request.strict_validation,
    )
    output_path = dump_json(payload, _result_path_for(request))
    return LmEvalResult(
        checkpoint_name=request.checkpoint_name,
        suite=suite_output_name(suite_path),
        status="completed",
        output_path=str(output_path),
        raw_output_path=str(raw_result_path),
        metrics=metrics,
    )
