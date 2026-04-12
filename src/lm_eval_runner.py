from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.benchmarking import suite_output_name
from src.load import CheckpointLoadError, LoadedCheckpoint, load_checkpoint
from src.result_schema import build_result_payload
from src.scoring import normalize_lm_eval_tasks
from src.utils import dump_json, ensure_dir, load_json, load_yaml, project_path


DEFAULT_LM_EVAL_BIN = os.environ.get("LRA_LM_EVAL_BIN", "lm-eval")


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
    extra_model_args: dict[str, Any] = field(default_factory=dict)


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


def _build_model_args(loaded: LoadedCheckpoint, extra_model_args: dict[str, Any]) -> dict[str, Any]:
    record = loaded.record
    if record.source == "local":
        model_args: dict[str, Any] = {"pretrained": loaded.local_path or loaded.locator}
    elif record.source == "huggingface":
        model_args = {"pretrained": record.repo_id}
        if record.revision:
            model_args["revision"] = record.revision
        if record.subpath:
            model_args["subfolder"] = record.subpath
    else:
        raise CheckpointLoadError(f"Unsupported checkpoint source for lm-eval: {record.source}")

    model_args.setdefault("dtype", "auto")
    model_args.update(extra_model_args)
    return model_args


def _build_command(
    request: LmEvalRequest,
    suite_config: dict[str, Any],
    loaded: LoadedCheckpoint,
    raw_output_dir: Path,
) -> list[str]:
    eval_config = suite_config.get("eval", {})
    tasks = [str(task) for task in eval_config.get("tasks", [])]
    if not tasks:
        raise ValueError(f"No lm-eval tasks configured in {request.suite_path}.")

    model_args = _build_model_args(loaded, request.extra_model_args)
    command: list[str] = [
        request.lm_eval_bin,
        "run",
        "--model",
        request.model_backend,
        "--model_args",
        *[f"{key}={_serialize_cli_value(value)}" for key, value in model_args.items()],
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

    return command


def _find_raw_result_path(raw_output_dir: Path) -> Path:
    results = sorted(raw_output_dir.rglob("results_*.json"))
    if not results:
        raise FileNotFoundError(f"No lm-eval results JSON found under {raw_output_dir}.")
    return results[-1]


def _result_path_for(request: LmEvalRequest) -> Path:
    output_root = Path(request.output_dir) if request.output_dir else project_path("results", "eval")
    ensure_dir(output_root)
    suite_name = suite_output_name(request.suite_path)
    return output_root / f"{suite_name}__{request.checkpoint_name}.json"


def _raw_output_root(request: LmEvalRequest) -> Path:
    root = Path(request.raw_output_root) if request.raw_output_root else project_path("results", "eval", "raw")
    return ensure_dir(root)


def run_lm_eval_suite(request: LmEvalRequest) -> LmEvalResult:
    suite_path = Path(request.suite_path)
    suite_config = load_yaml(suite_path)
    if suite_config.get("kind", "eval") != "eval":
        raise ValueError(f"Suite {suite_path} is not an eval suite.")

    loaded = load_checkpoint(request.checkpoint_name, index_path=str(request.index_path))
    raw_output_dir = _raw_output_root(request) / suite_output_name(suite_path) / request.checkpoint_name
    ensure_dir(raw_output_dir)

    command = _build_command(request, suite_config, loaded, raw_output_dir)
    subprocess.run(command, check=True, cwd=project_path())

    raw_result_path = _find_raw_result_path(raw_output_dir)
    raw_payload = load_json(raw_result_path)
    eval_config = suite_config.get("eval", {})
    preferred_metrics = [str(eval_config.get("metric"))] if eval_config.get("metric") else []
    preferred_metrics.extend([str(item) for item in eval_config.get("metric_fallbacks", [])])
    tasks, summary = normalize_lm_eval_tasks(raw_payload.get("results", {}), preferred_metrics=preferred_metrics)
    metrics = {
        "primary_metric": summary["primary_metric"],
        "mean": summary["mean"],
        "task_count": summary["task_count"],
        "scored_task_count": summary["scored_task_count"],
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
            "device": request.device if request.device is not None else eval_config.get("device"),
            "batch_size": request.batch_size if request.batch_size is not None else eval_config.get("batch_size"),
            "limit": request.limit if request.limit is not None else eval_config.get("limit"),
            "num_fewshot": request.num_fewshot if request.num_fewshot is not None else eval_config.get("num_fewshot"),
            "model_backend": request.model_backend,
        },
        metrics=metrics,
        artifacts={
            "command": shlex.join(command),
            "result_path": str(raw_result_path),
        },
        runtime={
            "config": raw_payload.get("config", {}),
            "n_samples": raw_payload.get("n-samples", {}),
        },
        details={
            "summary": summary,
            "tasks": tasks,
        },
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
