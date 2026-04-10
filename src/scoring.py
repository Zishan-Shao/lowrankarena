from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any


@dataclass(slots=True)
class TaskMetric:
    metric: str
    metric_key: str
    filter_name: str
    value: float
    stderr: float | str | None


def _parse_metric_key(key: str) -> tuple[str, str]:
    metric_name, separator, filter_name = key.partition(",")
    return metric_name, filter_name if separator else "default"


def iter_numeric_task_metrics(task_result: dict[str, Any]) -> list[TaskMetric]:
    metrics: list[TaskMetric] = []
    for key, value in task_result.items():
        if key == "alias" or key.endswith("_stderr"):
            continue
        if not isinstance(value, (int, float)):
            continue
        metric_name, filter_name = _parse_metric_key(key)
        stderr_key = f"{metric_name}_stderr,{filter_name}" if filter_name != "default" else f"{metric_name}_stderr"
        metrics.append(
            TaskMetric(
                metric=metric_name,
                metric_key=key,
                filter_name=filter_name,
                value=float(value),
                stderr=task_result.get(stderr_key),
            )
        )
    return metrics


def choose_task_metric(task_result: dict[str, Any], preferred_metrics: list[str]) -> TaskMetric | None:
    available = iter_numeric_task_metrics(task_result)
    for preferred in preferred_metrics:
        for metric in available:
            if metric.metric == preferred:
                return metric
    if available:
        return available[0]
    return None


def normalize_lm_eval_tasks(
    raw_results: dict[str, dict[str, Any]],
    preferred_metrics: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    chosen_values: list[float] = []

    for task_name, task_result in raw_results.items():
        chosen = choose_task_metric(task_result, preferred_metrics)
        available_metrics = [metric.metric for metric in iter_numeric_task_metrics(task_result)]
        normalized[task_name] = {
            "metric": chosen.metric if chosen else None,
            "metric_key": chosen.metric_key if chosen else None,
            "filter": chosen.filter_name if chosen else None,
            "value": chosen.value if chosen else None,
            "stderr": chosen.stderr if chosen else None,
            "available_metrics": available_metrics,
        }
        if chosen is not None:
            chosen_values.append(chosen.value)

    primary_metric = preferred_metrics[0] if preferred_metrics else None
    summary = {
        "primary_metric": primary_metric,
        "resolved_metrics": sorted({item["metric"] for item in normalized.values() if item["metric"]}),
        "mean": mean(chosen_values) if chosen_values else None,
        "task_count": len(normalized),
        "scored_task_count": len(chosen_values),
    }
    return normalized, summary


def summarize_speed_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {
            "case_count": 0,
            "mean_prefill_tokens_per_second": None,
            "mean_decode_tokens_per_second": None,
            "mean_end_to_end_tokens_per_second": None,
            "mean_latency_seconds": None,
        }

    def collect(name: str) -> list[float]:
        return [float(case[name]) for case in cases if case.get(name) is not None]

    prefill_values = collect("prefill_tokens_per_second")
    decode_values = collect("decode_tokens_per_second")
    end_to_end_values = collect("end_to_end_tokens_per_second")
    latency_values = collect("latency_seconds")

    return {
        "case_count": len(cases),
        "mean_prefill_tokens_per_second": mean(prefill_values) if prefill_values else None,
        "mean_decode_tokens_per_second": mean(decode_values) if decode_values else None,
        "mean_end_to_end_tokens_per_second": mean(end_to_end_values) if end_to_end_values else None,
        "mean_latency_seconds": mean(latency_values) if latency_values else None,
    }
