from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
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
        if metric_name.endswith("_stderr"):
            continue
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


def collect_tracked_task_metrics(
    task_result: dict[str, Any],
    tracked_metrics: list[str],
) -> dict[str, dict[str, Any]]:
    available = iter_numeric_task_metrics(task_result)
    collected: dict[str, dict[str, Any]] = {}
    for metric_name in tracked_metrics:
        chosen = next((metric for metric in available if metric.metric == metric_name), None)
        collected[metric_name] = {
            "metric": chosen.metric if chosen else metric_name,
            "metric_key": chosen.metric_key if chosen else None,
            "filter": chosen.filter_name if chosen else None,
            "value": chosen.value if chosen else None,
            "stderr": chosen.stderr if chosen else None,
        }
    return collected


def normalize_lm_eval_tasks(
    raw_results: dict[str, dict[str, Any]],
    preferred_metrics: list[str],
    aggregation: str = "macro_mean",
    primary_metric: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    unique_metrics: list[str] = []
    for metric_name in preferred_metrics:
        if metric_name and metric_name not in unique_metrics:
            unique_metrics.append(metric_name)
    aggregated_values: dict[str, list[float]] = {metric_name: [] for metric_name in unique_metrics}

    for task_name, task_result in raw_results.items():
        chosen = choose_task_metric(task_result, preferred_metrics)
        tracked = collect_tracked_task_metrics(task_result, unique_metrics)
        available_metrics = [metric.metric for metric in iter_numeric_task_metrics(task_result)]
        normalized[task_name] = {
            "metric": chosen.metric if chosen else None,
            "metric_key": chosen.metric_key if chosen else None,
            "filter": chosen.filter_name if chosen else None,
            "value": chosen.value if chosen else None,
            "stderr": chosen.stderr if chosen else None,
            "tracked_metrics": tracked,
            "available_metrics": available_metrics,
        }
        for metric_name, metric_payload in tracked.items():
            value = metric_payload.get("value")
            if value is not None:
                aggregated_values[metric_name].append(float(value))

    by_metric = {
        metric_name: {
            "mean": aggregate_values(values, aggregation=aggregation),
            "scored_task_count": len(values),
        }
        for metric_name, values in aggregated_values.items()
    }
    resolved_primary_metric = primary_metric if primary_metric in by_metric else None
    primary_summary = by_metric.get(resolved_primary_metric or "")
    summary = {
        "primary_metric": resolved_primary_metric,
        "aggregation": aggregation,
        "resolved_metrics": sorted({item["metric"] for item in normalized.values() if item["metric"]}),
        "tracked_metrics": unique_metrics,
        "by_metric": by_metric,
        "mean": primary_summary["mean"] if primary_summary else None,
        "task_count": len(normalized),
        "scored_task_count": primary_summary["scored_task_count"] if primary_summary else 0,
    }
    return normalized, summary


def aggregate_values(values: list[float], aggregation: str) -> float | None:
    if not values:
        return None
    normalized = aggregation.strip().lower()
    if normalized == "macro_mean":
        return mean(values)
    if normalized == "macro_geometric_mean":
        if any(value <= 0 for value in values):
            raise ValueError("macro_geometric_mean requires strictly positive values.")
        return exp(mean(log(value) for value in values))
    raise ValueError(f"Unsupported aggregation rule: {aggregation}")


def summarize_speed_cases(cases: list[dict[str, Any]], *, aggregation: str = "macro_mean") -> dict[str, Any]:
    if not cases:
        return {
            "case_count": 0,
            "aggregation": aggregation,
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
        "aggregation": aggregation,
        "mean_prefill_tokens_per_second": aggregate_values(prefill_values, aggregation=aggregation),
        "mean_decode_tokens_per_second": aggregate_values(decode_values, aggregation=aggregation),
        "mean_end_to_end_tokens_per_second": aggregate_values(end_to_end_values, aggregation=aggregation),
        "mean_latency_seconds": aggregate_values(latency_values, aggregation=aggregation),
    }
