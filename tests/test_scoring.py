from __future__ import annotations

from src.scoring import normalize_lm_eval_tasks


def test_normalize_lm_eval_tasks_records_acc_and_acc_norm() -> None:
    tasks, summary = normalize_lm_eval_tasks(
        {
            "task_a": {
                "acc,none": 0.25,
                "acc_stderr,none": 0.1,
                "acc_norm,none": 0.5,
                "acc_norm_stderr,none": 0.2,
            },
            "task_b": {
                "acc,none": 0.75,
                "acc_stderr,none": 0.3,
                "acc_norm,none": 0.25,
                "acc_norm_stderr,none": 0.4,
            },
        },
        preferred_metrics=["acc", "acc_norm"],
        aggregation="macro_mean",
        primary_metric=None,
    )

    assert summary["primary_metric"] is None
    assert summary["tracked_metrics"] == ["acc", "acc_norm"]
    assert summary["by_metric"]["acc"]["mean"] == 0.5
    assert summary["by_metric"]["acc_norm"]["mean"] == 0.375
    assert summary["mean"] is None
    assert summary["scored_task_count"] == 0

    assert tasks["task_a"]["tracked_metrics"]["acc"]["value"] == 0.25
    assert tasks["task_a"]["tracked_metrics"]["acc_norm"]["value"] == 0.5
    assert tasks["task_b"]["tracked_metrics"]["acc"]["value"] == 0.75
    assert tasks["task_b"]["tracked_metrics"]["acc_norm"]["value"] == 0.25


def test_normalize_lm_eval_tasks_uses_metric_fallbacks_for_headline_mean() -> None:
    tasks, summary = normalize_lm_eval_tasks(
        {
            "task_a": {
                "acc,none": 0.25,
                "acc_norm,none": 0.5,
            },
            "task_b": {
                "acc,none": 0.75,
            },
        },
        preferred_metrics=["acc_norm", "acc"],
        aggregation="macro_mean",
        primary_metric="acc_norm",
    )

    assert summary["primary_metric"] == "acc_norm"
    assert summary["by_metric"]["acc_norm"]["mean"] == 0.5
    assert summary["by_metric"]["acc"]["mean"] == 0.5
    assert summary["mean"] == 0.625
    assert summary["scored_task_count"] == 2
    assert tasks["task_a"]["metric"] == "acc_norm"
    assert tasks["task_b"]["metric"] == "acc"


def test_normalize_lm_eval_tasks_preserves_custom_filter_metrics() -> None:
    tasks, summary = normalize_lm_eval_tasks(
        {
            "mmlu_pro": {
                "exact_match,custom-extract": 0.42,
                "exact_match_stderr,custom-extract": 0.01,
            }
        },
        preferred_metrics=["exact_match"],
        aggregation="macro_mean",
        primary_metric="exact_match",
    )

    assert summary["primary_metric"] == "exact_match"
    assert summary["mean"] == 0.42
    assert summary["scored_task_count"] == 1
    assert tasks["mmlu_pro"]["filter"] == "custom-extract"
    assert tasks["mmlu_pro"]["tracked_metrics"]["exact_match"]["value"] == 0.42
