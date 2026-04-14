from __future__ import annotations

from src.ppl_runner import _build_contiguous_blocks, _summarize_tasks


def test_build_contiguous_blocks_drops_remainder() -> None:
    blocks = _build_contiguous_blocks(list(range(10)), max_length=4)

    assert blocks.tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_summarize_tasks_reports_macro_mean_ppl() -> None:
    summary = _summarize_tasks(
        [
            {"name": "wikitext2", "ppl": 10.0},
            {"name": "c4_stream", "ppl": 14.0},
        ],
        aggregation="macro_mean",
    )

    assert summary == {
        "primary_metric": "ppl",
        "mean": 12.0,
        "task_count": 2,
        "scored_task_count": 2,
        "aggregation": "macro_mean",
        "tracked_metrics": ["ppl"],
        "by_metric": {
            "ppl": {
                "wikitext2": 10.0,
                "c4_stream": 14.0,
            }
        },
    }
