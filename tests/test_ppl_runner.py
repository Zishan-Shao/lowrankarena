from __future__ import annotations

import src.ppl_runner as ppl_runner
from src.ppl_runner import (
    _build_contiguous_blocks,
    _hash_token_ids,
    _load_c4_stream_token_ids,
    _summarize_tasks,
)


def test_build_contiguous_blocks_drops_remainder() -> None:
    blocks = _build_contiguous_blocks(list(range(10)), max_length=4)

    assert blocks.tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_hash_token_ids_is_stable() -> None:
    assert _hash_token_ids([0, 1, 2, 2048]) == _hash_token_ids([0, 1, 2, 2048])
    assert _hash_token_ids([0, 1, 2, 2048]) != _hash_token_ids([0, 1, 2, 2049])


def test_c4_stream_loader_uses_revision_and_document_offset() -> None:
    calls = []

    def fake_load_dataset(path, name, *, split, streaming, cache_dir, revision):
        calls.append(
            {
                "path": path,
                "name": name,
                "split": split,
                "streaming": streaming,
                "cache_dir": cache_dir,
                "revision": revision,
            }
        )
        return [
            {"text": "skip-me"},
            {"text": "first"},
            {"text": "second"},
        ]

    class ToyTokenizer:
        def __call__(
            self,
            text,
            *,
            add_special_tokens,
            return_attention_mask,
            return_token_type_ids,
        ):
            del add_special_tokens, return_attention_mask, return_token_type_ids
            return {
                "input_ids": {
                    "\n\n": [99],
                    "first": [1, 2],
                    "second": [3, 4],
                }[text]
            }

    original_load_dataset = ppl_runner.load_dataset
    ppl_runner.load_dataset = fake_load_dataset
    try:
        token_ids, docs_scanned = _load_c4_stream_token_ids(
            ToyTokenizer(),
            split="validation",
            max_eval_tokens=4,
            max_length=2,
            cache_dir="/tmp/cache",
            revision="dataset-revision",
            dataset_path="fixed/c4",
            dataset_name="en",
            document_offset=1,
        )
    finally:
        ppl_runner.load_dataset = original_load_dataset

    assert calls == [
        {
            "path": "fixed/c4",
            "name": "en",
            "split": "validation",
            "streaming": True,
            "cache_dir": "/tmp/cache",
            "revision": "dataset-revision",
        }
    ]
    assert token_ids == [1, 2, 99, 3]
    assert docs_scanned == 2


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
