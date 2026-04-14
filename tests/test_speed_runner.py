from __future__ import annotations

from pathlib import Path

from src.speed_runner import VllmSpeedRequest, _resolve_speed_case_specs


def test_resolve_speed_case_specs_prefers_named_suite_cases() -> None:
    speed_config = {
        "cases": [
            {
                "name": "interactive_short",
                "batch_size": 1,
                "prompt_length": 512,
                "generation_length": 128,
            },
            {
                "name": "throughput_long_decode",
                "batch_size": 4,
                "prompt_length": 512,
                "generation_length": 512,
            },
        ]
    }
    request = VllmSpeedRequest(
        checkpoint_name="demo",
        suite_path=Path("benchmark/speed/serve.yaml"),
        index_path=Path("checkpoints/index.csv"),
    )

    cases = _resolve_speed_case_specs(speed_config, request)

    assert cases == [
        {
            "name": "interactive_short",
            "batch_size": 1,
            "prompt_length": 512,
            "generation_length": 128,
        },
        {
            "name": "throughput_long_decode",
            "batch_size": 4,
            "prompt_length": 512,
            "generation_length": 512,
        },
    ]


def test_resolve_speed_case_specs_lets_cli_axes_override_named_cases() -> None:
    speed_config = {
        "cases": [
            {
                "name": "interactive_short",
                "batch_size": 1,
                "prompt_length": 512,
                "generation_length": 128,
            }
        ],
        "batch_sizes": [1],
        "prompt_lengths": [512],
        "generation_lengths": [128],
    }
    request = VllmSpeedRequest(
        checkpoint_name="demo",
        suite_path=Path("benchmark/speed/serve.yaml"),
        index_path=Path("checkpoints/index.csv"),
        batch_sizes=[4],
        prompt_lengths=[256],
        generation_lengths=[64],
    )

    cases = _resolve_speed_case_specs(speed_config, request)

    assert cases == [
        {
            "name": "batch4_prompt256_gen64",
            "batch_size": 4,
            "prompt_length": 256,
            "generation_length": 64,
        }
    ]
