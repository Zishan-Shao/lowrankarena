from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import load_suite_config, select_checkpoints_for_suite, suite_id
from src.lm_eval_runner import LmEvalRequest, run_lm_eval_suite
from src.speed_runner import VllmSpeedRequest, run_vllm_speed_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggregate benchmark suites from benchmark/**/*.yaml.")
    parser.add_argument("--suites", nargs="*", default=["main"], help="Suite names such as main, accuracy/mcq, or speed/speed.")
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval limit override for smoke runs.")
    parser.add_argument("--lm-eval-bin", default="lm-eval")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--speed-repeat", type=int, default=None)
    parser.add_argument("--speed-warmup", type=int, default=None)
    parser.add_argument("--speed-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--speed-max-model-len", type=int, default=None)
    parser.add_argument("--speed-enforce-eager", action="store_true")
    return parser.parse_args()


def run_suite(suite_name: str, index_path: str, args: argparse.Namespace) -> list[dict[str, str]]:
    config_path, config = load_suite_config(suite_name)
    kind = config.get("kind", "eval")
    suite_name_resolved = suite_id(config_path)

    if kind == "aggregate":
        outputs: list[dict[str, str]] = []
        for included in config.get("includes", []):
            outputs.extend(run_suite(str(included), index_path=index_path, args=args))
        return outputs

    outputs: list[dict[str, str]] = []
    for record in select_checkpoints_for_suite(config, index_path=index_path):
        if kind == "speed":
            result = run_vllm_speed_suite(
                VllmSpeedRequest(
                    checkpoint_name=record.name,
                    suite_path=config_path,
                    index_path=index_path,
                    repeat=args.speed_repeat,
                    warmup=args.speed_warmup,
                    gpu_memory_utilization=args.speed_gpu_memory_utilization,
                    max_model_len=args.speed_max_model_len,
                    enforce_eager=True if args.speed_enforce_eager else None,
                    show_progress=True,
                )
            )
        else:
            result = run_lm_eval_suite(
                LmEvalRequest(
                    checkpoint_name=record.name,
                    suite_path=config_path,
                    index_path=index_path,
                    lm_eval_bin=args.lm_eval_bin,
                    device=args.eval_device,
                    limit=args.limit,
                )
            )
        outputs.append(
            {
                "suite": suite_name_resolved,
                "checkpoint": record.name,
                "status": result.status,
                "output_path": result.output_path,
            }
        )
    return outputs


def main() -> None:
    args = parse_args()
    summary: list[dict[str, str]] = []
    for suite_name in args.suites:
        summary.extend(run_suite(suite_name, index_path=args.index, args=args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
