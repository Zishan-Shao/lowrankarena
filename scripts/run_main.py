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
from src.memory_runner import MemoryRequest, run_memory_measurement
from src.speed_runner import VllmSpeedRequest, run_speed_suite
from src.utils import dump_json, ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggregate benchmark suites from benchmark/**/*.yaml.")
    parser.add_argument("--suites", nargs="*", default=["main"], help="Suite names such as main, accuracy/mcq, or speed/serve.")
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval limit override for smoke runs.")
    parser.add_argument("--lm-eval-bin", default="lm-eval")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--speed-repeat", type=int, default=None)
    parser.add_argument("--speed-warmup", type=int, default=None)
    parser.add_argument("--speed-batch-size", type=int, action="append", default=None)
    parser.add_argument("--speed-prompt-length", type=int, action="append", default=None)
    parser.add_argument("--speed-generation-length", type=int, action="append", default=None)
    parser.add_argument("--speed-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--speed-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--speed-dtype", default=None)
    parser.add_argument("--speed-max-model-len", type=int, default=None)
    parser.add_argument("--speed-enforce-eager", action="store_true")
    parser.add_argument("--memory-device", default="cuda:0")
    parser.add_argument("--memory-dtype", default=None)
    parser.add_argument("--memory-batch-size", type=int, default=None)
    parser.add_argument("--memory-prompt-length", type=int, default=None)
    parser.add_argument("--memory-generation-length", type=int, default=None)
    parser.add_argument("--memory-attn-implementation", default=None)
    return parser.parse_args()


def _normalize_precision(value: object) -> str | None:
    if value is None:
        return None
    return str(value).replace("torch.", "").strip().lower()


def _precision_audit_path(suite_names: list[str]) -> Path:
    audit_root = ensure_dir(ROOT / "results" / "leaderboard")
    label = "__".join(str(item).replace("/", "__") for item in suite_names) or "leaderboard"
    return audit_root / f"precision_audit__{label}.json"


def _write_precision_audit(summary: list[dict[str, str]], suite_names: list[str]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    runtime_precisions: set[str] = set()
    missing_runtime_precision: list[str] = []
    mixed_checkpoint_precision: list[str] = []

    for item in summary:
        payload = load_json(item["output_path"])
        config = payload.get("config", {})
        validation = payload.get("validation", {})
        precision = validation.get("precision", {})
        runtime_precision = _normalize_precision(config.get("dtype"))
        factor_dtypes = [_normalize_precision(value) for value in precision.get("low_rank_factor_dtypes", []) if value is not None]
        factor_dtypes = [value for value in factor_dtypes if value]
        if runtime_precision is None:
            missing_runtime_precision.append(item["output_path"])
        else:
            runtime_precisions.add(runtime_precision)
        if len(set(factor_dtypes)) > 1:
            mixed_checkpoint_precision.append(item["output_path"])

        records.append(
            {
                "checkpoint": payload.get("checkpoint", {}).get("name"),
                "suite": payload.get("suite", {}).get("id"),
                "output_path": item["output_path"],
                "layout_kind": validation.get("layout_kind"),
                "runtime_precision": runtime_precision,
                "low_rank_factor_dtypes": factor_dtypes,
                "uniform_low_rank_precision": precision.get("uniform_low_rank_precision"),
                "matches_reference_torch_dtype": precision.get("matches_reference_torch_dtype"),
            }
        )

    audit = {
        "suite_names": suite_names,
        "passed": not missing_runtime_precision and len(runtime_precisions) <= 1 and not mixed_checkpoint_precision,
        "runtime_precisions": sorted(runtime_precisions),
        "missing_runtime_precision": missing_runtime_precision,
        "mixed_checkpoint_precision": mixed_checkpoint_precision,
        "records": records,
    }
    dump_json(audit, _precision_audit_path(suite_names))
    if missing_runtime_precision:
        raise RuntimeError("Leaderboard precision audit found results without an explicit runtime dtype.")
    if len(runtime_precisions) > 1:
        raise RuntimeError(
            "Leaderboard precision audit found mixed runtime dtypes: " + ", ".join(sorted(runtime_precisions))
        )
    if mixed_checkpoint_precision:
        raise RuntimeError(
            "Leaderboard precision audit found checkpoints with mixed low-rank factor dtypes: "
            + ", ".join(mixed_checkpoint_precision)
        )
    return audit


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
            result = run_speed_suite(
                VllmSpeedRequest(
                    checkpoint_name=record.name,
                    suite_path=config_path,
                    index_path=index_path,
                    batch_sizes=args.speed_batch_size,
                    prompt_lengths=args.speed_prompt_length,
                    generation_lengths=args.speed_generation_length,
                    repeat=args.speed_repeat,
                    warmup=args.speed_warmup,
                    tensor_parallel_size=args.speed_tensor_parallel_size,
                    gpu_memory_utilization=args.speed_gpu_memory_utilization,
                    dtype=args.speed_dtype,
                    max_model_len=args.speed_max_model_len,
                    enforce_eager=True if args.speed_enforce_eager else None,
                    lm_eval_bin=args.lm_eval_bin,
                    eval_device=args.eval_device,
                    eval_limit=args.limit,
                    show_progress=True,
                    run_label="leaderboard",
                    strict_validation=True,
                )
            )
        elif kind == "memory":
            memory_config = config.get("memory", {})
            result = run_memory_measurement(
                MemoryRequest(
                    checkpoint_name=record.name,
                    index_path=index_path,
                    device=args.memory_device,
                    dtype=args.memory_dtype or memory_config.get("dtype", "float16"),
                    batch_size=args.memory_batch_size or int(memory_config.get("batch_size", 1)),
                    prompt_length=args.memory_prompt_length or int(memory_config.get("prompt_length", 512)),
                    generation_length=args.memory_generation_length
                    or int(memory_config.get("generation_length", 128)),
                    attn_implementation=args.memory_attn_implementation or memory_config.get("attn_implementation"),
                    run_label="leaderboard",
                    strict_validation=True,
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
                    run_label="leaderboard",
                    strict_validation=True,
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
    _write_precision_audit(summary, args.suites)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
