from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import load_suite_config, select_checkpoints_for_suite, suite_id
from src.dtype_utils import normalize_dtype_name
from src.lm_eval_runner import LmEvalRequest, run_lm_eval_suite
from src.utils import dump_json, ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation lanes or eval suites from benchmark/**/*.yaml.")
    parser.add_argument(
        "--suites",
        nargs="*",
        default=["base"],
        help="Evaluation suite names such as base, instruct, mcq, ppl, or base/base_math.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint name from the index. Repeat to run multiple explicit checkpoints and bypass suite selection filters.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[],
        help="Checkpoint names from the index. Bypasses suite selection filters.",
    )
    parser.add_argument("--index", default=str(ROOT / "checkpoints" / "index.csv"))
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval limit override for smoke runs.")
    parser.add_argument("--lm-eval-bin", default="lm-eval")
    parser.add_argument(
        "--eval-model-backend",
        default=None,
        help="Override suite model backend for lm-eval suites, e.g. hf or vllm. PPL uses its native runner.",
    )
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--eval-dtype", default=None, help="Override eval suite dtype, e.g. auto, float16, fp16, bfloat16, or bf16.")
    parser.add_argument("--eval-tensor-parallel-size", type=int, default=None, help="vLLM tensor_parallel_size override.")
    parser.add_argument("--eval-gpu-memory-utilization", type=float, default=None, help="vLLM gpu_memory_utilization override.")
    parser.add_argument("--eval-max-model-len", type=int, default=None, help="vLLM max_model_len override.")
    parser.add_argument("--eval-enforce-eager", action="store_true", help="Pass enforce_eager=True to vLLM eval suites.")
    return parser.parse_args()


def _checkpoint_selection_override(args: argparse.Namespace) -> dict | None:
    names: list[str] = []
    for item in [*args.checkpoint, *args.checkpoints]:
        names.extend(part.strip() for part in str(item).split(",") if part.strip())
    if not names:
        return None
    return {
        "checkpoints": names,
        "enabled_only": False,
    }


def _normalize_precision(value: object) -> str | None:
    if value is None:
        return None
    try:
        return normalize_dtype_name(value)
    except ValueError:
        return str(value).replace("torch.", "").strip().lower()


def _runtime_precision(config: dict, precision: dict, factor_dtypes: list[str]) -> str | None:
    requested = _normalize_precision(config.get("dtype"))
    if requested and requested != "auto":
        return requested
    reference = _normalize_precision(precision.get("reference_torch_dtype"))
    if reference:
        return reference
    unique_factor_dtypes = sorted(set(factor_dtypes))
    if len(unique_factor_dtypes) == 1:
        return unique_factor_dtypes[0]
    return requested


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
        factor_dtypes = [_normalize_precision(value) for value in precision.get("low_rank_factor_dtypes", []) if value is not None]
        factor_dtypes = [value for value in factor_dtypes if value]
        runtime_precision = _runtime_precision(config, precision, factor_dtypes)
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
        "passed": not mixed_checkpoint_precision,
        "runtime_precisions": sorted(runtime_precisions),
        "missing_runtime_precision": missing_runtime_precision,
        "mixed_checkpoint_precision": mixed_checkpoint_precision,
        "records": records,
    }
    dump_json(audit, _precision_audit_path(suite_names))
    if mixed_checkpoint_precision:
        raise RuntimeError(
            "Leaderboard precision audit found checkpoints with mixed low-rank factor dtypes: "
            + ", ".join(mixed_checkpoint_precision)
        )
    return audit


def run_suite(
    suite_name: str,
    index_path: str,
    args: argparse.Namespace,
    *,
    selection_override: dict | None = None,
) -> list[dict[str, str]]:
    config_path, config = load_suite_config(suite_name)
    kind = config.get("kind", "eval")
    suite_name_resolved = suite_id(config_path)

    if kind == "aggregate":
        outputs: list[dict[str, str]] = []
        child_selection = selection_override if selection_override is not None else config.get("selection")
        for included in config.get("includes", []):
            outputs.extend(
                run_suite(
                    str(included),
                    index_path=index_path,
                    args=args,
                    selection_override=child_selection,
                )
            )
        return outputs

    outputs: list[dict[str, str]] = []
    for record in select_checkpoints_for_suite(config, index_path=index_path, selection_override=selection_override):
        if kind not in {"eval", None}:
            raise ValueError(
                f"Suite '{suite_name_resolved}' has kind '{kind}'. "
                "Use scripts/run_speed.py for speed suites and scripts/run_memory.py for memory suites."
            )
        extra_model_args = {"dtype": args.eval_dtype} if args.eval_dtype is not None else {}
        result = run_lm_eval_suite(
            LmEvalRequest(
                checkpoint_name=record.name,
                suite_path=config_path,
                index_path=index_path,
                lm_eval_bin=args.lm_eval_bin,
                model_backend=args.eval_model_backend,
                device=args.eval_device,
                limit=args.limit,
                tensor_parallel_size=args.eval_tensor_parallel_size,
                gpu_memory_utilization=args.eval_gpu_memory_utilization,
                max_model_len=args.eval_max_model_len,
                enforce_eager=True if args.eval_enforce_eager else None,
                extra_model_args=extra_model_args,
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
    selection_override = _checkpoint_selection_override(args)
    summary: list[dict[str, str]] = []
    for suite_name in args.suites:
        summary.extend(
            run_suite(
                suite_name,
                index_path=args.index,
                args=args,
                selection_override=selection_override,
            )
        )
    _write_precision_audit(summary, args.suites)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
