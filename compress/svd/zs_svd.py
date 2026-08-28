#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compress.common import ensure_empty_output, missing_packages, run_logged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the validated one-stage ZS-SVD recipe.")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--calibration", default="wikitext2")
    parser.add_argument("--calibration-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--target-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--unsafe-overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> dict[str, object]:
    errors: list[str] = []
    root = args.upstream_root.expanduser().resolve()
    for relative in ("main_zero_sum.py", "export_lowrank_hf.py"):
        if not (root / relative).is_file():
            errors.append(f"missing ZS-SVD file: {root / relative}")
    if not 0.0 < args.keep_ratio < 1.0:
        errors.append("--keep-ratio must be in (0, 1) for ZS-SVD")
    if args.nsamples < 1:
        errors.append("--nsamples must be >= 1")
    if args.calibration not in {"wikitext2", "ptb", "c4"}:
        errors.append("--calibration must be one of wikitext2, ptb, c4")
    if args.calibration_file and not args.calibration_file.expanduser().is_file():
        errors.append(f"calibration file is missing: {args.calibration_file}")
    missing = missing_packages(
        ("torch", "transformers", "datasets", "accelerate", "safetensors")
    )
    if missing:
        errors.append("missing Python packages: " + ", ".join(missing))
    return {"ok": not errors, "errors": errors, "upstream_root": str(root)}


def execute(args: argparse.Namespace) -> dict[str, object]:
    root = args.upstream_root.expanduser().resolve()
    output_dir = ensure_empty_output(args.output_dir, args.unsafe_overwrite)
    run_root = output_dir.parent
    native_root = run_root / "zs_svd_native"
    environment = {}
    if args.calibration_file:
        environment["LOWRANKARENA_CALIBRATION_FILE"] = str(
            args.calibration_file.expanduser().resolve()
        )
    global_prune_ratio = 1.0 - args.keep_ratio
    run_logged(
        [
            sys.executable,
            "-u",
            "main_zero_sum.py",
            "--model",
            args.model,
            "--save_path",
            str(native_root),
            "--dataset",
            args.calibration,
            "--global_prune_ratio",
            str(global_prune_ratio),
            "--keep_rank_ratio",
            "0",
            "--num_stages",
            "1",
            "--nsamples",
            str(args.nsamples),
            "--nsamples_gradient_subset",
            "1",
            "--selection_mode",
            "zero_sum",
            "--importance_seq_len",
            "2048",
            "--sub_with_teacher_module",
            "--eval_ppl",
            "--final_eval_datasets",
            args.calibration,
            "--save_after_truncation",
            "--seed",
            str(args.seed),
            "--model_seq_len",
            "2048",
            "--DEV",
            args.device,
        ],
        cwd=root,
        log_path=run_root / "zs_svd_compress.log",
        environment=environment,
    )
    candidates = sorted(native_root.rglob("final_ppl*_fp16_compressed.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"ZS-SVD completed without a final fp16 checkpoint under {native_root}"
        )
    native_checkpoint = candidates[-1]
    run_logged(
        [
            sys.executable,
            "-u",
            "export_lowrank_hf.py",
            "--checkpoint",
            str(native_checkpoint),
            "--output-dir",
            str(output_dir),
            "--keep-ratio",
            str(args.keep_ratio),
            "--target-dtype",
            args.target_dtype,
            "--max-shard-size",
            "5GB",
        ],
        cwd=root,
        log_path=run_root / "zs_svd_export.log",
        environment=environment,
    )
    return {
        "method": "ZS-SVD",
        "model": args.model,
        "keep_ratio": args.keep_ratio,
        "native_checkpoint": str(native_checkpoint),
        "output_dir": str(output_dir),
    }


def build(request):
    """Build or execute ZS-SVD through the unified compression contract."""
    from compress.common import prepare_baseline
    from compress.save import execute_artifact, save_artifact

    baseline = prepare_baseline(request)
    if baseline.path is None:
        raise RuntimeError("ZS-SVD source is unavailable; use --clone-baseline")
    output_dir = request.artifact_root / request.artifact_id / "weights"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--upstream-root",
        baseline.path,
        "--model",
        request.model,
        "--keep-ratio",
        str(request.ratio),
        "--calibration",
        request.calibration,
        "--output-dir",
        str(output_dir.resolve()),
        "--device",
        str(request.extra.get("device", "cuda")),
        "--seed",
        str(request.extra.get("seed", 3 if request.seed == 0 else request.seed)),
        "--nsamples",
        str(request.extra.get("nsamples", 256)),
        "--target-dtype",
        str(request.extra.get("target_dtype", "float16")),
    ]
    if request.extra.get("calibration_file"):
        command.extend(["--calibration-file", str(request.extra["calibration_file"])])
    artifact = save_artifact(
        request,
        baseline=baseline,
        command=command,
        output_format="lowrankarena_hf",
        status="planned",
        ready_for_load=False,
        notes="ZS-SVD one-stage zero-sum recipe with canonical HF export.",
        register=False,
    )
    return execute_artifact(request, artifact, baseline=baseline) if request.execute else artifact


def main() -> None:
    args = parse_args()
    report = preflight(args)
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["ok"] else 2)
    if not report["ok"]:
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(execute(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
