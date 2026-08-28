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
    parser = argparse.ArgumentParser(description="Run Swift-SVD uniform allocation and HF export.")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--svd-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute-device", default="cpu")
    parser.add_argument("--target-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--unsafe-overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> dict[str, object]:
    errors: list[str] = []
    root = args.upstream_root.expanduser().resolve()
    svd_file = args.svd_file.expanduser().resolve()
    for relative in ("uniform_rank_allocation.py", "export_lowrank_hf.py"):
        if not (root / relative).is_file():
            errors.append(f"missing Swift-SVD file: {root / relative}")
    if not svd_file.is_file():
        errors.append(f"SVD statistics file is missing: {svd_file}")
    if not 0.0 < args.keep_ratio <= 1.0:
        errors.append("--keep-ratio must be in (0, 1]")
    missing = missing_packages(
        ("torch", "transformers", "numpy", "safetensors", "huggingface_hub")
    )
    if missing:
        errors.append("missing Python packages: " + ", ".join(missing))
    return {"ok": not errors, "errors": errors, "upstream_root": str(root)}


def execute(args: argparse.Namespace) -> dict[str, object]:
    root = args.upstream_root.expanduser().resolve()
    svd_file = args.svd_file.expanduser().resolve()
    output_dir = ensure_empty_output(args.output_dir, args.unsafe_overwrite)
    run_root = output_dir.parent
    allocation = run_root / "swift_svd_uniform_allocation.pkl"
    run_logged(
        [
            sys.executable,
            "-u",
            "uniform_rank_allocation.py",
            "--local_model_path",
            args.model,
            "--svd_file",
            str(svd_file),
            "--compression_ratio",
            str(args.keep_ratio),
            "--output_file",
            str(allocation),
        ],
        cwd=root,
        log_path=run_root / "swift_svd_allocation.log",
    )
    run_logged(
        [
            sys.executable,
            "-u",
            "export_lowrank_hf.py",
            "--base-model",
            args.model,
            "--svd-file",
            str(svd_file),
            "--rank-allocation",
            str(allocation),
            "--output-dir",
            str(output_dir),
            "--keep-ratio",
            str(args.keep_ratio),
            "--compute-device",
            args.compute_device,
            "--target-dtype",
            args.target_dtype,
            "--max-shard-size",
            "5GB",
        ],
        cwd=root,
        log_path=run_root / "swift_svd_export.log",
    )
    return {
        "method": "Swift-SVD",
        "model": args.model,
        "keep_ratio": args.keep_ratio,
        "svd_file": str(svd_file),
        "output_dir": str(output_dir),
    }


def build(request):
    """Build or execute Swift-SVD through the unified compression contract."""
    from compress.common import prepare_baseline
    from compress.save import execute_artifact, save_artifact

    baseline = prepare_baseline(request)
    if baseline.path is None:
        raise RuntimeError("Swift-SVD source is unavailable; use --clone-baseline")
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
        "--svd-file",
        str(request.extra.get("svd_file", "<required:svd_file>")),
        "--output-dir",
        str(output_dir.resolve()),
        "--compute-device",
        str(request.extra.get("device", "cpu")),
        "--target-dtype",
        str(request.extra.get("target_dtype", "float16")),
    ]
    artifact = save_artifact(
        request,
        baseline=baseline,
        command=command,
        output_format="lowrankarena_hf",
        status="planned",
        ready_for_load=False,
        notes="Swift-SVD uniform allocation with canonical HF export.",
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
