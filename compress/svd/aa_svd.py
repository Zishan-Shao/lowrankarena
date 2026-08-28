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
    parser = argparse.ArgumentParser(description="Run the validated AA-SVD LowRankArena recipe.")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--calibration", default="wikitext2")
    parser.add_argument("--calibration-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--target-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--unsafe-overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> dict[str, object]:
    errors: list[str] = []
    root = args.upstream_root.expanduser().resolve()
    for relative in ("main.py", "export_lowrank_hf.py", "config/config.yaml"):
        if not (root / relative).is_file():
            errors.append(f"missing AA-SVD file: {root / relative}")
    if not 0.0 < args.keep_ratio <= 1.0:
        errors.append("--keep-ratio must be in (0, 1]")
    if args.calibration not in {"wikitext2", "ptb", "c4"}:
        errors.append("--calibration must be one of wikitext2, ptb, c4")
    if args.calibration_file and not args.calibration_file.expanduser().is_file():
        errors.append(f"calibration file is missing: {args.calibration_file}")
    missing = missing_packages(
        ("torch", "transformers", "datasets", "hydra", "omegaconf", "wandb")
    )
    if missing:
        errors.append("missing Python packages: " + ", ".join(missing))
    return {"ok": not errors, "errors": errors, "upstream_root": str(root)}


def infer_model_config(model: str, revision: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=True)
    if config.model_type != "llama":
        raise ValueError(
            "AA-SVD's unified adapter currently supports its bundled LLaMA configs only; "
            "provide --model-config explicitly for another supported upstream config"
        )
    model_name = model.lower()
    if "llama-3" in model_name or getattr(config, "vocab_size", 0) > 100_000:
        return "llama3-8B"
    if "llama-2" in model_name:
        return "llama2-7B"
    return "llama-7B"


def execute(args: argparse.Namespace) -> dict[str, object]:
    root = args.upstream_root.expanduser().resolve()
    output_dir = ensure_empty_output(args.output_dir, args.unsafe_overwrite)
    run_root = output_dir.parent
    native_hf = run_root / "aa_svd_native_hf"
    native_modules = run_root / "aa_svd_native_modules"
    hydra_root = run_root / "aa_svd_hydra"
    model_config = infer_model_config(args.model, args.revision, args.model_config)
    environment = {}
    if args.calibration_file:
        environment["LOWRANKARENA_CALIBRATION_FILE"] = str(
            args.calibration_file.expanduser().resolve()
        )

    main_command = [
        sys.executable,
        "-u",
        "main.py",
        f"model={model_config}",
        f"model.name={args.model}",
        f"model.dtype={args.model_dtype}",
        f"compression.target_param_ratio={args.keep_ratio}",
        "compression.sub_method=obj2",
        "compression.dobi_remapping=false",
        "compression.finetune.enabled=true",
        f"compression.save_path={native_modules}",
        "wandb.use=false",
        "evaluate=null",
        f"+save={{dir:{run_root},name:{native_hf.name}}}",
        f"paths.output_dir={hydra_root}",
        f"hydra.run.dir={hydra_root / 'run'}",
        "hydra.job.chdir=false",
    ]
    # Keep the recorded reproduction command unchanged for its default values.
    # Only add overrides when the caller explicitly asks for a non-default run.
    if args.revision != "main":
        main_command.insert(4, f"model.revision={args.revision}")
    if args.calibration != "wikitext2":
        main_command.insert(5, f"data={args.calibration}")
    run_logged(
        main_command,
        cwd=root,
        log_path=run_root / "aa_svd_compress.log",
        environment=environment,
    )
    export_command = [
        sys.executable,
        "-u",
        "export_lowrank_hf.py",
        "--native-checkpoint",
        str(native_hf),
        "--base-model",
        args.model,
        "--output-dir",
        str(output_dir),
        "--keep-ratio",
        str(args.keep_ratio),
        "--target-dtype",
        args.target_dtype,
        "--max-shard-size",
        "5GB",
    ]
    run_logged(
        export_command,
        cwd=root,
        log_path=run_root / "aa_svd_export.log",
        environment=environment,
    )
    return {
        "method": "AA-SVD",
        "model": args.model,
        "keep_ratio": args.keep_ratio,
        "model_config": model_config,
        "output_dir": str(output_dir),
    }


def build(request):
    """Build or execute AA-SVD through the unified compression contract."""
    from compress.common import prepare_baseline
    from compress.save import execute_artifact, save_artifact

    baseline = prepare_baseline(request)
    if baseline.path is None:
        raise RuntimeError("AA-SVD source is unavailable; use --clone-baseline")
    output_dir = request.artifact_root / request.artifact_id / "weights"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--upstream-root",
        baseline.path,
        "--model",
        request.model,
        "--revision",
        request.revision,
        "--keep-ratio",
        str(request.ratio),
        "--calibration",
        request.calibration,
        "--output-dir",
        str(output_dir.resolve()),
        "--model-dtype",
        str(request.extra.get("model_dtype", "bfloat16")),
        "--target-dtype",
        str(request.extra.get("target_dtype", "float16")),
    ]
    if request.extra.get("model_config"):
        command.extend(["--model-config", str(request.extra["model_config"])])
    if request.extra.get("calibration_file"):
        command.extend(["--calibration-file", str(request.extra["calibration_file"])])
    artifact = save_artifact(
        request,
        baseline=baseline,
        command=command,
        output_format="lowrankarena_hf",
        status="planned",
        ready_for_load=False,
        notes="AA-SVD validated recipe with canonical HF export.",
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
