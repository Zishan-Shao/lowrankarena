from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.common import (
    BASELINE_SPECS,
    CompressionRequest,
    preflight_request,
    prepare_baseline,
)


MODULE_BY_METHOD = {
    ("svd", "aa_svd"): "compress.svd.aa_svd",
    ("svd", "asvd"): "compress.svd.asvd",
    ("svd", "basis_sharing"): "compress.svd.basis_sharing",
    ("svd", "dobi_svd"): "compress.svd.dobi_svd",
    ("svd", "fwsvd"): "compress.svd.fwsvd",
    ("svd", "gfw_svd"): "compress.svd.gfw_svd",
    ("svd", "svd"): "compress.svd.svd",
    ("svd", "svd_llm"): "compress.svd.svd_llm",
    ("svd", "swift_svd"): "compress.svd.swift_svd",
    ("svd", "zs_svd"): "compress.svd.zs_svd",
    ("prune", "bonsai"): "compress.prune.bonsai",
    ("prune", "llm_pruner"): "compress.prune.llm_pruner",
    ("prune", "slicegpt"): "compress.prune.slicegpt",
    ("prune", "wanda_sp"): "compress.prune.wanda_sp",
    ("quant", "awq"): "compress.quant.awq",
    ("quant", "gptq"): "compress.quant.gptq",
    ("quant", "rtn"): "compress.quant.rtn",
}


def parse_extra(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--extra entries must look like key=value, got: {value}")
        key, raw = value.split("=", 1)
        parsed[key.strip()] = raw.strip()
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional artifact-generation entrypoint for LowRankArena.")
    parser.add_argument("--list-methods", action="store_true", help="Print integration and execution status for every public method.")
    parser.add_argument("--family", choices=["svd", "prune", "quant"])
    parser.add_argument("--method")
    parser.add_argument("--model", help="Dense model ID or local dense model path.")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--calibration", default="wikitext2")
    parser.add_argument("--recovery", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", default="huggingface")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-root", default=str(ROOT / "compress" / "artifacts"))
    parser.add_argument("--baseline-root", default=None, help="Optional alternate checkout root for third-party baselines.")
    parser.add_argument("--clone-baseline", action="store_true")
    parser.add_argument("--refresh-baseline", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run the method after strict preflight instead of writing a plan only.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate dependencies, source, and required assets without loading model weights.")
    parser.add_argument("--register", action="store_true", help="Register the generated artifact if it becomes ready_for_load.")
    parser.add_argument("--enabled", action="store_true", help="Enable the registered artifact by default.")
    parser.add_argument("--extra", action="append", default=[], help="Repeatable key=value metadata fields.")
    return parser.parse_args()


def list_methods() -> list[dict[str, object]]:
    methods: list[dict[str, object]] = []
    for family, method in sorted(MODULE_BY_METHOD):
        spec = BASELINE_SPECS[(family, method)]
        methods.append(
            {
                "family": family,
                "method": method,
                "display_name": spec.display_name,
                "integration_status": spec.integration_status,
                "supports_execute": spec.supports_execute,
                "required_extra": list(spec.required_extra),
                "required_packages": list(spec.required_packages),
                "git_url": spec.git_url,
                "git_ref": spec.git_ref,
            }
        )
    return methods


def main() -> None:
    args = parse_args()
    if args.list_methods:
        print(json.dumps(list_methods(), indent=2, sort_keys=True))
        return
    missing = [name for name in ("family", "method", "model") if not getattr(args, name)]
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(f"--{name}" for name in missing))

    module_name = MODULE_BY_METHOD.get((args.family, args.method))
    if not module_name:
        known = ", ".join(f"{family}/{method}" for family, method in sorted(MODULE_BY_METHOD))
        raise SystemExit(f"Unsupported method '{args.family}/{args.method}'. Known methods: {known}")

    request = CompressionRequest(
        family=args.family,
        method=args.method,
        model=args.model,
        ratio=args.ratio,
        output_root=args.output_root,
        tokenizer=args.tokenizer,
        revision=args.revision,
        precision=args.precision,
        calibration=args.calibration,
        recovery=args.recovery,
        seed=args.seed,
        source=args.source,
        notes=args.notes,
        clone_baseline=args.clone_baseline,
        refresh_baseline=args.refresh_baseline,
        execute=args.execute,
        register=args.register,
        enabled=args.enabled,
        baseline_root=args.baseline_root,
        extra=parse_extra(args.extra),
    )

    if args.execute or args.preflight_only:
        baseline = prepare_baseline(request)
        report = preflight_request(
            request,
            baseline,
            for_execute=(args.execute or args.preflight_only),
        )
        if args.preflight_only:
            print(json.dumps(asdict(report), indent=2, sort_keys=True))
            raise SystemExit(0 if report.ok else 2)
        if not report.ok:
            print(json.dumps(asdict(report), indent=2, sort_keys=True), file=sys.stderr)
            raise SystemExit(2)

    module = importlib.import_module(module_name)
    artifact = module.build(request)
    print(json.dumps(asdict(artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
