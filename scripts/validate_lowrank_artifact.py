#!/usr/bin/env python3
"""Validate a LowRankArena HF artifact's budget metadata and one finite forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-keep", type=float, required=True)
    parser.add_argument("--target-modules", type=int, default=224)
    parser.add_argument("--keep-tolerance", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Storage/compute dtype used when loading the artifact.",
    )
    parser.add_argument("--wiki-ppl", action="store_true")
    parser.add_argument("--ppl-batch-size", type=int, default=16)
    parser.add_argument(
        "--wiki-data-root",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "datasets" / "wikitext",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    artifact = args.artifact.expanduser().resolve()
    metadata = json.loads((artifact / "lowrankarena_method.json").read_text())
    achieved_keep = float(metadata["achieved_target_keep_ratio"])
    if abs(achieved_keep - args.expected_keep) > args.keep_tolerance:
        raise AssertionError(
            f"keep ratio {achieved_keep} is outside tolerance of {args.expected_keep}"
        )

    config = AutoConfig.from_pretrained(
        artifact, trust_remote_code=True, local_files_only=True
    )
    low_rank_modules = len(config.low_rank_modules)
    if not 0 < low_rank_modules <= args.target_modules:
        raise AssertionError(
            f"invalid low-rank module count {low_rank_modules}/{args.target_modules}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        artifact,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=args.device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        artifact, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    inputs = tokenizer(
        "Low-rank compression should preserve finite model behavior.",
        return_tensors="pt",
    ).to(args.device)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits
    finite = bool(torch.isfinite(logits).all())
    if not finite:
        raise AssertionError("non-finite logits")

    result = {
        "artifact": str(artifact),
        "method": metadata.get("method"),
        "expected_keep_ratio": args.expected_keep,
        "achieved_target_keep_ratio": achieved_keep,
        "target_modules": args.target_modules,
        "low_rank_modules": low_rank_modules,
        "dense_target_modules": args.target_modules - low_rank_modules,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "finite": finite,
    }

    if args.wiki_ppl:
        repo_root = Path(__file__).resolve().parents[1]
        swift_root = repo_root / "compress" / "svd" / "Swift-SVD"
        sys.path.insert(0, str(swift_root))
        from evaluater import ppl_eval

        ppls = ppl_eval(
            model=model,
            tokenizer=tokenizer,
            datasets=["wikitext2"],
            data_root=str(args.wiki_data_root.expanduser().resolve()),
            model_seq_len=2048,
            batch_size=args.ppl_batch_size,
            device=args.device,
            seed=7,
        )
        result["wikitext2_ppl"] = float(ppls["wikitext2"])

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
