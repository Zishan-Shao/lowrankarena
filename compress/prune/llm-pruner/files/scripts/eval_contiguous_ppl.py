#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWRANKARENA_ROOT = REPO_ROOT.parent / "lowrankarena"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOWRANKARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(LOWRANKARENA_ROOT))

from hf_prune import LlamaForCausalLMWithGen  # noqa: E402,F401
from src.ppl_runner import (  # noqa: E402
    _build_contiguous_blocks,
    _dataset_token_ids,
)


def _load_pruned_model(checkpoint_path: str) -> Any:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    raise ValueError(f"Unsupported pruned checkpoint payload in {checkpoint_path}")


def _load_tensor_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    if (
        isinstance(payload, dict)
        and "state_dict" in payload
        and isinstance(payload["state_dict"], dict)
        and all(torch.is_tensor(v) for v in payload["state_dict"].values())
    ):
        return payload["state_dict"]
    raise ValueError(f"Unsupported state_dict checkpoint payload in {checkpoint_path}")


def _load_state_dict_model(
    *,
    base_model_path: str,
    state_dict_checkpoint: str,
    device: torch.device,
    dtype_name: str,
) -> Any:
    torch_dtype = None
    if device.type == "cuda":
        if dtype_name == "float16":
            torch_dtype = torch.float16
        elif dtype_name == "bfloat16":
            torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    state_dict = _load_tensor_state_dict(state_dict_checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            json.dumps(
                {
                    "state_dict_checkpoint": state_dict_checkpoint,
                    "missing_keys": list(incompatible.missing_keys),
                    "unexpected_keys": list(incompatible.unexpected_keys),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
    return model


def _maybe_cast_model(model: Any, *, device: torch.device, dtype_name: str) -> Any:
    if dtype_name == "float16" and device.type == "cuda":
        model = model.half()
    elif dtype_name == "bfloat16" and device.type == "cuda":
        model = model.bfloat16()
    else:
        model = model.float()
    return model.to(device)


def _evaluate_blocks(
    model: Any,
    blocks: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, int(blocks.shape[0]), int(batch_size)):
        batch = blocks[start : start + int(batch_size)].to(device)
        with torch.inference_mode():
            outputs = model(input_ids=batch, use_cache=False)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = batch[:, 1:].contiguous()
            loss_sum = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                reduction="sum",
            )
        total_nll += float(loss_sum.item())
        total_tokens += int(labels.numel())
    return total_nll, total_tokens


def _dataset_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        if name == "wikitext2":
            configs.append({"name": "wikitext2", "kind": "wikitext2", "split": args.wikitext2_split})
        elif name == "c4_stream":
            configs.append(
                {
                    "name": "c4_stream",
                    "kind": "c4_stream",
                    "split": args.c4_split,
                    "max_eval_tokens": int(args.c4_max_eval_tokens),
                }
            )
        else:
            raise ValueError(f"Unsupported dataset name: {name}")
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate contiguous PPL for LLM-Pruner checkpoints.")
    parser.add_argument("--base-model-path", required=True, help="Local HF path for the base tokenizer/model.")
    parser.add_argument("--checkpoint-path", default=None, help="Path to pruned pytorch_model.bin. Omit for baseline.")
    parser.add_argument(
        "--state-dict-checkpoint",
        default=None,
        help="Path to a raw state_dict checkpoint. The base model will be loaded from --base-model-path first.",
    )
    parser.add_argument("--checkpoint-label", required=True, help="Human-readable label stored in the output JSON.")
    parser.add_argument("--output-json", required=True, help="Where to save metrics.")
    parser.add_argument("--datasets", default="wikitext2,c4_stream", help="Comma-separated datasets.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--wikitext2-split", default="test")
    parser.add_argument("--c4-split", default="validation")
    parser.add_argument("--c4-max-eval-tokens", type=int, default=262144)
    args = parser.parse_args()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    # We build contiguous blocks ourselves, so the raw joined corpus may exceed
    # tokenizer.model_max_length without causing an indexing issue downstream.
    tokenizer.model_max_length = int(1e30)

    if args.checkpoint_path and args.state_dict_checkpoint:
        raise ValueError("Specify at most one of --checkpoint-path or --state-dict-checkpoint")

    if args.checkpoint_path:
        model = _load_pruned_model(args.checkpoint_path)
        checkpoint_mode = "pruned"
    elif args.state_dict_checkpoint:
        model = _load_state_dict_model(
            base_model_path=args.base_model_path,
            state_dict_checkpoint=args.state_dict_checkpoint,
            device=device,
            dtype_name=args.dtype,
        )
        checkpoint_mode = "state_dict"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == "cuda" and args.dtype == "float16" else None,
        )
        checkpoint_mode = "baseline"

    model.eval()
    model = _maybe_cast_model(model, device=device, dtype_name=args.dtype)

    task_results: list[dict[str, Any]] = []
    for dataset_config in _dataset_configs(args):
        token_ids, dataset_meta = _dataset_token_ids(
            dataset_config,
            tokenizer=tokenizer,
            max_length=int(args.max_length),
            default_cache_dir=None,
        )
        blocks = _build_contiguous_blocks(token_ids, max_length=int(args.max_length))
        total_nll, total_tokens = _evaluate_blocks(
            model,
            blocks,
            batch_size=int(args.batch_size),
            device=device,
        )
        ppl = math.exp(total_nll / float(total_tokens))
        task_results.append(
            {
                "name": str(dataset_config["name"]),
                "kind": str(dataset_config["kind"]),
                "ppl": ppl,
                "total_tokens": total_tokens,
                "num_blocks": int(blocks.shape[0]),
                "max_length": int(args.max_length),
                "dataset_meta": dataset_meta,
            }
        )

    payload = {
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_mode": checkpoint_mode,
        "base_model_path": args.base_model_path,
        "checkpoint_path": args.checkpoint_path,
        "state_dict_checkpoint": args.state_dict_checkpoint,
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "tasks": task_results,
        "by_metric": {item["name"]: item["ppl"] for item in task_results},
        "mean_ppl": sum(item["ppl"] for item in task_results) / float(len(task_results)),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
