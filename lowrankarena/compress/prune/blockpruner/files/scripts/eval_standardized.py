#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWRANKARENA_ROOT = REPO_ROOT.parent / "lowrankarena"
if str(LOWRANKARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(LOWRANKARENA_ROOT))

from src.ppl_runner import _build_contiguous_blocks, _dataset_token_ids  # noqa: E402
from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402


DEFAULT_TASKS = ["openbookqa", "arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag", "boolq"]


class MaskedLlamaDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = None
        self.mlp = None
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.mask_block = ""

    def setting_layer(self, layer):
        if "mha" not in self.mask_block:
            self.input_layernorm = layer.input_layernorm
            self.self_attn = layer.self_attn
        else:
            self.input_layernorm = None
            self.self_attn = None
        if "mlp" not in self.mask_block:
            self.post_attention_layernorm = layer.post_attention_layernorm
            self.mlp = layer.mlp
        else:
            self.post_attention_layernorm = None
            self.mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "mha" not in self.mask_block:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = attn_outputs[0]
            self_attn_weights = None
            present_key_value = None
            if output_attentions and len(attn_outputs) >= 2:
                self_attn_weights = attn_outputs[1]
            if use_cache:
                cache_idx = 2 if output_attentions else 1
                if len(attn_outputs) > cache_idx:
                    present_key_value = attn_outputs[cache_idx]
            hidden_states = residual.to(hidden_states.device) + hidden_states
        else:
            self_attn_weights = None
            present_key_value = None

        if "mlp" not in self.mask_block:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual.to(hidden_states.device) + hidden_states

        if not output_attentions and not use_cache:
            return hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardized PPL + 7-task MCQ eval for BlockPruner masks.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ppl-search-file", required=True)
    parser.add_argument("--del-block-num", type=int, required=True)
    parser.add_argument("--requested-keep-ratio", required=True)
    parser.add_argument("--achieved-keep-ratio", type=float, required=True)
    parser.add_argument("--achieved-prune-ratio", type=float, required=True)
    parser.add_argument("--ppl-output-json", required=True)
    parser.add_argument("--mcq-output-json", required=True)
    parser.add_argument("--datasets", default="wikitext2,c4_stream")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument("--lm-eval-batch-size", type=int, default=1)
    parser.add_argument("--c4-max-eval-tokens", type=int, default=262144)
    parser.add_argument("--wikitext2-split", default="test")
    parser.add_argument("--c4-split", default="validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--limit", type=int, default=None, help="Optional lm-eval sample limit for smoke tests.")
    return parser.parse_args()


def load_del_block_list(path: Path, del_block_num: int) -> list[list[str | int]]:
    payload = json.loads(path.read_text())
    return payload[str(del_block_num)]


def apply_block_masks(model, seq):
    del_layer_dict = {}
    for block_type, block_id in seq:
        chosen_layer = model.model.layers[block_id]
        if isinstance(chosen_layer, MaskedLlamaDecoderLayer):
            chosen_layer.mask_block += block_type
            chosen_layer.setting_layer(del_layer_dict[str(block_id)])
        else:
            new_layer = MaskedLlamaDecoderLayer()
            new_layer.mask_block += block_type
            new_layer.setting_layer(chosen_layer)
            del_layer_dict[str(block_id)] = chosen_layer
            model.model.layers[block_id] = new_layer
    return del_layer_dict


def _torch_dtype(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


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


def _evaluate_blocks(model: Any, blocks: torch.Tensor, *, batch_size: int, device: torch.device) -> tuple[float, int]:
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


def run_ppl(model: Any, tokenizer: Any, args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any]:
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
            batch_size=int(args.ppl_batch_size),
            device=torch.device(args.device if torch.cuda.is_available() else "cpu"),
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
    return {
        "metadata": metadata,
        "tasks": task_results,
        "by_metric": {item["name"]: item["ppl"] for item in task_results},
        "mean_ppl": sum(item["ppl"] for item in task_results) / float(len(task_results)),
    }


def run_mcq(model: Any, tokenizer: Any, args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any]:
    task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=int(args.lm_eval_batch_size))
    raw = simple_evaluate(
        model=hflm,
        tasks=task_names,
        num_fewshot=0,
        batch_size=int(args.lm_eval_batch_size),
        limit=args.limit,
    )
    raw_results = raw["results"]
    normalized: dict[str, dict[str, float | None]] = {}
    acc_vals: list[float] = []
    report_vals: list[float] = []
    for task in task_names:
        metrics = raw_results[task]
        acc = metrics.get("acc,none")
        acc_norm = metrics.get("acc_norm,none")
        report = acc_norm if acc_norm is not None else acc
        if acc is not None:
            acc_vals.append(float(acc))
        if report is not None:
            report_vals.append(float(report))
        normalized[task] = {
            "acc": float(acc) if acc is not None else None,
            "acc_norm": float(acc_norm) if acc_norm is not None else None,
            "report": float(report) if report is not None else None,
        }
    return {
        "metadata": metadata,
        "tasks": task_names,
        "results": normalized,
        "raw_results": raw_results,
        "mcq_acc_mean": sum(acc_vals) / float(len(acc_vals)) if acc_vals else None,
        "mcq_report_mean": sum(report_vals) / float(len(report_vals)) if report_vals else None,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch_dtype = _torch_dtype(device, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    tokenizer.model_max_length = int(1e30)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False,
    )
    model = model.to(device)
    model.eval()

    del_block_list = load_del_block_list(Path(args.ppl_search_file), int(args.del_block_num))
    apply_block_masks(model, del_block_list)

    metadata = {
        "model_path": args.model_path,
        "ppl_search_file": args.ppl_search_file,
        "del_block_num": int(args.del_block_num),
        "del_block_list": del_block_list,
        "requested_keep_ratio": str(args.requested_keep_ratio),
        "achieved_keep_ratio": float(args.achieved_keep_ratio),
        "achieved_prune_ratio": float(args.achieved_prune_ratio),
        "device": str(device),
        "dtype": args.dtype,
        "max_length": int(args.max_length),
        "ppl_batch_size": int(args.ppl_batch_size),
        "lm_eval_batch_size": int(args.lm_eval_batch_size),
        "c4_max_eval_tokens": int(args.c4_max_eval_tokens),
        "limit": args.limit,
    }

    ppl_payload = run_ppl(model, tokenizer, args, metadata)
    mcq_payload = run_mcq(model, tokenizer, args, metadata)

    ppl_output = Path(args.ppl_output_json)
    mcq_output = Path(args.mcq_output_json)
    ppl_output.parent.mkdir(parents=True, exist_ok=True)
    mcq_output.parent.mkdir(parents=True, exist_ok=True)
    ppl_output.write_text(json.dumps(ppl_payload, indent=2) + "\n")
    mcq_output.write_text(json.dumps(mcq_payload, indent=2) + "\n")

    print(json.dumps({
        "ppl_output_json": str(ppl_output),
        "mcq_output_json": str(mcq_output),
        "mean_ppl": ppl_payload["mean_ppl"],
        "mcq_acc_mean": mcq_payload["mcq_acc_mean"],
        "mcq_report_mean": mcq_payload["mcq_report_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
