import argparse
import datetime
import json
import os
from typing import Dict

import torch

from utils.eval_utils import ppl_eval
from utils.quant_utils import load_quantized_pt
from compression.evaluation import (
    commonsense_eval,
    set_deterministic_seeds,
    _sanitize_for_json,
    _extract_commonsense_metrics,
)

# NEW
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_csv(s: str):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _resolve_eval_dtype(s: str):
    s = (s or "").lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    return torch.float32


def main():
    ap = argparse.ArgumentParser(description="Evaluate a saved checkpoint on PPL and commonsense tasks.")

    # CHANGED: ckpt_path is optional, model_id added
    ap.add_argument("--ckpt_path", type=str, default=None, help="Path to a .pt checkpoint saved by your pipeline.")
    ap.add_argument("--model_id", type=str, default=None, help="Hugging Face model id to load if ckpt_path is not set.")

    ap.add_argument("--save_path", type=str, default="./", help="Directory to store logs and summary json.")
    ap.add_argument("--DEV", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--model_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--eval_ppl", action="store_true", help="If set, run perplexity evaluation.")
    ap.add_argument(
        "--ppl_datasets",
        type=str,
        default="wikitext2,ptb,c4",
        help="Comma separated dataset names for ppl_eval.",
    )

    ap.add_argument(
        "--evaluate_commonsense",
        action="store_true",
        help="If set, run commonsense benchmark evaluation using lm_eval.",
    )
    ap.add_argument(
        "--commonsense_tasks",
        type=str,
        default="arc_easy,arc_challenge,openbookqa,winogrande,hellaswag,piqa",
        help="Comma separated lm_eval task names.",
    )
    ap.add_argument(
        "--eval_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Model dtype to use during evaluation.",
    )

    args = ap.parse_args()

    set_deterministic_seeds(args.seed)

    do_any_eval = bool(
        getattr(args, "eval_ppl", False)
        or getattr(args, "evaluate_commonsense", False)
    )
    if not do_any_eval:
        print("Nothing to evaluate. Set --eval_ppl or --evaluate_commonsense.")
        return

    # NEW: validate source
    if args.ckpt_path is None and args.model_id is None:
        raise ValueError("Provide either --ckpt_path or --model_id")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # NEW: choose base dir and base name
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        base_dir = os.path.dirname(os.path.abspath(ckpt_path))
        base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    else:
        base_dir = os.path.abspath(args.save_path)
        base_name = args.model_id.replace("/", "__")

    run_tag = f"eval_{base_name}_{timestamp}"
    run_dir = os.path.join(base_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(run_dir, f"eval_config_{timestamp}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"💾 Saved eval configuration to {config_path}")

    # CHANGED: load from ckpt or model_id
    if args.ckpt_path is not None:
        print(f"📦 Loading checkpoint: {args.ckpt_path}")
        model, tokenizer = load_quantized_pt(args.ckpt_path, device=args.DEV)
        # Patch older configs for newer transformers masking utils
        if not hasattr(model.config, "_attn_implementation_internal"):
            model.config._attn_implementation_internal = "eager"
        source_kind = "ckpt"
        source_id = args.ckpt_path
    else:
        print(f"📦 Loading Hugging Face model: {args.model_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
        except ImportError as e:
            if "protobuf" in str(e).lower():
                print("protobuf missing, falling back to slow tokenizer (use_fast=False)")
                tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
            else:
                raise
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
        ).to(args.DEV)
        source_kind = "model_id"
        source_id = args.model_id

    model.eval()

    prev_dtype = next(model.parameters()).dtype
    eval_dtype = _resolve_eval_dtype(args.eval_dtype)
    model = model.to(dtype=eval_dtype)
    print(f"🔧 Eval dtype: {args.eval_dtype} (from {prev_dtype})")

    final_ppl_results: Dict[str, float] = {}
    final_cs_results = None

    if getattr(args, "eval_ppl", False):
        ppl_datasets = _parse_csv(args.ppl_datasets)
        if not ppl_datasets:
            ppl_datasets = ["wikitext2", "ptb", "c4"]

        print(f"\n📈 Evaluating perplexity on: {ppl_datasets}")
        ppl_results = ppl_eval(
            model,
            tokenizer,
            datasets=ppl_datasets,
            model_seq_len=args.model_seq_len,
            batch_size=args.batch_size,
            device=args.DEV,
        )
        if isinstance(ppl_results, dict) and ppl_results:
            final_ppl_results = ppl_results
            for ds_name, ppl_val in ppl_results.items():
                try:
                    print(f"   ✓ {ds_name} PPL: {float(ppl_val):.4f}")
                except Exception:
                    print(f"   ✓ {ds_name} PPL: {ppl_val}")

    if getattr(args, "evaluate_commonsense", False):
        tasks_csv = args.commonsense_tasks
        print(f"\n🧠 Evaluating commonsense tasks: {tasks_csv}")
        final_cs_results = commonsense_eval(model, tokenizer, tasks_csv=tasks_csv, device=args.DEV)

        try:
            res = final_cs_results.get("results", {})
            print("\n🧠 Commonsense results summary")
            for task_name, metrics in res.items():
                if isinstance(metrics, dict):
                    keys = [k for k in ["acc", "acc_norm", "exact_match", "mc1", "mc2"] if k in metrics]
                    if keys:
                        msg = ", ".join([f"{k}={metrics[k]:.4f}" for k in keys if isinstance(metrics[k], (int, float))])
                        print(f"   {task_name}: {msg}")
                    else:
                        print(f"   {task_name}: {metrics}")
                else:
                    print(f"   {task_name}: {metrics}")
        except Exception:
            print("Commonsense evaluation finished, results object printed below")
            print(final_cs_results)

    model = model.to(dtype=prev_dtype)

    summary = {
        "source_kind": source_kind,
        "source_id": source_id,
        "device": args.DEV,
        "seed": args.seed,
        "model_seq_len": args.model_seq_len,
        "batch_size": args.batch_size,
        "eval_dtype": args.eval_dtype,
        "eval_ppl": bool(args.eval_ppl),
        "ppl_datasets": _parse_csv(args.ppl_datasets),
        "final_perplexity": final_ppl_results,
        "evaluate_commonsense": bool(args.evaluate_commonsense),
        "commonsense_tasks": _parse_csv(args.commonsense_tasks),
        "commonsense": _extract_commonsense_metrics(final_cs_results),
    }

    if bool(args.eval_ppl) and bool(args.evaluate_commonsense):
        eval_kind = "ppl_multichoice"
    elif bool(args.eval_ppl):
        eval_kind = "ppl"
    elif bool(args.evaluate_commonsense):
        eval_kind = "multichoice"
    else:
        eval_kind = "none"

    summary_filename = f"{base_name}_{timestamp}_{eval_kind}.json"
    summary_path = os.path.join(run_dir, summary_filename)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(summary), f, indent=2)

    print(f"\n💾 Saved eval summary to {summary_path}")


if __name__ == "__main__":
    main()
    print("\n🎉 All done!")
