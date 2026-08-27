#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a SAES-SVD checkpoint saved by this release's `saes_svd.py`
with token-level PPL and/or LM Evaluation Harness.

Expected checkpoint directory layout:
  - saes_state.pt
  - saes_manifest.json
  - tokenizer files

Example (from the LowRankArena repository root):
CUDA_VISIBLE_DEVICES=0 python -u compress/svd/SAES-SVD-reproduction/eval_lm_eval_saes.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --ckpt_dir /path/to/llama2_saes_r0.4 \
  --device cuda \
  --dtype float16 \
  --factor_dtype float32 \
  --run_ppl \
  --ppl_datasets wikitext2,ptb,c4 \
  --run_lm_eval \
  --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande \
  --num_fewshot 0 \
  --batch_size 4
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Preserve the original research-workspace lookup while allowing local imports
# from this release directory through Python's normal script-path handling.
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LM_EVAL_ROOT = os.path.join(_REPO_ROOT, "lm-evaluation-harness")
if os.path.isdir(_LM_EVAL_ROOT) and _LM_EVAL_ROOT not in sys.path:
    sys.path.insert(0, _LM_EVAL_ROOT)

from evaluater import ppl_eval


def parse_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def _is_valid_tokenizer(tokenizer_obj: Any) -> bool:
    return tokenizer_obj is not None and not isinstance(tokenizer_obj, bool) and callable(tokenizer_obj)


def _load_tokenizer_with_fallback(
    ckpt_dir: str,
    base_model: str,
    hf_token: Optional[str],
) -> Any:
    """
    Robust tokenizer loader:
    - Prefer tokenizer saved in ckpt_dir.
    - Handle transformers edge case where fast tokenizer loader returns bool(False).
    - Fall back to base model tokenizer if needed.
    """
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        load_kwargs["token"] = hf_token

    attempts = [
        (ckpt_dir, True, "ckpt/use_fast=True"),
        (ckpt_dir, False, "ckpt/use_fast=False"),
        (base_model, True, "base/use_fast=True"),
        (base_model, False, "base/use_fast=False"),
    ]
    last_err: Optional[Exception] = None
    for source, use_fast, tag in attempts:
        try:
            tok = AutoTokenizer.from_pretrained(source, use_fast=use_fast, **load_kwargs)
        except Exception as e:
            last_err = e
            print(f"[Warn] tokenizer load failed ({tag}): {e}", flush=True)
            continue
        if _is_valid_tokenizer(tok):
            if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            return tok
        print(f"[Warn] tokenizer load returned invalid object ({tag}): type={type(tok).__name__}", flush=True)

    if last_err is not None:
        raise RuntimeError(f"Failed to load a valid tokenizer from ckpt/base fallback chain: {last_err}")
    raise RuntimeError("Failed to load a valid tokenizer from ckpt/base fallback chain.")


def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    cur: nn.Module = root
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def set_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent: nn.Module = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if callable(obj):
        return getattr(obj, "__name__", repr(obj))
    if isinstance(obj, torch.dtype):
        return str(obj)
    if torch.is_tensor(obj):
        return f"<tensor shape={tuple(obj.shape)} dtype={obj.dtype}>"
    return str(obj)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


class SAESFactorizedLinear(nn.Module):
    """
    Minimal SAES linear for evaluation/state_dict loading:
      y = (x @ B^T) @ A^T + bias
    with A:[out,r], B:[r,in].
    """

    def __init__(self, in_features: int, out_features: int, rank: int, dtype: torch.dtype):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.register_buffer("A", torch.empty((self.out_features, self.rank), dtype=dtype))
        self.register_buffer("B", torch.empty((self.rank, self.in_features), dtype=dtype))
        self.register_buffer("bias", torch.empty(0, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        y2d = self.forward_flat(x2d)
        return y2d.reshape(*orig_shape[:-1], self.out_features)

    def forward_flat(self, x2d: torch.Tensor) -> torch.Tensor:
        in_dtype = x2d.dtype
        z = x2d.to(self.B.dtype) @ self.B.t()
        y = z @ self.A.t()
        if self.bias.numel() > 0:
            y = y + self.bias
        if y.dtype != in_dtype:
            try:
                if in_dtype.is_floating_point:
                    maxv = torch.finfo(in_dtype).max
                    y = torch.clamp(y, min=-maxv, max=maxv)
            except Exception:
                pass
            y = y.to(in_dtype)
        return y


def patch_model_from_manifest(model: nn.Module, manifest: Dict[str, Any], factor_dtype: torch.dtype) -> None:
    items = manifest.get("items", [])
    if not isinstance(items, list):
        raise TypeError("saes_manifest.json: expected 'items' to be a list")
    for it in items:
        name = it["name"]
        in_f = int(it["in_features"])
        out_f = int(it["out_features"])
        rank = int(it["rank"])
        new_mod = SAESFactorizedLinear(in_features=in_f, out_features=out_f, rank=rank, dtype=factor_dtype)
        set_module_by_name(model, name, new_mod)


def run_lm_eval(
    model: nn.Module,
    tokenizer: Any,
    *,
    device: str,
    tasks: str,
    batch_size: int,
    max_batch_size: int,
    max_length: int,
    num_fewshot: int,
    limit: Optional[int],
    include_path: Optional[str],
    add_bos_token: Optional[bool],
    prefix_token_id: Optional[int],
    out_json: Optional[str],
    disable_tqdm: bool,
) -> Dict[str, Any]:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    task_manager = TaskManager(include_path=include_path) if include_path else None

    print(
        f"[LM-Eval] tasks={task_list} fewshot={num_fewshot} bs={batch_size} max_len={max_length} limit={limit}",
        flush=True,
    )
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        max_length=max_length,
        trust_remote_code=True,
        add_bos_token=add_bos_token,
        prefix_token_id=prefix_token_id,
    )
    if disable_tqdm:
        def _wrap_disable(method):
            def wrapped(requests, disable_tqdm: bool = False, **kwargs):
                return method(requests, disable_tqdm=True, **kwargs)

            return wrapped

        if hasattr(lm, "loglikelihood"):
            lm.loglikelihood = _wrap_disable(lm.loglikelihood)  # type: ignore[assignment]
        if hasattr(lm, "loglikelihood_rolling"):
            lm.loglikelihood_rolling = _wrap_disable(lm.loglikelihood_rolling)  # type: ignore[assignment]
        if hasattr(lm, "generate_until"):
            lm.generate_until = _wrap_disable(lm.generate_until)  # type: ignore[assignment]

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        limit=limit,
        task_manager=task_manager,
    )
    if results is None:
        raise RuntimeError("LM Evaluation Harness returned no results (not rank 0).")
    if out_json:
        _write_json(out_json, results)

    print("\nLM-Eval results (summary):", flush=True)
    res = results.get("results", {})
    if isinstance(res, dict) and res:
        for t in task_list:
            if t in res and isinstance(res[t], dict):
                keys = ["acc_norm", "acc", "f1", "exact_match", "bleu", "rouge1", "rougeL", "ppl"]
                metrics = {k: res[t][k] for k in keys if k in res[t]}
                print(f"  {t}: {metrics if metrics else res[t]}", flush=True)
            else:
                print(f"  {t}: (no entry)", flush=True)
    else:
        print(res if res else results, flush=True)
    if out_json:
        print(f"[LM-Eval] Wrote full results to: {out_json}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True, help="HF base model id, e.g. meta-llama/Llama-2-7b-hf")
    ap.add_argument("--ckpt_dir", type=str, required=True, help="SAES checkpoint directory saved by saes_svd.py")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", help="Model dtype (float16/bfloat16/float32).")
    ap.add_argument("--factor_dtype", type=str, default="float32", help="Factor module dtype.")
    ap.add_argument("--hf_token", type=str, default=None)

    # PPL
    ap.add_argument("--run_ppl", action="store_true", help="Run token-level PPL.")
    ap.add_argument("--ppl_datasets", type=str, default="wikitext2,ptb,c4")
    ap.add_argument("--ppl_seq_len", type=int, default=2048)
    ap.add_argument("--ppl_batch_size", type=int, default=4)
    ap.add_argument("--ppl_max_batches", type=int, default=None)
    ap.add_argument(
        "--c4_val_stream",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Control C4 validation loading for PPL. 'auto' enables streaming when ppl_datasets contains c4_stream.",
    )
    ap.add_argument(
        "--c4_val_docs",
        type=int,
        default=None,
        help="Optional override for number of C4 validation docs used in PPL (env: SVDLLM_C4_VAL_DOCS).",
    )
    ap.add_argument(
        "--c4_val_dataset",
        type=str,
        default=None,
        help="Optional override for C4 dataset id used by PPL (env: SVDLLM_C4_VAL_DATASET).",
    )

    # LM-Eval
    ap.add_argument("--run_lm_eval", action="store_true", help="Run lm-evaluation-harness.")
    ap.add_argument("--tasks", type=str, default="arc_easy,arc_challenge,hellaswag,piqa,winogrande")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include_path", type=str, default=None)
    ap.add_argument("--add_bos_token", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--prefix_token_id", type=int, default=None)
    ap.add_argument("--tqdm", type=str, default="off", choices=["off", "on"])
    ap.add_argument("--out_json", type=str, default=None)

    args = ap.parse_args()

    if not args.run_ppl and not args.run_lm_eval:
        # Default behavior: run both when no explicit selector is set.
        args.run_ppl = True
        args.run_lm_eval = True

    ckpt_dir = args.ckpt_dir
    manifest_path = os.path.join(ckpt_dir, "saes_manifest.json")
    state_path = os.path.join(ckpt_dir, "saes_state.pt")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing {manifest_path}")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Missing {state_path}")

    model_dtype = parse_dtype(args.dtype)
    factor_dtype = parse_dtype(args.factor_dtype)

    tokenizer = _load_tokenizer_with_fallback(
        ckpt_dir=ckpt_dir,
        base_model=args.base_model,
        hf_token=args.hf_token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=model_dtype,
        device_map="cpu",
        trust_remote_code=True,
        token=args.hf_token,
    )
    try:
        model.config.use_cache = False
    except Exception:
        pass

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    patch_model_from_manifest(model, manifest, factor_dtype=factor_dtype)

    print(f"[Load] Loading state dict from {state_path} ...", flush=True)
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) < 20:
            print("  missing:", missing)
        if len(unexpected) < 20:
            print("  unexpected:", unexpected)

    model = model.to(args.device).eval()

    if args.run_ppl:
        datasets = [d.strip() for d in args.ppl_datasets.split(",") if d.strip()]
        has_c4_stream_alias = any("c4_stream" in d.lower() for d in datasets)
        if args.c4_val_stream == "on" or (args.c4_val_stream == "auto" and has_c4_stream_alias):
            os.environ["SVDLLM_C4_VAL_STREAM"] = "1"
        elif args.c4_val_stream == "off":
            os.environ["SVDLLM_C4_VAL_STREAM"] = "0"
        if args.c4_val_docs is not None and int(args.c4_val_docs) > 0:
            os.environ["SVDLLM_C4_VAL_DOCS"] = str(int(args.c4_val_docs))
        if args.c4_val_dataset:
            os.environ["SVDLLM_C4_VAL_DATASET"] = str(args.c4_val_dataset)
        if any("c4" in d.lower() for d in datasets):
            print(
                "[PPL] C4 config:",
                {
                    "stream": os.getenv("SVDLLM_C4_VAL_STREAM", "0"),
                    "docs": os.getenv("SVDLLM_C4_VAL_DOCS", "2000"),
                    "dataset": os.getenv("SVDLLM_C4_VAL_DATASET", "allenai/c4"),
                },
                flush=True,
            )
        ppl_eval(
            model,
            tokenizer,
            datasets=datasets,
            model_seq_len=args.ppl_seq_len,
            batch_size=args.ppl_batch_size,
            device=args.device,
            label="SAES Token PPL",
            max_batches=args.ppl_max_batches,
        )

    if args.run_lm_eval:
        add_bos_token = None
        if args.add_bos_token == "true":
            add_bos_token = True
        elif args.add_bos_token == "false":
            add_bos_token = False
        prefix_token_id = args.prefix_token_id
        if prefix_token_id is None and getattr(tokenizer, "bos_token_id", None) is not None:
            prefix_token_id = tokenizer.bos_token_id
        if args.add_bos_token == "auto":
            if getattr(tokenizer, "bos_token_id", None) is None:
                add_bos_token = None
            elif prefix_token_id == tokenizer.bos_token_id:
                add_bos_token = False
            else:
                add_bos_token = True

        out_json = args.out_json
        if out_json is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_json = os.path.join(ckpt_dir, f"lm_eval_results.{ts}.json")

        run_lm_eval(
            model,
            tokenizer,
            device=args.device,
            tasks=args.tasks,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            max_length=args.max_length,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            include_path=args.include_path,
            add_bos_token=add_bos_token,
            prefix_token_id=prefix_token_id,
            out_json=out_json,
            disable_tqdm=(args.tqdm == "off"),
        )


if __name__ == "__main__":
    main()
