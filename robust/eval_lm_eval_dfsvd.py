#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a DF-SVD checkpoint (saved by robust/df_svd.py) with LM Evaluation Harness.

The DF-SVD checkpoint directory is expected to contain:
  - dfsvd_state.pt
  - dfsvd_manifest.json
  - tokenizer files (tokenizer.json / tokenizer.model / tokenizer_config.json ...)

Example:
CUDA_VISIBLE_DEVICES=0 python robust/eval_lm_eval_dfsvd.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --ckpt_dir robust/llama2_dfsvd_r0.4_full \
  --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande \
  --num_fewshot 0 \
  --batch_size 4 \
  --device cuda
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_LM_EVAL_ROOT = os.path.join(_REPO_ROOT, "lm-evaluation-harness")
if os.path.isdir(_LM_EVAL_ROOT) and _LM_EVAL_ROOT not in sys.path:
    sys.path.insert(0, _LM_EVAL_ROOT)


def parse_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


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
    # lm-eval results may contain non-JSON-serializable objects (e.g., callables).
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
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    except Exception as e:
        # Fallback: write just the per-task results dict (usually what's needed).
        minimal = {"results": data.get("results", data), "error": repr(e)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(minimal, f, ensure_ascii=False, indent=2, default=_json_default)


class DFSVDFactorizedLinear(nn.Module):
    """
    Minimal DF-SVD linear for evaluation / state_dict loading.

    Row-major:
      x: (..., in)
      xk = x @ Wv^T           -> (..., k)
      Wu = Wp + Bm@Am         -> (out, k)
      y  = xk @ Wu^T          -> (..., out)
    """

    def __init__(self, in_features: int, out_features: int, rank_k: int, update_rank: int, dtype: torch.dtype):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank_k = int(rank_k)
        self.update_rank = int(update_rank)

        self.register_buffer("Wv", torch.empty((rank_k, in_features), dtype=dtype))
        self.register_buffer("Wp", torch.empty((out_features, rank_k), dtype=dtype))
        if update_rank > 0:
            self.Bm = nn.Parameter(torch.empty((out_features, update_rank), dtype=dtype))
            self.Am = nn.Parameter(torch.empty((update_rank, rank_k), dtype=dtype))
        else:
            self.Bm = None
            self.Am = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        y2d = self.forward_flat(x2d)
        return y2d.reshape(*orig_shape[:-1], self.out_features)

    def forward_flat(self, x2d: torch.Tensor) -> torch.Tensor:
        in_dtype = x2d.dtype
        xk = x2d.to(self.Wv.dtype) @ self.Wv.t()
        Wu = self.Wp
        if self.update_rank > 0 and self.Bm is not None and self.Am is not None:
            Wu = Wu + self.Bm @ self.Am
        y = xk @ Wu.t()
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
        raise TypeError("dfsvd_manifest.json: expected 'items' to be a list")
    for it in items:
        name = it["name"]
        in_f = int(it["in_features"])
        out_f = int(it["out_features"])
        k = int(it["rank_k"])
        r = int(it["update_rank"])
        new_mod = DFSVDFactorizedLinear(in_features=in_f, out_features=out_f, rank_k=k, update_rank=r, dtype=factor_dtype)
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
) -> None:
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
        # lm-eval's evaluator does not pass `disable_tqdm` through to model methods,
        # so we wrap the model API to force-disable progress bars.
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
                if metrics:
                    print(f"  {t}: {metrics}", flush=True)
                else:
                    print(f"  {t}: {res[t]}", flush=True)
            else:
                print(f"  {t}: (no entry)", flush=True)
    else:
        print(res if res else results, flush=True)
    if out_json:
        print(f"[LM-Eval] Wrote full results to: {out_json}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True, help="HF model id (base architecture), e.g. meta-llama/Llama-2-7b-hf")
    ap.add_argument("--ckpt_dir", type=str, required=True, help="DF-SVD checkpoint directory saved by robust/df_svd.py")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype for evaluation (float16/bfloat16/float32).")
    ap.add_argument("--factor_dtype", type=str, default="float32", help="Factor dtype to instantiate DF-SVD modules (float16/bfloat16/float32).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--tasks", type=str, default="arc_easy,arc_challenge,hellaswag,piqa,winogrande")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include_path", type=str, default=None)
    ap.add_argument(
        "--add_bos_token",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="LM-Eval add_bos_token behavior (auto tries to avoid double BOS).",
    )
    ap.add_argument("--prefix_token_id", type=int, default=None)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument(
        "--tqdm",
        type=str,
        default="off",
        choices=["off", "on"],
        help="Show lm-eval progress bars. Use 'off' to avoid extremely long logs in non-TTY sessions.",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Write full lm-eval results JSON to this path (defaults to <ckpt_dir>/lm_eval_results.<ts>.json).",
    )
    args = ap.parse_args()

    ckpt_dir = args.ckpt_dir
    manifest_path = os.path.join(ckpt_dir, "dfsvd_manifest.json")
    state_path = os.path.join(ckpt_dir, "dfsvd_state.pt")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing {manifest_path}")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Missing {state_path}")

    model_dtype = parse_dtype(args.dtype)
    factor_dtype = parse_dtype(args.factor_dtype)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True, token=args.hf_token)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Resolve lm-eval add_bos_token / prefix token
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
