import argparse
import math
import os
import sys
import ast
import contextlib
import datetime as _dt
import io
import json
import time
import re
from typing import List, Optional

import torch
from tqdm import tqdm

'''
1) 默认：token‑level PPL（wikitext2/ptb/c4）
CUDA_VISIBLE_DEVICES=3 python eval_general_ppl.py \
  --checkpoint ./checkpoints/jeffwan_llama_7b_hf_act_lora_lmwhiten_mixedlora_0.4.pt \
  --device cuda --seqlen 2048 --batch_size 4 --dtype bfloat16
  
  
2) 快速 smoke test（限制 batch 数）
CUDA_VISIBLE_DEVICES=3 python eval_general_ppl.py \
  --checkpoint ./checkpoints/jeffwan_llama_7b_hf_act_lora_lmwhiten_mixedlora_0.4.pt \
  --max_batches 50 --device cuda --seqlen 2048 --batch_size 4 --dtype bfloat16

3) 额外算 word/byte/bpb（会用 lm‑eval）
CUDA_VISIBLE_DEVICES=3 python eval_general_ppl.py \
  --checkpoint ./checkpoints/jeffwan_llama_7b_hf_act_lora_lmwhiten_mixedlora_0.4.pt \
  --metrics token,word,byte,bpb \
  --device cuda --seqlen 2048 --batch_size 4 --dtype bfloat16


更多的C4:
CUDA_VISIBLE_DEVICES=3 python eval_general_ppl.py \
  --checkpoint ./checkpoints/jeffwan_llama_7b_hf_act_lora_lmwhiten_mixedlora_0.4.pt \
  --datasets wikitext2,ptb,c4 \
  --c4_stream --c4_docs 2000 \
  --device cuda --seqlen 2048 --batch_size 4 --dtype bfloat16

Evaluate Dobi:
CUDA_VISIBLE_DEVICES=3 python eval_general_ppl.py \
  --dobi_model Qinsi1/DobiSVD-Llama-2-7b-hf-0.4 \
  --datasets wikitext2,ptb,c4 --device cuda --dtype bfloat16

Legacy PPL (baseline-style sample mean; tends to be lower):
CUDA_VISIBLE_DEVICES=3 python eval_general_ppl.py \
  --checkpoint ./checkpoints/jeffwan_llama_7b_hf_act_lora_lmwhiten_mixedlora_0.4.pt \
  --datasets wikitext2,ptb,c4 --ppl_method legacy \
  --device cuda --seqlen 2048 --batch_size 4 --dtype bfloat16

跑我们当前 checkpoint（legacy 方法）

CUDA_VISIBLE_DEVICES=3 python eval_general_ppl.py \
  --checkpoint ./checkpoints/mixed_calibrate/llama_2_7b_hf_act_lora_mixedwhiten_mixedlora_0.4_enhanced.pt \
  --datasets wikitext2,ptb,c4 \
  --ppl_method legacy \
  --device cuda --seqlen 2048 --batch_size 4 --dtype bfloat16
顺便对比 Dobi（同一 legacy 方法，公平）

CUDA_VISIBLE_DEVICES=3 python eval_general_ppl.py \
  --compare_dobi \
  --checkpoint ./checkpoints/mixed_calibrate/llama_2_7b_hf_act_lora_mixedwhiten_mixedlora_0.4_enhanced.pt \
  --dobi_model Qinsi1/DobiSVD-Llama-2-7b-hf-0.4 \
  --datasets wikitext2,ptb,c4 \
  --ppl_method legacy \
  --device cuda --seqlen 2048 --batch_size 4 --dtype bfloat16
'''
# ----------------------------------------------------------------------------
# JSON output helpers (similar to ASVD scripts)
# ----------------------------------------------------------------------------

def _jsonify(obj):
    """Make common python/torch/numpy objects JSON-serializable."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _safe_tag(s: str) -> str:
    s = os.path.basename(str(s)).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_") or "run"


def _auto_output_json(args, suffix: str):
    """
    Priority:
      1) --output_json
      2) --output_dir + (<run_name>_<suffix>.json)
      3) None
    """
    out_json = getattr(args, "output_json", None)
    if out_json:
        return out_json
    out_dir = getattr(args, "output_dir", None)
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    run_name = getattr(args, "run_name", None)
    if not run_name:
        base = getattr(args, "checkpoint", None) or getattr(args, "dobi_model", None) or "model"
        run_name = f"{_safe_tag(base)}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return os.path.join(out_dir, f"{run_name}_{suffix}.json")


def _write_json(path: str, payload):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2)
    print(f"[Output] Wrote JSON -> {path}")


class _Tee:
    """File-like that writes to multiple streams (keeps CLI output unchanged while capturing)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def _extract_balanced_braces(text: str):
    """Extract balanced {...} substrings from text."""
    out = []
    level = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if level == 0:
                start = i
            level += 1
        elif ch == "}":
            if level > 0:
                level -= 1
                if level == 0 and start is not None:
                    out.append(text[start : i + 1])
                    start = None
    return out


def _parse_best_dict(text: str, want_keys=None):
    """Try to parse a useful dict from captured stdout."""
    candidates = []
    for chunk in _extract_balanced_braces(text):
        try:
            val = ast.literal_eval(chunk)
        except Exception:
            continue
        if isinstance(val, dict):
            candidates.append(val)
    if not candidates:
        return None
    if want_keys:
        want = set(want_keys)
        best = None
        best_score = -1
        for d in candidates:
            try:
                keys = set(map(str, d.keys()))
            except Exception:
                continue
            score = len(keys & want)
            if score > best_score:
                best = d
                best_score = score
        if best is not None and best_score > 0:
            return best
    return candidates[-1]


def _call_and_capture_dict(fn, want_keys=None, **kwargs):
    """Call fn(**kwargs) while capturing stdout; return dict if fn returns it or prints it."""
    buf = io.StringIO()
    tee = _Tee(sys.stdout, buf)
    with contextlib.redirect_stdout(tee):
        ret = fn(**kwargs)
    if isinstance(ret, dict):
        return ret
    text = buf.getvalue()
    return _parse_best_dict(text, want_keys=want_keys)


 
  


# Ensure repo root is on PYTHONPATH
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from utils.model_utils import get_model_from_local, get_model_from_huggingface
from evaluater import ppl_eval


def _parse_datasets(s: str) -> List[str]:
    return [d.strip() for d in s.split(",") if d.strip()]


def _parse_dataset_sets(s: str) -> List[tuple]:
    sets = []
    for idx, chunk in enumerate([c for c in (s or "").split(";") if c.strip()]):
        name = None
        if ":" in chunk:
            name, chunk = chunk.split(":", 1)
            name = name.strip() or None
        datasets = _parse_datasets(chunk)
        sets.append((name or f"set_{idx+1}", datasets))
    return sets


def _to_device(model: torch.nn.Module, device: str, dtype: Optional[str]) -> torch.nn.Module:
    if dtype is not None:
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        target_dtype = dtype_map.get(dtype.lower())
        if target_dtype is not None:
            model = model.to(dtype=target_dtype)
    return model.to(device)


def _resolve_dobi_path(model_id: str, hf_token: Optional[str], revision: Optional[str], cache_dir: Optional[str]) -> str:
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub is required to download Dobi checkpoints: {e}")
    return snapshot_download(repo_id=model_id, revision=revision, cache_dir=cache_dir, token=hf_token)


@torch.no_grad()
def _legacy_ppl_eval(
    model,
    tokenizer,
    datasets: List[str],
    model_seq_len: int,
    batch_size: int,
    device: str,
    label: str,
    max_batches: Optional[int] = None,
):

    """
    Legacy sample-mean PPL: mimic the provided baseline code.
    Uses input_ids[:, :-1], then shifts logits again, and normalizes by (num_samples * seqlen).
    This intentionally underestimates PPL vs strict token-level averaging.
    """
    from utils.data_utils import get_test_data

    model.to(device)
    model.eval()
    ppls = {}
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
        loss_sum = 0.0
        num_samples = 0
        seq_len = None
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        for i, batch in enumerate(tqdm(test_loader, desc=f"legacy_ppl[{dataset}]")):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch.to(device)
            if seq_len is None:
                seq_len = int(batch.shape[1])
            input_ids = batch[:, :-1]
            output = model(
                input_ids,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = output.logits if hasattr(output, "logits") else output[0]
            if not torch.isfinite(logits).all():
                continue
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss_sum += loss.sum().item()
            num_samples += input_ids.shape[0]
        if num_samples == 0 or seq_len is None:
            ppls[dataset] = float("nan")
            continue
        denom = float(num_samples * seq_len)
        ppls[dataset] = float(math.exp(loss_sum / denom))
    print(f"{label} (legacy): {ppls}")
    return ppls


def _load_dobi_model(
    model_id: str,
    hf_token: Optional[str],
    revision: Optional[str],
    cache_dir: Optional[str],
    remapping: Optional[bool],
):
    dobi_root = os.path.join(_REPO_ROOT, "baselines", "Dobi-SVD")
    if dobi_root not in sys.path:
        sys.path.insert(0, dobi_root)
    try:
        from modelutils import load_remapping_model, load_unremapping_model
    except Exception as e:
        raise RuntimeError(f"Failed to import Dobi-SVD loaders from {dobi_root}: {e}")
    local_path = _resolve_dobi_path(model_id, hf_token=hf_token, revision=revision, cache_dir=cache_dir)
    if remapping is None:
        if os.path.exists(os.path.join(local_path, "remapping_weight.pt")):
            remapping = True
        elif os.path.exists(os.path.join(local_path, "DobiSVD_Model.pt")):
            remapping = False
        else:
            raise FileNotFoundError(
                f"Could not find remapping_weight.pt or DobiSVD_Model.pt under {local_path}"
            )
    if remapping:
        model, tokenizer = load_remapping_model(local_path)
    else:
        model, tokenizer = load_unremapping_model(local_path)
    return model, tokenizer, local_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a local checkpoint on general LM test sets (PPL)."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint saved by this repo (contains {'model','tokenizer'}).",
    )
    p.add_argument(
        "--dobi_model",
        type=str,
        default=None,
        help="Dobi-SVD checkpoint (HF repo id or local dir).",
    )
    p.add_argument(
        "--dobi_revision",
        type=str,
        default=None,
        help="Optional HF revision for Dobi-SVD model.",
    )
    p.add_argument(
        "--dobi_cache_dir",
        type=str,
        default=None,
        help="Optional HF cache dir for Dobi-SVD model.",
    )
    p.add_argument(
        "--dobi_remapping",
        action="store_true",
        help="Force Dobi remapping loader.",
    )
    p.add_argument(
        "--dobi_unremapping",
        action="store_true",
        help="Force Dobi unremapping loader.",
    )
    p.add_argument(
        "--compare_dobi",
        action="store_true",
        help="Run both --checkpoint and --dobi_model sequentially for comparison.",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="wikitext2,ptb,c4",
        help="Comma-separated dataset names. Supported: wikitext2, wikitext2_val, ptb, c4.",
    )
    p.add_argument(
        "--dataset_sets",
        type=str,
        default=None,
        help="Semicolon-separated dataset sets, e.g. 'base:wikitext2,ptb,c4;wt2:wikitext2'.",
    )
    p.add_argument("--seqlen", type=int, default=2048, help="Sequence length for evaluation.")
    p.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size.")
    p.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on.")
    p.add_argument("--hf_token", type=str, default=None, help="Optional Hugging Face token.")
    p.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Optional dtype override (float16, bfloat16, float32).",
    )
    p.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Limit number of batches per dataset (for quick smoke tests).",
    )
    p.add_argument(
        "--label",
        type=str,
        default="General PPL",
        help="Label printed with PPL results.",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default="token",
        help="Comma-separated metrics: token, word, byte, bpb. word/byte/bpb use lm-eval harness.",
    )
    p.add_argument(
        "--ppl_method",
        type=str,
        default="token",
        choices=["token", "legacy"],
        help="PPL computation method for token metric: token (default) or legacy (baseline sample-mean).",
    )
    p.add_argument(
        "--lm_eval_add_bos_token",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="For lm-eval metrics: whether to add BOS (auto avoids double BOS).",
    )
    p.add_argument(
        "--lm_eval_prefix_token_id",
        type=int,
        default=None,
        help="For lm-eval metrics: override rolling prefix token id (e.g., BOS token id).",
    )
    p.add_argument(
        "--c4_docs",
        type=int,
        default=None,
        help="Limit number of C4 validation documents (default 2000).",
    )
    p.add_argument(
        "--c4_stream",
        action="store_true",
        help="Use streaming C4 validation to avoid downloading shards.",
    )
    p.add_argument(
        "--auto_c4_stream",
        action="store_true",
        help="Automatically enable C4 streaming when C4 is requested (default: off).",
    )
    p.add_argument(
        "--c4_dataset",
        type=str,
        default=None,
        help="Override C4 dataset source (e.g., stas/c4-en-10k).",
    )
    p.add_argument(
        "--lm_eval_allow_c4_download",
        action="store_true",
        help="Allow lm-eval C4 task to download non-streaming shards (default: skip C4 for word/byte/bpb).",
    )

    # Optional: save results to JSON (similar to ASVD scripts)
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Write results to this JSON file (optional).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Write JSON into this directory with an auto-generated filename (optional).",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional name used when auto-generating filename inside --output_dir.",
    )

    args = p.parse_args()

    all_runs = {}
    models_info = []


    def _load_ours(ckpt_or_dir: str):
        # If it's a repo checkpoint (.pt), use the repo loader
        if ckpt_or_dir.endswith(".pt"):
            return get_model_from_local(ckpt_or_dir)
        # Otherwise treat it as HF model id OR local HF directory
        return get_model_from_huggingface(ckpt_or_dir, hf_token=args.hf_token)


    if args.dobi_model:
        if args.dobi_remapping and args.dobi_unremapping:
            raise ValueError("Only one of --dobi_remapping / --dobi_unremapping can be set.")
    if not args.checkpoint and not args.dobi_model:
        raise ValueError("Please provide --checkpoint or --dobi_model.")
    if args.checkpoint and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")


    def _run_one(model, tokenizer, label: str, datasets_override: Optional[List[str]] = None):
        model = _to_device(model, args.device, args.dtype)
        model.eval()

        if args.c4_docs is not None:
            os.environ["SVDLLM_C4_VAL_DOCS"] = str(int(args.c4_docs))
        if args.c4_stream:
            os.environ["SVDLLM_C4_VAL_STREAM"] = "1"
        elif args.auto_c4_stream:
            os.environ["SVDLLM_C4_VAL_STREAM"] = "1"
        if args.c4_dataset:
            os.environ["SVDLLM_C4_VAL_DATASET"] = args.c4_dataset

        datasets = datasets_override or _parse_datasets(args.datasets)
        # If user asked for C4 but didn't enable streaming, auto-stream to avoid downloads.
        if "c4" in datasets and not args.c4_stream and not args.c4_dataset:
            os.environ.setdefault("SVDLLM_C4_VAL_STREAM", "1")
            os.environ.setdefault("SVDLLM_C4_VAL_DOCS", "200")

        metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
        run_out = {
            "label": label,
            "datasets": list(datasets),
            "metrics": list(metrics),
            "ppl_method": args.ppl_method,
            "seqlen": int(args.seqlen),
            "batch_size": int(args.batch_size),
            "max_batches": args.max_batches,
        }

        if "token" in metrics:
            if args.ppl_method == "legacy":
                run_out["token_ppl"] = _legacy_ppl_eval(
                    model,
                    tokenizer,
                    datasets=datasets,
                    model_seq_len=args.seqlen,
                    batch_size=args.batch_size,
                    device=args.device,
                    label=label,
                    max_batches=args.max_batches,
                )
            else:
                run_out["token_ppl"] = _call_and_capture_dict(
                    ppl_eval,
                    want_keys=datasets,
                    model=model,
                    tokenizer=tokenizer,
                    datasets=datasets,
                    model_seq_len=args.seqlen,
                    batch_size=args.batch_size,
                    device=args.device,
                    label=label,
                    max_batches=args.max_batches,
                )

        if any(m in metrics for m in ("word", "byte", "bpb")):
            # Use lm-eval harness to compute word/byte/bpb for the same datasets.
            try:
                from lm_eval import evaluator
                from lm_eval.models.huggingface import HFLM
            except Exception as e:
                raise RuntimeError(f"lm-eval harness is required for word/byte/bpb metrics: {e}")
            # Disable KV-cache to avoid OOM or custom attention cache incompatibilities
            try:
                model.config.use_cache = False
            except Exception:
                pass

            lm_eval_datasets = list(datasets)
            if "c4" in lm_eval_datasets and not args.lm_eval_allow_c4_download:
                lm_eval_datasets.remove("c4")
                print("[LM-Eval] Skipping C4 for word/byte/bpb to avoid shard downloads. "
                      "Pass --lm_eval_allow_c4_download to enable.")

            # Resolve add_bos_token / prefix_token_id (avoid double BOS in rolling)
            add_bos_token = None
            if args.lm_eval_add_bos_token == "true":
                add_bos_token = True
            elif args.lm_eval_add_bos_token == "false":
                add_bos_token = False
            prefix_token_id = args.lm_eval_prefix_token_id
            if prefix_token_id is None and tokenizer.bos_token_id is not None:
                prefix_token_id = tokenizer.bos_token_id
            if args.lm_eval_add_bos_token == "auto":
                if tokenizer.bos_token_id is None:
                    add_bos_token = None
                elif prefix_token_id == tokenizer.bos_token_id:
                    add_bos_token = False
                else:
                    add_bos_token = True

            lm = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                device=args.device,
                batch_size=args.batch_size,
                max_batch_size=64,
                max_length=args.seqlen,
                trust_remote_code=True,
                add_bos_token=add_bos_token,
                prefix_token_id=prefix_token_id,
            )

            if not lm_eval_datasets:
                print("[LM-Eval] No datasets left for word/byte/bpb after filtering; skipping.")
                run_out["lm_eval"] = {}
                run_out["lm_eval_tasks"] = []
                return run_out

            res = evaluator.simple_evaluate(
                model=lm,
                tasks=lm_eval_datasets,
                num_fewshot=0,
                batch_size=args.batch_size,
                max_batch_size=64,
                device=args.device,
                limit=args.max_batches,
            )
            if res is None:
                raise RuntimeError("LM Evaluation Harness returned no results (not rank 0).")
            print("\nLM-Eval metrics (word/byte/bpb):")
            print(res.get("results", res))
            run_out["lm_eval"] = res.get("results", res)
            run_out["lm_eval_tasks"] = list(lm_eval_datasets)

        return run_out
    dataset_sets = _parse_dataset_sets(args.dataset_sets) if args.dataset_sets else None

    def _run_with_sets(model, tokenizer, label_prefix: str):
        if dataset_sets:
            for set_name, ds in dataset_sets:
                label = f"{label_prefix} [{set_name}]"
                all_runs[label] = _run_one(model, tokenizer, label=label, datasets_override=ds)
        else:
            all_runs[label_prefix] = _run_one(model, tokenizer, label=label_prefix)

    # ------------------------------------------------------------------
    # Run evaluation(s)
    # ------------------------------------------------------------------
    if args.compare_dobi:
        if not args.checkpoint or not args.dobi_model:
            raise ValueError("--compare_dobi requires both --checkpoint and --dobi_model.")

        print("[Compare] Evaluating our checkpoint...")
        models_info.append({"name": "ours", "source": args.checkpoint})
        model, tokenizer = _load_ours(args.checkpoint)
        _run_with_sets(model, tokenizer, label_prefix=f"{args.label} (ours)")

        print("[Compare] Evaluating Dobi checkpoint...")
        remap_flag = True if args.dobi_remapping else (False if args.dobi_unremapping else None)
        model, tokenizer, local_path = _load_dobi_model(
            args.dobi_model,
            hf_token=args.hf_token,
            revision=args.dobi_revision,
            cache_dir=args.dobi_cache_dir,
            remapping=remap_flag,
        )
        models_info.append({
            "name": "dobi",
            "source": args.dobi_model,
            "local_path": local_path,
            "revision": args.dobi_revision,
            "cache_dir": args.dobi_cache_dir,
            "remapping": remap_flag,
        })
        _run_with_sets(model, tokenizer, label_prefix=f"{args.label} (dobi)")

    elif args.dobi_model:
        remap_flag = True if args.dobi_remapping else (False if args.dobi_unremapping else None)
        model, tokenizer, local_path = _load_dobi_model(
            args.dobi_model,
            hf_token=args.hf_token,
            revision=args.dobi_revision,
            cache_dir=args.dobi_cache_dir,
            remapping=remap_flag,
        )
        models_info.append({
            "name": "dobi",
            "source": args.dobi_model,
            "local_path": local_path,
            "revision": args.dobi_revision,
            "cache_dir": args.dobi_cache_dir,
            "remapping": remap_flag,
        })
        _run_with_sets(model, tokenizer, label_prefix=args.label)

    else:
        models_info.append({"name": "ours", "source": args.checkpoint})
        model, tokenizer = _load_ours(args.checkpoint)
        _run_with_sets(model, tokenizer, label_prefix=args.label)

    # ------------------------------------------------------------------
    # Save JSON (optional)
    # ------------------------------------------------------------------
    out_json = _auto_output_json(args, "ppl")
    payload = {
        "schema": "svdllm_eval_v1",
        "script": os.path.basename(__file__),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join(sys.argv),
        "mode": "ppl",
        "args": vars(args),
        "models": models_info,
        "results": all_runs,
    }
    _write_json(out_json, payload)



if __name__ == "__main__":
    main()