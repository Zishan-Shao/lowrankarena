import argparse
import json
import os
import sys
import datetime as _dt
import re
import inspect
from typing import List, Optional, Tuple

import torch

# Ensure repo root is on PYTHONPATH (important if your HF repo's remote code imports `modules/` etc.)
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# If you vend lm-eval harness as a submodule in this repo, keep it importable.
_LM_EVAL_ROOT = os.path.join(_REPO_ROOT, "lm-evaluation-harness")
if os.path.isdir(_LM_EVAL_ROOT) and _LM_EVAL_ROOT not in sys.path:
    sys.path.insert(0, _LM_EVAL_ROOT)


def _parse_tasks(s: str) -> List[str]:
    return [t.strip() for t in (s or "").split(",") if t.strip()]


def _parse_task_sets(s: str) -> List[Tuple[str, List[str]]]:
    sets: List[Tuple[str, List[str]]] = []
    for idx, chunk in enumerate([c for c in (s or "").split(";") if c.strip()]):
        name = None
        if ":" in chunk:
            name, chunk = chunk.split(":", 1)
            name = name.strip() or None
        tasks = _parse_tasks(chunk)
        sets.append((name or f"set_{idx+1}", tasks))
    return sets


def _filter_existing_tasks(tasks: List[str], task_manager) -> List[str]:
    if task_manager is None:
        return tasks
    try:
        avail = set(getattr(task_manager, "all_tasks"))
    except Exception:
        try:
            avail = set(task_manager.list_all_tasks())
        except Exception:
            return tasks
    keep = [t for t in tasks if t in avail]
    drop = [t for t in tasks if t not in avail]
    if drop:
        print(f"[LM-Eval] Skipping unavailable tasks: {', '.join(drop)}")
    return keep


def _dtype_from_str(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    m = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return m.get(dtype.lower())


def _to_device(model: torch.nn.Module, device: str, dtype: Optional[str]) -> torch.nn.Module:
    td = _dtype_from_str(dtype)
    if td is not None:
        model = model.to(dtype=td)
    return model.to(device)


def _hf_from_pretrained_with_token_retry(cls, *args, hf_token: Optional[str], **kwargs):
    """
    Transformers changed auth kwarg naming across versions (token vs use_auth_token).
    We try token= first, then fall back to use_auth_token=.
    """
    if hf_token is None:
        return cls.from_pretrained(*args, **kwargs)

    try:
        return cls.from_pretrained(*args, token=hf_token, **kwargs)
    except TypeError:
        return cls.from_pretrained(*args, use_auth_token=hf_token, **kwargs)


def _load_hf_model_and_tokenizer(
    model_id_or_path: str,
    tokenizer_id_or_path: Optional[str],
    hf_token: Optional[str],
    revision: Optional[str],
    cache_dir: Optional[str],
    trust_remote_code: bool,
    dtype: Optional[str],
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "transformers is required to load ASVD HuggingFace repos. "
            "Install it (and sentencepiece for LLaMA tokenizers) and retry.\n"
            f"Original error: {e}"
        )

    torch_dtype = _dtype_from_str(dtype)

    model_kwargs = dict(
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    # If user asked for dtype, load directly in that dtype to save memory.
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = _hf_from_pretrained_with_token_retry(
        AutoModelForCausalLM,
        model_id_or_path,
        hf_token=hf_token,
        **model_kwargs,
    )

    tok_src = tokenizer_id_or_path or model_id_or_path
    tok_kwargs = dict(
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    # Prefer fast tokenizer, fallback to slow if needed.
    try:
        tokenizer = _hf_from_pretrained_with_token_retry(
            AutoTokenizer,
            tok_src,
            hf_token=hf_token,
            use_fast=True,
            **tok_kwargs,
        )
    except Exception:
        tokenizer = _hf_from_pretrained_with_token_retry(
            AutoTokenizer,
            tok_src,
            hf_token=hf_token,
            use_fast=False,
            **tok_kwargs,
        )

    # LLaMA-like tokenizers often have no pad token; lm-eval expects padding.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Also align generation config if present
    try:
        if getattr(model, "generation_config", None) is not None and tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    return model, tokenizer


def _write_md(path: str, model_name: str, tasks: List[str], res: dict) -> None:
    lines = []
    lines.append("# Linguistic Task Evaluation")
    lines.append("")
    lines.append(f"Model: {model_name}")
    lines.append(f"Tasks: {', '.join(tasks)}")
    lines.append("")
    lines.append("| Task | Metric | Value |")
    lines.append("|---|---|---:|")
    for task, metrics in res.get("results", {}).items():
        for metric, val in metrics.items():
            try:
                v = float(val)
                lines.append(f"| {task} | {metric} | {v:.4f} |")
            except Exception:
                lines.append(f"| {task} | {metric} | {val} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _jsonify(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)

def _safe_tag(s: str) -> str:
    s = os.path.basename((s or "").rstrip("/")) or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _auto_output_json(args, suffix: str) -> Optional[str]:
    if args.output_json:
        return args.output_json
    if not getattr(args, "output_dir", None):
        return None
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = getattr(args, "run_name", None) or f"{_safe_tag(args.model)}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return os.path.join(args.output_dir, f"{run_name}_{suffix}.json")


def _remove_lmeval_samples(obj):
    """Drop per-sample logs from lm-eval outputs to keep JSON small."""
    if not isinstance(obj, dict):
        return obj
    if "samples" not in obj:
        return obj
    out = dict(obj)
    out.pop("samples", None)
    return out


def _shrink_lmeval_output(obj) -> dict:
    """Keep only the most useful + compact pieces of lm-eval output."""
    if not isinstance(obj, dict):
        return {"value": str(obj)}
    obj = _remove_lmeval_samples(obj)
    out = {}
    out["results"] = obj.get("results", {})
    for k in (
        "config",
        "versions",
        "n-shot",
        "n-samples",
        "higher_is_better",
        "git_hash",
        "date",
        "errors",
        "groups",
        "group_subtasks",
    ):
        if k in obj:
            out[k] = obj[k]
    return out



def _is_dataset_access_error(err: Exception) -> bool:
    try:
        from datasets.exceptions import DatasetNotFoundError
        if isinstance(err, DatasetNotFoundError):
            return True
    except Exception:
        pass
    msg = str(err).lower()
    return "gated dataset" in msg or "datasetnotfounderror" in msg


def _safe_simple_evaluate(
    evaluator,
    model,
    tasks: List[str],
    num_fewshot: int,
    batch_size: int,
    max_batch_size: int,
    device: str,
    limit: Optional[int],
):
    try:
        kwargs = dict(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
            limit=limit,
        )
        # Keep returned object compact (avoid per-sample logs).
        try:
            sig = inspect.signature(evaluator.simple_evaluate)
            if "log_samples" in sig.parameters:
                kwargs["log_samples"] = False
        except Exception:
            pass
        res = evaluator.simple_evaluate(**kwargs)
        if res is None:
            raise RuntimeError("LM Evaluation Harness returned no results (not rank 0).")
        return res, list(tasks)
    except Exception as err:
        if not _is_dataset_access_error(err):
            raise
        print(f"[LM-Eval] Dataset access error: {err}. Falling back to per-task evaluation.")
        combined = {"results": {}, "errors": {}}
        used_tasks: List[str] = []
        for task in tasks:
            try:
                kwargs = dict(
                    model=model,
                    tasks=[task],
                    num_fewshot=num_fewshot,
                    batch_size=batch_size,
                    max_batch_size=max_batch_size,
                    device=device,
                    limit=limit,
                )
                try:
                    sig = inspect.signature(evaluator.simple_evaluate)
                    if "log_samples" in sig.parameters:
                        kwargs["log_samples"] = False
                except Exception:
                    pass
                res = evaluator.simple_evaluate(**kwargs)
                if res is None:
                    continue
                combined["results"].update(res.get("results", {}))
                used_tasks.append(task)
            except Exception as task_err:
                if _is_dataset_access_error(task_err):
                    print(f"[LM-Eval] Skipping gated/unavailable task: {task} ({task_err})")
                    combined["errors"][task] = str(task_err)
                    continue
                raise
        return combined, used_tasks


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate linguistic tasks (lm-eval) for ASVD HuggingFace repos (local folder or HF id)."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="ASVD HF model id or local directory (e.g. ./huggingface_repos/Llama-2-7b-hf-asvd40).",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer id/path. Use this if your ASVD HF folder does not include tokenizer files.",
    )
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--trust_remote_code", action="store_true", help="Enable custom modeling/config code. (Recommended for ASVD.)")

    p.add_argument(
        "--tasks",
        type=str,
        default="blimp,cola",
        help="Comma-separated lm-eval task/group names.",
    )
    p.add_argument(
        "--extra_tasks",
        type=str,
        default="mela_en,lingoly,zhoblimp",
        help="Optional extra linguistic tasks to include if available (comma-separated).",
    )
    p.add_argument(
        "--task_sets",
        type=str,
        default=None,
        help="Semicolon-separated task sets to evaluate, e.g. 'base:blimp,cola;plus:blimp,cola,mela_en'.",
    )

    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--hf_token", type=str, default=None)

    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--output_md", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None, help="If set (and --output_json not set), auto-write JSON to <output_dir>/<run_name>_ling.json")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name prefix for auto JSON naming")
    p.add_argument("--json_full", action="store_true", help="Include full lm-eval output in JSON (can still be large depending on lm-eval version).")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model, tokenizer = _load_hf_model_and_tokenizer(
        model_id_or_path=args.model,
        tokenizer_id_or_path=args.tokenizer,
        hf_token=args.hf_token,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )

    # If you loaded without torch_dtype above (or you want explicit casting), do it here.
    model = _to_device(model, args.device, args.dtype)
    model.eval()
    try:
        model.config.use_cache = False
    except Exception:
        pass

    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except Exception as e:
        raise RuntimeError(f"lm-eval harness is required: {e}")

    tasks = _parse_tasks(args.tasks)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        trust_remote_code=args.trust_remote_code,
    )

    try:
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager()
    except Exception:
        task_manager = None

    task_sets: List[Tuple[str, List[str]]] = []
    if args.task_sets:
        task_sets = _parse_task_sets(args.task_sets)
    else:
        extras = _parse_tasks(args.extra_tasks) if args.extra_tasks else []
        if extras:
            tasks = tasks + extras
        task_sets = [("default", tasks)]

    out_json = _auto_output_json(args, "ling")
    all_results = {
        "schema": "asvd_eval_v1",
        "script": os.path.basename(__file__),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join(sys.argv),
        "mode": "linguistic",
        "model": args.model,
        "args": vars(args),
        "task_sets": {},
    }
    tasks_used = {}

    for set_name, set_tasks in task_sets:
        set_tasks = _filter_existing_tasks(set_tasks, task_manager)
        if not set_tasks:
            print(f"[LM-Eval] No valid tasks for set '{set_name}', skipping.")
            continue

        res, used = _safe_simple_evaluate(
            evaluator=evaluator,
            model=lm,
            tasks=set_tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            limit=args.limit,
        )
        if not res or not res.get("results"):
            print(f"[LM-Eval] No results for set '{set_name}' after filtering; skipping.")
            continue

        tasks_used[set_name] = list(used) if used else list(set_tasks)
        print(res.get("results", res))
        # Keep JSON artifact small by default.
        all_results["task_sets"][set_name] = (_remove_lmeval_samples(res) if args.json_full else _shrink_lmeval_output(res))

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(_jsonify(all_results), f, indent=2)
        print(f"[Output] Wrote JSON -> {out_json}")

    if args.output_md:
        if len(all_results["task_sets"]) == 1:
            only_name = next(iter(all_results["task_sets"]))
            only_tasks = tasks_used.get(only_name, task_sets[0][1])
            _write_md(
                args.output_md,
                model_name=args.model,
                tasks=only_tasks,
                res=all_results["task_sets"][only_name],
            )
        else:
            lines = ["# Linguistic Task Evaluation", "", f"Model: {args.model}", ""]
            for set_name, res in all_results["task_sets"].items():
                lines += [f"## Task set: {set_name}", "", "| Task | Metric | Value |", "|---|---|---:|"]
                for task, metrics in res.get("results", {}).items():
                    for metric, val in metrics.items():
                        try:
                            v = float(val)
                            lines.append(f"| {task} | {metric} | {v:.4f} |")
                        except Exception:
                            lines.append(f"| {task} | {metric} | {val} |")
                lines.append("")
            with open(args.output_md, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))


if __name__ == "__main__":
    main()
