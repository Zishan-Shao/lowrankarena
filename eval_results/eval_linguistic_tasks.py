import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import torch

# Ensure repo root is on PYTHONPATH
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LM_EVAL_ROOT = os.path.join(_REPO_ROOT, "lm-evaluation-harness")
if os.path.isdir(_LM_EVAL_ROOT) and _LM_EVAL_ROOT not in sys.path:
    sys.path.insert(0, _LM_EVAL_ROOT)

from utils.model_utils import get_model_from_local, get_model_from_huggingface


def _parse_tasks(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


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
        res = evaluator.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
            limit=limit,
        )
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
                res = evaluator.simple_evaluate(
                    model=model,
                    tasks=[task],
                    num_fewshot=num_fewshot,
                    batch_size=batch_size,
                    max_batch_size=max_batch_size,
                    device=device,
                    limit=limit,
                )
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
        description="Evaluate linguistic tasks (lm-eval) for local or Dobi checkpoints."
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model id or local directory (from save_pretrained). If provided, overrides --checkpoint.",
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
    p.add_argument("--dobi_revision", type=str, default=None)
    p.add_argument("--dobi_cache_dir", type=str, default=None)
    p.add_argument("--dobi_remapping", action="store_true")
    p.add_argument("--dobi_unremapping", action="store_true")
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
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--output_md", type=str, default=None)
    args = p.parse_args()

    if args.dobi_model:
        if args.dobi_remapping and args.dobi_unremapping:
            raise ValueError("Only one of --dobi_remapping / --dobi_unremapping can be set.")
    else:
        if not args.model and not args.checkpoint:
            raise ValueError("Please provide --model, --checkpoint or --dobi_model.")
        if args.checkpoint and not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if args.dobi_model:
        remap_flag = True if args.dobi_remapping else (False if args.dobi_unremapping else None)
        model, tokenizer, _ = _load_dobi_model(
            args.dobi_model,
            hf_token=args.hf_token,
            revision=args.dobi_revision,
            cache_dir=args.dobi_cache_dir,
            remapping=remap_flag,
        )
        model_name = args.dobi_model
    else:
        if args.model:
            model, tokenizer = get_model_from_huggingface(args.model, hf_token=args.hf_token)
            model_name = args.model
        else:
            model, tokenizer = get_model_from_local(args.checkpoint)
            model_name = args.checkpoint

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
        max_batch_size=64,
        trust_remote_code=True,
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

    all_results = {
        "model": model_name,
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
            max_batch_size=64,
            device=args.device,
            limit=args.limit,
        )
        if not res or not res.get("results"):
            print(f"[LM-Eval] No results for set '{set_name}' after filtering; skipping.")
            continue
        tasks_used[set_name] = list(used) if used else list(set_tasks)
        print(res.get("results", res))
        all_results["task_sets"][set_name] = res

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(_jsonify(all_results), f, indent=2)
    if args.output_md:
        if len(all_results["task_sets"]) == 1:
            only_name = next(iter(all_results["task_sets"]))
            only_tasks = tasks_used.get(only_name, task_sets[0][1])
            _write_md(args.output_md, model_name=model_name, tasks=only_tasks, res=all_results["task_sets"][only_name])
        else:
            lines = []
            lines.append("# Linguistic Task Evaluation")
            lines.append("")
            lines.append(f"Model: {model_name}")
            lines.append("")
            for set_name, res in all_results["task_sets"].items():
                lines.append(f"## Task set: {set_name}")
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
            with open(args.output_md, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))


if __name__ == "__main__":
    main()
