import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import torch
import datetime as _dt
import re
import inspect


import time

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def _get_cpu_rss_bytes() -> Optional[int]:
    """Return current process RSS in bytes if available (psutil), else None."""
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            return None
    return None


def _get_cpu_maxrss_bytes() -> Optional[int]:
    """Return process max RSS in bytes (resource.ru_maxrss) if available."""
    try:
        import resource  # Unix-only

        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB; macOS: bytes.
        if sys.platform == "darwin":
            return int(v)
        return int(v) * 1024
    except Exception:
        return None


def _cuda_device_index(device: str) -> Optional[int]:
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return None
    s = str(device)
    if ":" in s:
        try:
            return int(s.split(":", 1)[1])
        except Exception:
            pass
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


def _maybe_cuda_sync(device: str) -> None:
    idx = _cuda_device_index(device)
    if idx is None:
        return
    try:
        torch.cuda.synchronize(idx)
    except Exception:
        torch.cuda.synchronize()


def _reset_cuda_peaks(device: str) -> None:
    idx = _cuda_device_index(device)
    if idx is None:
        return
    try:
        torch.cuda.reset_peak_memory_stats(idx)
    except Exception:
        torch.cuda.reset_peak_memory_stats()


def _cuda_mem_snapshot(device: str) -> Optional[dict]:
    idx = _cuda_device_index(device)
    if idx is None:
        return None
    try:
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(idx)),
            "reserved_bytes": int(torch.cuda.memory_reserved(idx)),
        }
    except Exception:
        return None


def _cuda_peak_snapshot(device: str) -> Optional[dict]:
    idx = _cuda_device_index(device)
    if idx is None:
        return None
    try:
        return {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(idx)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(idx)),
        }
    except Exception:
        return None


def _cuda_device_info(device: str) -> Optional[dict]:
    idx = _cuda_device_index(device)
    if idx is None:
        return None
    try:
        prop = torch.cuda.get_device_properties(idx)
        return {
            "index": int(idx),
            "name": str(prop.name),
            "total_memory_bytes": int(prop.total_memory),
        }
    except Exception:
        try:
            return {"index": int(idx), "name": str(torch.cuda.get_device_name(idx))}
        except Exception:
            return {"index": int(idx)}


class PerfRecorder:
    """Context manager to record wall time + CPU/GPU memory deltas for a code region."""

    def __init__(self, device: str, label: str = ""):
        self.device = device
        self.label = label

    def __enter__(self):
        _maybe_cuda_sync(self.device)
        _reset_cuda_peaks(self.device)
        self.t0 = time.perf_counter()
        self.cpu_rss0 = _get_cpu_rss_bytes()
        self.cpu_maxrss0 = _get_cpu_maxrss_bytes()
        self.gpu0 = _cuda_mem_snapshot(self.device)
        self.gpu_info = _cuda_device_info(self.device)
        return self

    def __exit__(self, exc_type, exc, tb):
        _maybe_cuda_sync(self.device)
        self.t1 = time.perf_counter()
        self.cpu_rss1 = _get_cpu_rss_bytes()
        self.cpu_maxrss1 = _get_cpu_maxrss_bytes()
        self.gpu1 = _cuda_mem_snapshot(self.device)
        self.gpu_peak = _cuda_peak_snapshot(self.device)

    def to_dict(self) -> dict:
        t0 = getattr(self, "t0", None)
        t1 = getattr(self, "t1", None)
        out = {
            "label": self.label,
            "wall_time_sec": float((t1 - t0) if (t0 is not None and t1 is not None) else 0.0),
        }

        # CPU
        if getattr(self, "cpu_rss0", None) is not None:
            out["cpu_rss_start_bytes"] = int(self.cpu_rss0)
        if getattr(self, "cpu_rss1", None) is not None:
            out["cpu_rss_end_bytes"] = int(self.cpu_rss1)
        if getattr(self, "cpu_rss0", None) is not None and getattr(self, "cpu_rss1", None) is not None:
            out["cpu_rss_delta_bytes"] = int(self.cpu_rss1 - self.cpu_rss0)

        if getattr(self, "cpu_maxrss0", None) is not None:
            out["cpu_maxrss_start_bytes"] = int(self.cpu_maxrss0)
        if getattr(self, "cpu_maxrss1", None) is not None:
            out["cpu_maxrss_end_bytes"] = int(self.cpu_maxrss1)
        if getattr(self, "cpu_maxrss0", None) is not None and getattr(self, "cpu_maxrss1", None) is not None:
            out["cpu_maxrss_delta_bytes"] = int(self.cpu_maxrss1 - self.cpu_maxrss0)

        # GPU
        if getattr(self, "gpu_info", None) is not None:
            out["gpu_device"] = self.gpu_info
        if getattr(self, "gpu0", None) is not None:
            out["gpu_start"] = self.gpu0
        if getattr(self, "gpu1", None) is not None:
            out["gpu_end"] = self.gpu1
        if getattr(self, "gpu_peak", None) is not None:
            out["gpu_peak"] = self.gpu_peak

        return out

# Ensure repo root is on PYTHONPATH
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LM_EVAL_ROOT = os.path.join(_REPO_ROOT, "lm-evaluation-harness")
if os.path.isdir(_LM_EVAL_ROOT) and _LM_EVAL_ROOT not in sys.path:
    sys.path.insert(0, _LM_EVAL_ROOT)

from utils.model_utils import get_model_from_local, get_model_from_huggingface
from utils.saes_svd_loader import looks_like_saes_svd_checkpoint, load_saes_svd_model
from utils.df_svd_loader import looks_like_dfsvd_checkpoint, load_dfsvd_model


# ----------------------------------------------------------------------------
# ASVD HuggingFace loader fallback
# ----------------------------------------------------------------------------


def _asvd_torch_dtype(dtype):
    if dtype is None:
        return None
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    return dtype_map.get(str(dtype).lower())


def _hf_from_pretrained_with_token_retry(cls, *args, hf_token=None, **kwargs):
    if hf_token is None:
        return cls.from_pretrained(*args, **kwargs)
    try:
        return cls.from_pretrained(*args, token=hf_token, **kwargs)
    except TypeError:
        return cls.from_pretrained(*args, use_auth_token=hf_token, **kwargs)


def _ensure_asvd_tokenizer_pad(tokenizer, padding_side="left"):
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    if padding_side is not None:
        try:
            tokenizer.padding_side = padding_side
        except Exception:
            pass


def _resolve_asvd_model_path(model_id_or_path, hf_token=None, revision=None, cache_dir=None):
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required to download HuggingFace/ASVD repos. "
            f"Original error: {e}"
        )
    return snapshot_download(
        repo_id=model_id_or_path,
        revision=revision,
        cache_dir=cache_dir,
        token=hf_token,
    )


def _read_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _import_module_from_file(module_name, file_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _find_asvd_module_file(module_name, search_roots):
    rel = module_name.replace(".", os.sep) + ".py"
    for root in search_roots:
        cand = os.path.join(root, rel)
        if os.path.isfile(cand):
            return cand
    base = os.path.basename(rel)
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if base in filenames:
                return os.path.join(dirpath, base)
    return None


def _asvd_search_roots(local_path):
    roots = []
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(this_dir, ".."))
    for root in (
        local_path,
        os.path.dirname(local_path),
        this_dir,
        parent_dir,
        os.path.join(this_dir, "huggingface_repos"),
        os.path.join(parent_dir, "huggingface_repos"),
    ):
        if root and root not in roots:
            roots.append(root)
    return roots


def _load_asvd_hf_model_and_tokenizer(
    model_id_or_path,
    *,
    tokenizer_id_or_path=None,
    hf_token=None,
    revision=None,
    cache_dir=None,
    trust_remote_code=True,
    dtype=None,
):
    local_path = _resolve_asvd_model_path(
        model_id_or_path,
        hf_token=hf_token,
        revision=revision,
        cache_dir=cache_dir,
    )

    search_roots = _asvd_search_roots(local_path)
    for root in search_roots:
        if os.path.isdir(root) and root not in sys.path:
            sys.path.insert(0, root)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers is required to load ASVD HuggingFace repos: {e}")

    torch_dtype = _asvd_torch_dtype(dtype)
    tok_src = tokenizer_id_or_path or local_path

    try:
        tokenizer = _hf_from_pretrained_with_token_retry(
            AutoTokenizer,
            tok_src,
            hf_token=hf_token,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except Exception:
        tokenizer = _hf_from_pretrained_with_token_retry(
            AutoTokenizer,
            tok_src,
            hf_token=hf_token,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
    _ensure_asvd_tokenizer_pad(tokenizer)

    model_kwargs = dict(
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    try:
        model = _hf_from_pretrained_with_token_retry(
            AutoModelForCausalLM,
            local_path,
            hf_token=hf_token,
            **model_kwargs,
        )
        return model, tokenizer, local_path
    except Exception as e_auto:
        print(f"[WARN] Standard transformers loader failed: {type(e_auto).__name__}: {e_auto}")
        print("[WARN] Trying ASVD auto_map fallback loader...")

    config_json = os.path.join(local_path, "config.json")
    if not os.path.isfile(config_json):
        raise FileNotFoundError(f"Could not find config.json under {local_path}")
    cfg_dict = _read_json_file(config_json)
    auto_map = cfg_dict.get("auto_map") or {}
    cfg_entry = auto_map.get("AutoConfig")
    model_entry = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel")
    if not cfg_entry or not model_entry:
        raise RuntimeError(
            "ASVD fallback loader requires AutoConfig and AutoModelForCausalLM/AutoModel entries in config.json"
        )

    def _split_entry(entry):
        if ":" in entry:
            mod, cls = entry.split(":", 1)
        else:
            parts = entry.split(".")
            if len(parts) < 2:
                raise ValueError(f"Invalid auto_map entry: {entry}")
            mod, cls = ".".join(parts[:-1]), parts[-1]
        return mod, cls

    cfg_mod_name, cfg_cls_name = _split_entry(cfg_entry)
    model_mod_name, model_cls_name = _split_entry(model_entry)

    cfg_file = _find_asvd_module_file(cfg_mod_name, search_roots)
    model_file = _find_asvd_module_file(model_mod_name, search_roots)
    if cfg_file is None or model_file is None:
        raise FileNotFoundError(
            "Could not locate ASVD remote-code python files for fallback loader.\n"
            f"  config module: {cfg_mod_name} -> {cfg_file}\n"
            f"  model module:  {model_mod_name} -> {model_file}"
        )

    cfg_mod = _import_module_from_file(cfg_mod_name, cfg_file)
    model_mod = _import_module_from_file(model_mod_name, model_file)
    cfg_cls = getattr(cfg_mod, cfg_cls_name)
    model_cls = getattr(model_mod, model_cls_name)

    config = cfg_cls.from_pretrained(local_path)
    model_kwargs = dict(config=config, low_cpu_mem_usage=True)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = model_cls.from_pretrained(local_path, **model_kwargs)
    return model, tokenizer, local_path


def _load_hf_or_asvd_model_and_tokenizer(
    model_id_or_path,
    *,
    tokenizer_id_or_path=None,
    hf_token=None,
    revision=None,
    cache_dir=None,
    trust_remote_code=True,
    dtype=None,
):
    should_try_default_loader = (
        tokenizer_id_or_path is None
        and revision is None
        and cache_dir is None
    )
    if should_try_default_loader:
        try:
            model, tokenizer = get_model_from_huggingface(model_id_or_path, hf_token=hf_token)
            return model, tokenizer, None
        except Exception as e:
            print(f"[WARN] Default HF loader failed for {model_id_or_path}: {type(e).__name__}: {e}")
            print("[WARN] Falling back to ASVD-compatible loader...")

    return _load_asvd_hf_model_and_tokenizer(
        model_id_or_path,
        tokenizer_id_or_path=tokenizer_id_or_path,
        hf_token=hf_token,
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )



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
        base = (
            getattr(args, "dobi_model", None)
            or getattr(args, "model", None)
            or getattr(args, "checkpoint", None)
            or "model"
        )
        run_name = f"{_safe_tag(base)}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return os.path.join(out_dir, f"{run_name}_{suffix}.json")


def _write_json(path: str, payload, *, compact: bool = False) -> None:
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    dump_kwargs = {
        "ensure_ascii": False,
        "indent": None if compact else 2,
    }
    if compact:
        dump_kwargs["separators"] = (",", ":")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, **dump_kwargs)
    print(f"[Output] Wrote JSON -> {path}")



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





def _normalize_tokenizer_hint(tokenizer_hint: Optional[str]) -> Optional[str]:
    if tokenizer_hint is None:
        return None
    try:
        hint = str(tokenizer_hint).strip()
    except Exception:
        return None
    if not hint:
        return None
    if os.path.isfile(hint):
        hint = os.path.dirname(hint)
    return hint


def _resolve_hflm_tokenizer_arg(model, tokenizer, tokenizer_override: Optional[str] = None):
    override = _normalize_tokenizer_hint(os.getenv("SVDLLM_TOKENIZER_MODEL", "").strip()) or _normalize_tokenizer_hint(tokenizer_override)
    if override:
        return override

    try:
        import transformers
        if isinstance(tokenizer, (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast)):
            return tokenizer
    except Exception:
        pass

    try:
        hint = _normalize_tokenizer_hint(getattr(getattr(model, "config", None), "_name_or_path", None) or getattr(model, "name_or_path", None))
    except Exception:
        hint = None
    return hint

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
    try:
        model.config._name_or_path = local_path
    except Exception:
        pass
    try:
        model.name_or_path = local_path
    except Exception:
        pass
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


def _drop_keys_recursive(obj, drop_keys: set):
    if isinstance(obj, dict):
        return {k: _drop_keys_recursive(v, drop_keys) for k, v in obj.items() if k not in drop_keys}
    if isinstance(obj, list):
        return [_drop_keys_recursive(v, drop_keys) for v in obj]
    if isinstance(obj, tuple):
        return [_drop_keys_recursive(v, drop_keys) for v in obj]
    return obj


def _remove_lmeval_samples(obj):
    return _drop_keys_recursive(obj, drop_keys={"samples"})


def _shrink_lmeval_output(obj) -> dict:
    if not isinstance(obj, dict):
        return {"value": str(obj)}

    obj = _remove_lmeval_samples(obj)

    out = {"results": obj.get("results", {})}
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


def _simple_evaluate_with_optional_log_samples(evaluator, **kwargs):
    try:
        sig = inspect.signature(evaluator.simple_evaluate)
        params = sig.parameters
        want = bool(kwargs.pop("_log_samples", False))

        if "log_samples" in params:
            kwargs["log_samples"] = want
        elif "write_out" in params:
            kwargs["write_out"] = want
    except Exception:
        kwargs.pop("_log_samples", None)

    return evaluator.simple_evaluate(**kwargs)


def _safe_simple_evaluate(
    evaluator,
    model,
    tasks: List[str],
    num_fewshot: int,
    batch_size: int,
    max_batch_size: int,
    device: str,
    limit: Optional[int],
    log_samples: bool,
):
    try:
        res = _simple_evaluate_with_optional_log_samples(
            evaluator,
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
            limit=limit,
            _log_samples=log_samples,
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
                res = _simple_evaluate_with_optional_log_samples(
                    evaluator,
                    model=model,
                    tasks=[task],
                    num_fewshot=num_fewshot,
                    batch_size=batch_size,
                    max_batch_size=max_batch_size,
                    device=device,
                    limit=limit,
                    _log_samples=log_samples,
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
        description="Evaluate linguistic tasks (lm-eval) for local HF checkpoints or Dobi-SVD checkpoints."
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
        "--saes_model",
        type=str,
        default=None,
        help="SAES-SVD checkpoint (local dir or HF repo id).",
    )
    p.add_argument(
        "--saes_base_model",
        type=str,
        default=None,
        help="Base HF model id/path to apply SAES-SVD onto (required when using SAES-SVD).",
    )
    p.add_argument("--saes_revision", type=str, default=None)
    p.add_argument("--saes_cache_dir", type=str, default=None)

    p.add_argument(
        "--dfsvd_model",
        type=str,
        default=None,
        help="DF-SVD checkpoint (local dir or HF repo id).",
    )
    p.add_argument("--dfsvd_base_model", type=str, default=None, help="Base HF model id/path to apply DF-SVD onto (required when using DF-SVD).")
    p.add_argument("--dfsvd_revision", type=str, default=None)
    p.add_argument("--dfsvd_cache_dir", type=str, default=None)

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

    p.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer id/path for HuggingFace/ASVD model loading.")
    p.add_argument("--revision", type=str, default=None, help="Optional HF revision for --model when loading from HuggingFace.")
    p.add_argument("--cache_dir", type=str, default=None, help="Optional HF cache dir for --model when loading from HuggingFace.")
    p.add_argument("--trust_remote_code", action="store_true", help="Enable transformers trust_remote_code for HuggingFace/ASVD loading.")
    p.add_argument("--no_trust_remote_code", action="store_true", help="Disable transformers trust_remote_code for HuggingFace/ASVD loading.")

    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Write results JSON to this path. If omitted, can be auto-generated with --output_dir.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write auto-named JSON (<run_name>_lm_eval.json). Ignored if --output_json is set.",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name prefix for auto JSON naming when using --output_dir.",
    )
    p.add_argument("--output_md", type=str, default=None)

    # Size controls (default: small JSON)
    p.add_argument(
        "--log_samples",
        action="store_true",
        help="Ask lm-eval to collect per-sample logs (can be very large).",
    )
    p.add_argument(
        "--json_full",
        action="store_true",
        help="Include full lm-eval output per task set in JSON (still prunes samples unless --json_keep_samples).",
    )
    p.add_argument(
        "--json_keep_samples",
        action="store_true",
        help="Keep per-sample logs in JSON (WARNING: huge). Implies --log_samples.",
    )
    p.add_argument(
        "--json_compact",
        action="store_true",
        help="Write compact (minified) JSON (no indentation) to further reduce file size.",
    )

    args = p.parse_args()

    trust_remote_code = True
    if args.no_trust_remote_code:
        trust_remote_code = False
    if args.trust_remote_code:
        trust_remote_code = True

    if args.json_keep_samples:
        args.log_samples = True


    overall_t0 = time.perf_counter()
    overall_cpu_rss0 = _get_cpu_rss_bytes()
    overall_cpu_maxrss0 = _get_cpu_maxrss_bytes()
    overall_gpu0 = _cuda_mem_snapshot(args.device)


    perf = {"phases": {"task_sets": {}}}
    with PerfRecorder(args.device, label="model_load_and_to_device") as _perf_load:
        if args.dobi_model:
            if args.dobi_remapping and args.dobi_unremapping:
                raise ValueError("Only one of --dobi_remapping / --dobi_unremapping can be set.")
        else:
            if not args.model and not args.checkpoint and not args.saes_model and not args.dfsvd_model:
                raise ValueError("Please provide --model, --checkpoint, --saes_model, --dfsvd_model or --dobi_model.")
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
            if args.dfsvd_model:
                if not args.dfsvd_base_model:
                    raise ValueError("--dfsvd_model requires --dfsvd_base_model.")
                model, tokenizer, _ = load_dfsvd_model(
                    args.dfsvd_model,
                    base_model=args.dfsvd_base_model,
                    hf_token=args.hf_token,
                    revision=args.dfsvd_revision,
                    cache_dir=args.dfsvd_cache_dir,
                )
                model_name = args.dfsvd_model
            elif args.saes_model:
                if not args.saes_base_model:
                    raise ValueError("--saes_model requires --saes_base_model.")
                model, tokenizer, _ = load_saes_svd_model(
                    args.saes_model,
                    base_model=args.saes_base_model,
                    hf_token=args.hf_token,
                    revision=args.saes_revision,
                    cache_dir=args.saes_cache_dir,
                )
                model_name = args.saes_model
            elif args.model:
                # Allow --model to be a local SAES-SVD directory too (auto-detect)
                if os.path.isdir(args.model) and looks_like_saes_svd_checkpoint(args.model):
                    if not args.saes_base_model:
                        raise ValueError("SAES-SVD checkpoint detected but --saes_base_model is missing.")
                    model, tokenizer, _ = load_saes_svd_model(
                        args.model,
                        base_model=args.saes_base_model,
                        hf_token=args.hf_token,
                        revision=args.saes_revision,
                        cache_dir=args.saes_cache_dir,
                    )
                # Allow --model to be a local DF-SVD directory too (auto-detect)
                elif os.path.isdir(args.model) and looks_like_dfsvd_checkpoint(args.model):
                    if not args.dfsvd_base_model:
                        raise ValueError("DF-SVD checkpoint detected but --dfsvd_base_model is missing.")
                    model, tokenizer, _ = load_dfsvd_model(
                        args.model,
                        base_model=args.dfsvd_base_model,
                        hf_token=args.hf_token,
                        revision=args.dfsvd_revision,
                        cache_dir=args.dfsvd_cache_dir,
                    )
                else:
                    model, tokenizer, _ = _load_hf_or_asvd_model_and_tokenizer(
                        args.model,
                        tokenizer_id_or_path=args.tokenizer,
                        hf_token=args.hf_token,
                        revision=args.revision,
                        cache_dir=args.cache_dir,
                        trust_remote_code=trust_remote_code,
                        dtype=args.dtype,
                    )
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

    perf["phases"]["model_load"] = _perf_load.to_dict()

    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except Exception as e:
        raise RuntimeError(f"lm-eval harness is required: {e}")

    tasks = _parse_tasks(args.tasks)
    lm_tokenizer = _resolve_hflm_tokenizer_arg(model, tokenizer, tokenizer_override=args.tokenizer)
    lm = HFLM(
        pretrained=model,
        tokenizer=lm_tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_size=64,
        trust_remote_code=trust_remote_code,
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

        with PerfRecorder(args.device, label=f"lm_eval[{set_name}]") as _perf_set:
            res, used = _safe_simple_evaluate(
                evaluator=evaluator,
                model=lm,
                tasks=set_tasks,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                max_batch_size=64,
                device=args.device,
                limit=args.limit,
                log_samples=args.log_samples,
            )
        perf["phases"]["task_sets"][set_name] = _perf_set.to_dict()
        if not res or not res.get("results"):
            print(f"[LM-Eval] No results for set '{set_name}' after filtering; skipping.")
            continue

        tasks_used[set_name] = list(used) if used else list(set_tasks)
        print(res.get("results", res))

        if args.json_keep_samples:
            res_out = res
        elif args.json_full:
            res_out = _remove_lmeval_samples(res)
        else:
            res_out = _shrink_lmeval_output(res)

        all_results["task_sets"][set_name] = res_out

    out_json = _auto_output_json(args, "lm_eval")

    # Redact secrets from args before serializing.
    args_dict = dict(vars(args))
    if args_dict.get("hf_token"):
        args_dict["hf_token"] = "<REDACTED>"


    overall_t1 = time.perf_counter()
    overall_cpu_rss1 = _get_cpu_rss_bytes()
    overall_cpu_maxrss1 = _get_cpu_maxrss_bytes()
    overall_gpu1 = _cuda_mem_snapshot(args.device)

    perf["overall"] = {
        "wall_time_sec": float(overall_t1 - overall_t0),
        "cpu_rss_start_bytes": int(overall_cpu_rss0) if overall_cpu_rss0 is not None else None,
        "cpu_rss_end_bytes": int(overall_cpu_rss1) if overall_cpu_rss1 is not None else None,
        "cpu_rss_delta_bytes": (int(overall_cpu_rss1 - overall_cpu_rss0) if (overall_cpu_rss0 is not None and overall_cpu_rss1 is not None) else None),
        "cpu_maxrss_start_bytes": int(overall_cpu_maxrss0) if overall_cpu_maxrss0 is not None else None,
        "cpu_maxrss_end_bytes": int(overall_cpu_maxrss1) if overall_cpu_maxrss1 is not None else None,
        "cpu_maxrss_delta_bytes": (int(overall_cpu_maxrss1 - overall_cpu_maxrss0) if (overall_cpu_maxrss0 is not None and overall_cpu_maxrss1 is not None) else None),
        "gpu_start": overall_gpu0,
        "gpu_end": overall_gpu1,
    }
    payload = {
        "schema": "svdllm_eval_v1",
        "script": os.path.basename(__file__),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join(sys.argv),
        "mode": "lm_eval",
        "args": args_dict,
        "perf": perf,
        "model": model_name,
        "tasks_used": tasks_used,
        "results": all_results,
    }

    _write_json(out_json, payload, compact=args.json_compact)

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
