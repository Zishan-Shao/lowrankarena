import argparse
import ast
import contextlib
import datetime as _dt
import io
import json
import math
import os
import re
import sys
from typing import List, Optional

import torch
from tqdm import tqdm

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



# ----------------------------------------------------------------------------
# JSON output helpers (mirrors eval_SVDLLM_benchmark.py)
# ----------------------------------------------------------------------------


def _jsonify(obj):
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
    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _safe_tag(s: str) -> str:
    s = os.path.basename(str(s)).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_") or 'run'


def _auto_output_json(args, suffix: str):
    # Priority:
    #   1) --output_json
    #   2) --output_dir + (<run_name>_<suffix>.json)
    #   3) None
    out_json = getattr(args, 'output_json', None)
    if out_json:
        return out_json
    out_dir = getattr(args, 'output_dir', None)
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    run_name = getattr(args, 'run_name', None)
    if not run_name:
        base = None
        if getattr(args, 'compare_dobi', False):
            base = f"compare_{_safe_tag(getattr(args,'checkpoint', 'ours'))}_vs_{_safe_tag(getattr(args,'dobi_model','dobi'))}"
        base = base or getattr(args, 'checkpoint', None) or getattr(args, 'dobi_model', None) or 'model'
        run_name = f"{_safe_tag(base)}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return os.path.join(out_dir, f"{run_name}_{suffix}.json")


def _write_json(path: str, payload):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_jsonify(payload), f, indent=2)
    print(f"[Output] Wrote JSON -> {path}")


class _Tee:
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
    out = []
    level = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if level == 0:
                start = i
            level += 1
        elif ch == '}':
            if level > 0:
                level -= 1
                if level == 0 and start is not None:
                    out.append(text[start : i + 1])
                    start = None
    return out


def _parse_best_dict(text: str, want_keys=None):
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
    buf = io.StringIO()
    tee = _Tee(sys.stdout, buf)
    with contextlib.redirect_stdout(tee):
        ret = fn(**kwargs)
    if isinstance(ret, dict):
        return ret
    return _parse_best_dict(buf.getvalue(), want_keys=want_keys)


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

# Ensure repo root is on PYTHONPATH
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from utils.model_utils import get_model_from_local, get_model_from_huggingface
from utils.saes_svd_loader import looks_like_saes_svd_checkpoint, load_saes_svd_model
from utils.df_svd_loader import looks_like_dfsvd_checkpoint, load_dfsvd_model
from evaluater import ppl_eval


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
        "--model",
        type=str,
        default=None,
        help="Alias for --checkpoint. Useful for local/HF ASVD model folders or repo ids.",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer id/path when evaluating a HuggingFace/ASVD model.",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision for --model/--checkpoint when loading from HuggingFace.",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional HF cache dir for --model/--checkpoint when loading from HuggingFace.",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable transformers trust_remote_code for HuggingFace/ASVD loading.",
    )
    p.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Disable transformers trust_remote_code for HuggingFace/ASVD loading.",
    )

    p.add_argument(
        "--saes_model",
        type=str,
        default=None,
        help="SAES-SVD checkpoint (local dir or HF repo id). If set, overrides --checkpoint for the 'ours' model.",
    )
    p.add_argument(
        "--saes_base_model",
        type=str,
        default=None,
        help="Base HF model id/path to apply SAES-SVD onto (required when using SAES-SVD).",
    )
    p.add_argument(
        "--saes_revision",
        type=str,
        default=None,
        help="Optional HF revision for SAES-SVD checkpoint.",
    )
    p.add_argument(
        "--saes_cache_dir",
        type=str,
        default=None,
        help="Optional HF cache dir for SAES-SVD checkpoint.",
    )

    p.add_argument(
        "--dfsvd_model",
        type=str,
        default=None,
        help="DF-SVD checkpoint (local dir or HF repo id). If set, overrides --checkpoint for the 'ours' model.",
    )
    p.add_argument(
        "--dfsvd_base_model",
        type=str,
        default=None,
        help="Base HF model id/path to apply DF-SVD onto (required when using DF-SVD).",
    )
    p.add_argument(
        "--dfsvd_revision",
        type=str,
        default=None,
        help="Optional HF revision for DF-SVD checkpoint.",
    )
    p.add_argument(
        "--dfsvd_cache_dir",
        type=str,
        default=None,
        help="Optional HF cache dir for DF-SVD checkpoint.",
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
    p.add_argument("--output_json", type=str, default=None,
                   help="Write evaluation results to this JSON file (optional).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to write JSON results (auto-named). If unset, no JSON is written unless --output_json is provided.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional run name used when auto-naming JSON under --output_dir.")

    args = p.parse_args()

    if args.model and not args.checkpoint:
        args.checkpoint = args.model
    elif args.model and args.checkpoint and args.model != args.checkpoint:
        raise ValueError("Please pass only one of --checkpoint or --model.")

    trust_remote_code = True
    if args.no_trust_remote_code:
        trust_remote_code = False
    if args.trust_remote_code:
        trust_remote_code = True

    overall_t0 = time.perf_counter()
    overall_cpu_rss0 = _get_cpu_rss_bytes()
    overall_cpu_maxrss0 = _get_cpu_maxrss_bytes()
    overall_gpu0 = _cuda_mem_snapshot(args.device)


    runs = []  # collected results for JSON output


    def _load_ours(ckpt_or_dir: str):
        # 1) Repo checkpoint (.pt) saved by this repo
        if ckpt_or_dir.endswith(".pt") and os.path.isfile(ckpt_or_dir):
            return get_model_from_local(ckpt_or_dir)

        # 2) Local SAES-SVD dir (contains saes_manifest.json + saes_state.pt)
        if os.path.isdir(ckpt_or_dir) and looks_like_saes_svd_checkpoint(ckpt_or_dir):
            if not args.saes_base_model:
                raise ValueError("SAES-SVD checkpoint detected but --saes_base_model is missing.")
            model, tokenizer, _ = load_saes_svd_model(
                ckpt_or_dir,
                base_model=args.saes_base_model,
                hf_token=args.hf_token,
                revision=args.saes_revision,
                cache_dir=args.saes_cache_dir,
            )
            return model, tokenizer

        # 3) Otherwise treat it as HF model id OR local HF directory (including ASVD repos)
        model, tokenizer, _ = _load_hf_or_asvd_model_and_tokenizer(
            ckpt_or_dir,
            tokenizer_id_or_path=args.tokenizer,
            hf_token=args.hf_token,
            revision=args.revision,
            cache_dir=args.cache_dir,
            trust_remote_code=trust_remote_code,
            dtype=args.dtype,
        )
        return model, tokenizer


    if args.dobi_model:
        if args.dobi_remapping and args.dobi_unremapping:
            raise ValueError("Only one of --dobi_remapping / --dobi_unremapping can be set.")
    if not args.checkpoint and not args.dobi_model and not args.saes_model and not args.dfsvd_model:
        raise ValueError("Please provide --checkpoint/--model, --saes_model, --dfsvd_model, or --dobi_model.")
    # Only enforce existence for local repo checkpoints; HF ids may not exist on disk.
    if args.checkpoint and args.checkpoint.endswith(".pt") and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    # If a local SAES checkpoint dir is provided via --checkpoint, require base model.
    if args.checkpoint and os.path.isdir(args.checkpoint) and looks_like_saes_svd_checkpoint(args.checkpoint):
        if not args.saes_base_model:
            raise ValueError("SAES-SVD checkpoint detected but --saes_base_model is missing.")
    # If a local DF-SVD checkpoint dir is provided via --checkpoint, require base model.
    if args.checkpoint and os.path.isdir(args.checkpoint) and looks_like_dfsvd_checkpoint(args.checkpoint):
        if not args.dfsvd_base_model:
            raise ValueError("DF-SVD checkpoint detected but --dfsvd_base_model is missing.")
    if args.saes_model and not args.saes_base_model:
        raise ValueError("--saes_model requires --saes_base_model.")
    if args.dfsvd_model and not args.dfsvd_base_model:
        raise ValueError("--dfsvd_model requires --dfsvd_base_model.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    def _run_one(model, tokenizer, label: str, datasets_override: Optional[List[str]] = None):
        run_perf = {"metrics": {}}
        with PerfRecorder(args.device, label=f"{label}:to_device") as _perf_td:
            model = _to_device(model, args.device, args.dtype)
            model.eval()
        run_perf["to_device"] = _perf_td.to_dict()

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

        run_res = {
            "model": label,
            "label": label,
            "datasets": list(datasets),
            "metrics_requested": list(metrics),
            "ppl_method": args.ppl_method,
            "seqlen": args.seqlen,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "dtype": args.dtype,
            "device": args.device,
            "perf": run_perf,
        }
        if "token" in metrics:
            with PerfRecorder(args.device, label=f"{label}:token_ppl") as _perf_tok:
                if args.ppl_method == "legacy":
                    run_res["token_ppl"] = _legacy_ppl_eval(
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
                    run_res["token_ppl"] = _call_and_capture_dict(
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

            run_perf["metrics"]["token_ppl"] = _perf_tok.to_dict()
        if not any(m in metrics for m in ("word", "byte", "bpb")):
            runs.append(run_res)
            return run_res

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
                runs.append(run_res)
                return run_res
            with PerfRecorder(args.device, label=f"{label}:lm_eval_metrics") as _perf_lm:
                res = evaluator.simple_evaluate(
                    model=lm,
                    tasks=lm_eval_datasets,
                    num_fewshot=0,
                    batch_size=args.batch_size,
                    max_batch_size=64,
                    device=args.device,
                    limit=args.max_batches,
                )
            run_perf["metrics"]["lm_eval"] = _perf_lm.to_dict()
            if res is None:
                raise RuntimeError("LM Evaluation Harness returned no results (not rank 0).")
            print("\nLM-Eval metrics (word/byte/bpb):")
            print(res.get("results", res))
            run_res["lm_eval_tasks"] = list(lm_eval_datasets)
            run_res["lm_eval_results"] = res.get("results", res)
            runs.append(run_res)
            return run_res

    dataset_sets = _parse_dataset_sets(args.dataset_sets) if args.dataset_sets else None

    def _run_with_sets(model, tokenizer, label_prefix: str):
        if dataset_sets:
            for set_name, ds in dataset_sets:
                _run_one(model, tokenizer, label=f"{label_prefix} [{set_name}]", datasets_override=ds)
        else:
            _run_one(model, tokenizer, label=label_prefix)


    if args.compare_dobi:
        if not (args.checkpoint or args.saes_model or args.dfsvd_model) or not args.dobi_model:
            raise ValueError("--compare_dobi requires (--checkpoint or --saes_model or --dfsvd_model) and --dobi_model.")
        print("[Compare] Evaluating our checkpoint...")
        if args.dfsvd_model:
            model, tokenizer, _ = load_dfsvd_model(
                args.dfsvd_model,
                base_model=args.dfsvd_base_model,
                hf_token=args.hf_token,
                revision=args.dfsvd_revision,
                cache_dir=args.dfsvd_cache_dir,
            )
        elif args.saes_model:
            model, tokenizer, _ = load_saes_svd_model(
                args.saes_model,
                base_model=args.saes_base_model,
                hf_token=args.hf_token,
                revision=args.saes_revision,
                cache_dir=args.saes_cache_dir,
            )
        else:
            model, tokenizer = _load_ours(args.checkpoint)
        _run_with_sets(model, tokenizer, label_prefix=f"{args.label} (ours)")
        print("[Compare] Evaluating Dobi checkpoint...")
        remap_flag = True if args.dobi_remapping else (False if args.dobi_unremapping else None)
        model, tokenizer, _ = _load_dobi_model(
            args.dobi_model,
            hf_token=args.hf_token,
            revision=args.dobi_revision,
            cache_dir=args.dobi_cache_dir,
            remapping=remap_flag,
        )
        _run_with_sets(model, tokenizer, label_prefix=f"{args.label} (dobi)")

        overall_t1 = time.perf_counter()
        overall_cpu_rss1 = _get_cpu_rss_bytes()
        overall_cpu_maxrss1 = _get_cpu_maxrss_bytes()
        overall_gpu1 = _cuda_mem_snapshot(args.device)

        perf_overall = {
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

        args_dict = dict(vars(args))
        if args_dict.get("hf_token"):
            args_dict["hf_token"] = "<REDACTED>"

        out_json = _auto_output_json(args, "general_ppl")
        payload = {
            "schema": "svdllm_eval_v1",
            "script": os.path.basename(__file__),
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "cmd": " ".join(sys.argv),
            "mode": "general_ppl_compare",
            "args": args_dict,
            "perf": {"overall": perf_overall},
            "runs": runs,
        }
        _write_json(out_json, payload)
        return

    if args.dobi_model:
        remap_flag = True if args.dobi_remapping else (False if args.dobi_unremapping else None)
        model, tokenizer, _ = _load_dobi_model(
            args.dobi_model,
            hf_token=args.hf_token,
            revision=args.dobi_revision,
            cache_dir=args.dobi_cache_dir,
            remapping=remap_flag,
        )
        _run_with_sets(model, tokenizer, label_prefix=args.label)
    elif args.dfsvd_model:
        model, tokenizer, _ = load_dfsvd_model(
            args.dfsvd_model,
            base_model=args.dfsvd_base_model,
            hf_token=args.hf_token,
            revision=args.dfsvd_revision,
            cache_dir=args.dfsvd_cache_dir,
        )
        _run_with_sets(model, tokenizer, label_prefix=args.label)
    elif args.saes_model:
        model, tokenizer, _ = load_saes_svd_model(
            args.saes_model,
            base_model=args.saes_base_model,
            hf_token=args.hf_token,
            revision=args.saes_revision,
            cache_dir=args.saes_cache_dir,
        )
        _run_with_sets(model, tokenizer, label_prefix=args.label)
    else:
        model, tokenizer = _load_ours(args.checkpoint)
        _run_with_sets(model, tokenizer, label_prefix=args.label)

    overall_t1 = time.perf_counter()
    overall_cpu_rss1 = _get_cpu_rss_bytes()
    overall_cpu_maxrss1 = _get_cpu_maxrss_bytes()
    overall_gpu1 = _cuda_mem_snapshot(args.device)

    perf_overall = {
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

    args_dict = dict(vars(args))
    if args_dict.get("hf_token"):
        args_dict["hf_token"] = "<REDACTED>"

    out_json = _auto_output_json(args, "general_ppl")
    payload = {
        "schema": "svdllm_eval_v1",
        "script": os.path.basename(__file__),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join(sys.argv),
        "mode": "general_ppl",
        "args": args_dict,
        "perf": {"overall": perf_overall},
        "runs": runs,
    }
    _write_json(out_json, payload)


if __name__ == "__main__":
    main()
