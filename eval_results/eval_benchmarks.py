import argparse
import ast
import contextlib
import datetime as _dt
import io
import json
import os
import re
import sys
from typing import List, Dict, Any, Tuple, Optional
from utils.df_svd_loader import looks_like_dfsvd_checkpoint, load_dfsvd_model
import torch
from torch.nn import functional as F


def load_dataset(*args, **kwargs):
    repo_root_guess = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    removed = []
    for idx in range(len(sys.path) - 1, -1, -1):
        raw = sys.path[idx]
        abs_path = os.path.abspath(raw or os.getcwd())
        if abs_path == repo_root_guess:
            removed.append((idx, raw))
            sys.path.pop(idx)
    try:
        from datasets import load_dataset as _hf_load_dataset
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Hugging Face `datasets` is required for benchmark evaluation."
        ) from e
    finally:
        for idx, raw in sorted(removed, key=lambda x: x[0]):
            sys.path.insert(idx, raw)
    return _hf_load_dataset(*args, **kwargs)


try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None
# Try to import local dataset loaders without conflicting with the HF `datasets` package name
load_piqa_local = None
load_mathqa_local = None
try:
    # This may resolve to the HF package; handle failure below
    from datasets.load_data import load_piqa_local as _lp_local, load_mathqa_local as _lm_local  # type: ignore
    load_piqa_local = _lp_local
    load_mathqa_local = _lm_local
except Exception:
    try:
        import importlib.util as _ilu
        _here = os.path.dirname(os.path.abspath(__file__))
        _base = os.path.join(_here, 'datasets', 'load_data.py')
        if os.path.isfile(_base):
            _spec = _ilu.spec_from_file_location('local_datasets_load_data', _base)
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore
                load_piqa_local = getattr(_mod, 'load_piqa_local', None)
                load_mathqa_local = getattr(_mod, 'load_mathqa_local', None)
    except Exception:
        load_piqa_local = None
        load_mathqa_local = None
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
    # Priority:
    #   1) --output_json
    #   2) --output_dir + (<run_name>_<suffix>.json)
    #   3) None
    out_json = getattr(args, "output_json", None)
    if out_json:
        return out_json
    out_dir = getattr(args, "output_dir", None)
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    run_name = getattr(args, "run_name", None)
    if not run_name:
        base = getattr(args, "model", None) or getattr(args, "dobi_model", None) or "model"
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


# Ensure repo root on PYTHONPATH so imports work regardless of launch cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.model_utils import get_model_from_huggingface, get_model_from_local
from utils.saes_svd_loader import looks_like_saes_svd_checkpoint, load_saes_svd_model
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


'''
A. 纯 lm-eval (最可比)
python eval_benchmarks.py \
  --model checkpoints/llama2_act_lora_expressivity_mix_0.4_mcq.pt \
  --device cuda \
  --batch_size 1 \
  --use_lm_eval \
  --lm_eval_tasks arc_easy,arc_challenge,piqa,winogrande \
  --lm_eval_num_fewshot 0 \
  --dtype bfloat16

B. 本地评测但强制一致 padding
python eval_benchmarks.py \
  --model checkpoints/llama2_act_lora_expressivity_mix_0.4_mcq.pt \
  --device cuda \
  --batch_size 1 \
  --dtype bfloat16 \
  --force_right_padding

CUDA_VISIBLE_DEVICES=3 python eval_benchmarks.py   --model ./checkpoints/jeffwan_llama_7b_hf_act_lora_lmwhiten_mixedlora_0.4_linguistic.pt   --device cuda   --batch_size 16   --use_lm_eval   --lm_eval_tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa,truthfulqa_mc1   --lm_eval_num_fewshot 0   --dtype bfloat16   --force_right_padding   --fix_pad_query_mask

CUDA_VISIBLE_DEVICES=3 python expressivity/svd_act_lora_joint_QK.py   --model jeffwan/llama-7b-hf   --keep_ratio 0.4   --whitened_cache ./checkpoints/jeffwan_llama_7b_hf_whitening_only_0.4_jointqk.pt   --trust_whitened_cache   --lora_nsamples 4096 --seqlen 2048 --train_batch_size 8   --epochs 1 --lr 5e-4 --lora_rank 16 --lora_alpha 32   --mix_buckets   --bucket_props LM:0.4,INST:0.4,MATH:0.2   --bucket_lm_datasets wikitext2,ptb,c4   --bucket_inst_datasets hellaswag,piqa,winogrande_xl,ai2_arc_easy,ai2_arc_challenge,openbookqa   --bucket_math_datasets aime   --save_path ./checkpoints/jeffwan_ll
ama_7b_hf_act_lora_lmwhiten_mixedlora_0.4_jointqk.pt

CUDA_VISIBLE_DEVICES=3 python eval_benchmarks.py \
  --model ./checkpoints/llama-2-7b-hf_act_lora_lmwhiten_mixedlora_0.4_linguistic_enhanced.pt \
  --device cuda --batch_size 16 \
  --dtype bfloat16 \
  --force_right_padding --fix_pad_query_mask \
  --skip_truthfulqa --skip_gsm8k


# evaluate dobi:
CUDA_VISIBLE_DEVICES=3 python eval_benchmarks.py   --model Qinsi1/DobiSVD-Llama-2-7b-hf-0.4   --device cuda --batch_size 8 --use_lm_eval   --lm_eval_tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa


'''


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Create a pad token if absolutely necessary
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    # For decoder-only models, use left-padding for generation
    try:
        tokenizer.padding_side = 'left'
    except Exception:
        pass
    # Provide a reasonable max length hint to silence truncation warnings
    try:
        if getattr(tokenizer, 'model_max_length', None) in (None, int(1e30)):
            tokenizer.model_max_length = 4096
    except Exception:
        pass


def _normalize_tokenizer_hint(model_hint: Optional[str]) -> Optional[str]:
    if model_hint is None:
        return None
    try:
        hint = str(model_hint).strip()
    except Exception:
        return None
    if not hint:
        return None
    if os.path.isfile(hint):
        hint = os.path.dirname(hint)
    return hint


def _tokenizer_ok(tokenizer) -> bool:
    try:
        return tokenizer is not None and not isinstance(tokenizer, bool) and callable(tokenizer)
    except Exception:
        return False


def _load_tokenizer_from_hint(model_hint: str, hf_token: Optional[str] = None):
    model_hint = _normalize_tokenizer_hint(model_hint)
    if not model_hint:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_hint, trust_remote_code=True, use_fast=True, token=hf_token
        )
        if _tokenizer_ok(tok):
            return tok
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_hint, trust_remote_code=True, use_fast=False, token=hf_token
        )
        if _tokenizer_ok(tok):
            return tok
    except Exception:
        pass
    try:
        from transformers import LlamaTokenizerFast, LlamaTokenizer
        for cls in (LlamaTokenizerFast, LlamaTokenizer):
            try:
                kwargs = {"token": hf_token}
                if cls.__name__ == "LlamaTokenizer":
                    kwargs["legacy"] = True
                tok = cls.from_pretrained(model_hint, **kwargs)
                if _tokenizer_ok(tok):
                    return tok
            except Exception:
                continue
    except Exception:
        pass
    return None


def _ensure_tokenizer_compat(model, tokenizer, hf_token: Optional[str] = None, tokenizer_hint: Optional[str] = None):
    override = _normalize_tokenizer_hint(os.getenv("SVDLLM_TOKENIZER_MODEL", "").strip())
    tokenizer_hint = _normalize_tokenizer_hint(tokenizer_hint)
    model_hint = None
    try:
        model_hint = _normalize_tokenizer_hint(getattr(getattr(model, "config", None), "_name_or_path", None))
    except Exception:
        model_hint = None
    # Prefer explicit override
    if override:
        tok = _load_tokenizer_from_hint(override, hf_token=hf_token)
        if tok is not None:
            tokenizer = tok
    # Repair invalid tokenizer
    if not _tokenizer_ok(tokenizer):
        hint = override or tokenizer_hint or model_hint
        if hint:
            tok = _load_tokenizer_from_hint(hint, hf_token=hf_token)
            if tok is not None:
                tokenizer = tok
    # Validate vocab size match when possible
    try:
        model_vocab = int(model.get_input_embeddings().weight.shape[0])
    except Exception:
        model_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        try:
            model_vocab = int(model_vocab) if model_vocab is not None else None
        except Exception:
            model_vocab = None
    tok_vocab = getattr(tokenizer, "vocab_size", None)
    try:
        tok_vocab = int(tok_vocab) if tok_vocab is not None else None
    except Exception:
        tok_vocab = None
    if model_vocab and tok_vocab and model_vocab != tok_vocab:
        print(f"[Warn] Tokenizer vocab {tok_vocab} != model vocab {model_vocab}; reloading tokenizer.")
        hint = override or tokenizer_hint or model_hint
        if hint:
            tok = _load_tokenizer_from_hint(hint, hf_token=hf_token)
            if tok is not None:
                tokenizer = tok
                tok_vocab = getattr(tokenizer, "vocab_size", None)
    if not _tokenizer_ok(tokenizer):
        raise TypeError(
            "Tokenizer object is not callable and could not be reconstructed; "
            "set SVDLLM_TOKENIZER_MODEL to a valid local tokenizer."
        )
    return tokenizer


def _to_device(model, device: str, dtype_prefer: Optional[str] = None):
    if dtype_prefer is not None:
        _map = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        dt = _map.get(dtype_prefer.lower(), None)
        if dt is not None:
            model = model.to(dtype=dt)
    else:
        try:
            prefer_bf16 = str(device).startswith('cuda') and torch.cuda.is_bf16_supported()
        except Exception:
            prefer_bf16 = False
        model = model.to(dtype=(torch.bfloat16 if prefer_bf16 else torch.float16))
    return model.to(device).eval()


def _force_right_padding(tokenizer):
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})


def _set_pad_query_fix_flags(model, fix: bool):
    if not fix:
        return
    for module in model.modules():
        if module.__class__.__name__ == "SVD_LlamaAttention":
            module.fix_pad_query_mask = True


def _run_lm_eval_harness(
    model,
    tokenizer,
    device: str,
    tasks: str,
    limit: Optional[int],
    batch_size: int,
    max_batch_size: int,
    max_length: int,
    num_fewshot: int,
    include_path: Optional[str],
    add_bos_token: Optional[bool],
    prefix_token_id: Optional[int],
):
    # Lazy import to avoid requiring lm-eval if not used
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    task_manager = TaskManager(include_path=include_path) if include_path else None

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
    # Print compact results dict
    print("\nLM-Eval results:")
    print(results.get("results", results))
    return results


@torch.no_grad()
def _score_mc_batch(
    model,
    tokenizer,
    batch_samples: List[Dict[str, Any]],
    device: str,
) -> List[int]:
    """
    Score a batch of multiple-choice items by summing log-probs of answer tokens
    conditioned on the prompt. For each sample, returns predicted choice index.
    batch_samples: list of { 'prompt': str, 'choices': List[str], 'answer_idx': int }
    """
    _ensure_pad_token(tokenizer)
    # Flatten choices
    input_ids_list = []
    labels_list = []
    group_sizes = []
    for s in batch_samples:
        prompt_ids = tokenizer.encode(s['prompt'], add_special_tokens=False)
        k = 0
        for ch in s['choices']:
            # Prepend a space to align subword tokenization for many tokenizers
            ans_ids = tokenizer.encode(" " + ch, add_special_tokens=False)
            ids = prompt_ids + ans_ids
            labels = [-100] * len(prompt_ids) + ans_ids
            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            k += 1
        group_sizes.append(k)
    # Pad to max length
    max_len = max(x.size(0) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    for i, (ids, lbs) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : ids.size(0)] = ids
        labels[i, : lbs.size(0)] = lbs
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    # Forward
    out = model(input_ids=input_ids, use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
    logits = out.logits  # [N, L, V]
    logprobs = F.log_softmax(logits, dim=-1)
    # Shift for next-token prediction
    lp = logprobs[:, :-1, :]
    lbl = labels[:, 1:]
    # Gather only where labels != -100
    mask = (lbl != -100)
    lbl_clamped = torch.where(mask, lbl, torch.zeros_like(lbl))
    token_lp = lp.gather(dim=-1, index=lbl_clamped.unsqueeze(-1)).squeeze(-1)
    token_lp = torch.where(mask, token_lp, torch.zeros_like(token_lp))
    seq_scores = token_lp.sum(dim=1)  # [N]
    # Group back into per-sample predictions
    preds = []
    idx = 0
    for g in group_sizes:
        group_scores = seq_scores[idx: idx + g]
        pred = int(torch.argmax(group_scores).item())
        preds.append(pred)
        idx += g
    return preds


def _accuracy(preds: List[int], golds: List[int]) -> float:
    if not preds:
        return float('nan')
    correct = sum(int(p == g) for p, g in zip(preds, golds))
    return 100.0 * correct / len(preds)


def _eval_dataset_mc(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    device: str,
    batch_size: int,
    limit: Optional[int] = None,
    desc: str = "MC",
) -> float:
    preds, golds = [], []
    total = len(samples) if limit is None else min(len(samples), limit)
    i = 0
    # Progress bar over batches
    n_batches = (total + max(1, batch_size) - 1) // max(1, batch_size)
    pbar = tqdm(total=n_batches, desc=desc, leave=False)
    while i < total:
        batch = samples[i: i + batch_size]
        preds.extend(_score_mc_batch(model, tokenizer, batch, device))
        golds.extend([s['answer_idx'] for s in batch])
        i += batch_size
        pbar.update(1)
    pbar.close()
    return _accuracy(preds, golds)


def eval_openbookqa(model, tokenizer, device: str, batch_size: int, limit: Optional[int]):
    ds_all = load_dataset('openbookqa', 'main')
    split = 'validation' if 'validation' in ds_all else ('test' if 'test' in ds_all else 'train')
    ds = ds_all[split]
    items = []
    for ex in ds:
        stem = ex.get('question_stem') or ex.get('question') or ''
        ch_texts = ex['choices']['text'] if isinstance(ex.get('choices'), dict) else ex.get('choices', [])
        ch_labels = ex['choices']['label'] if isinstance(ex.get('choices'), dict) else None
        key = ex.get('answerKey')
        idx = None
        if key is not None and ch_labels is not None:
            try:
                idx = list(ch_labels).index(key)
            except Exception:
                idx = 0
        elif isinstance(ex.get('label'), int):
            idx = int(ex['label'])
        else:
            # fallback: assume first is correct (rare, but prevents crash)
            idx = 0
        items.append({'prompt': stem, 'choices': list(ch_texts), 'answer_idx': idx})
    return _eval_dataset_mc(model, tokenizer, items, device, batch_size, limit, desc="OpenBookQA")


def eval_arc_easy(model, tokenizer, device: str, batch_size: int, limit: Optional[int]):
    ds_all = load_dataset('ai2_arc', 'ARC-Easy')
    split = 'validation' if 'validation' in ds_all else ('test' if 'test' in ds_all else 'train')
    ds = ds_all[split]
    items = []
    for ex in ds:
        q = ex.get('question', '')
        ch = ex.get('choices')
        key = ex.get('answerKey') or ex.get('answer') or ex.get('label')
        ch_texts, ch_labels = [], []
        if isinstance(ch, dict):
            ch_texts = list(ch.get('text') or ch.get('texts') or [])
            ch_labels = list(ch.get('label') or ch.get('labels') or [])
        elif isinstance(ch, list):
            try:
                ch_texts = [c.get('text', '') for c in ch]
                ch_labels = [c.get('label', None) for c in ch]
            except Exception:
                # If list of strings
                ch_texts = list(ch)
                ch_labels = [None] * len(ch_texts)
        idx = 0
        if key is not None and ch_labels:
            try:
                idx = list(ch_labels).index(key)
            except Exception:
                idx = 0
        items.append({'prompt': q, 'choices': ch_texts, 'answer_idx': idx})
    return _eval_dataset_mc(model, tokenizer, items, device, batch_size, limit, desc="ARC-Easy")


def eval_winogrande(model, tokenizer, device: str, batch_size: int, limit: Optional[int]):
    ds_all = load_dataset('winogrande', 'winogrande_xl')
    split = 'validation' if 'validation' in ds_all else 'test'
    ds = ds_all[split]
    items = []
    for ex in ds:
        sent = ex.get('sentence', '')
        opt1 = ex.get('option1', '')
        opt2 = ex.get('option2', '')
        ans = ex.get('answer', '1')
        ans_idx = 0 if str(ans).strip() in ('1', 'A', 'a') else 1
        # Use left-context partial scoring
        if '_' in sent:
            left = sent.split('_', 1)[0]
        else:
            left = sent + ' '
        items.append({'prompt': left, 'choices': [opt1, opt2], 'answer_idx': ans_idx})
    return _eval_dataset_mc(model, tokenizer, items, device, batch_size, limit, desc="WinoGrande")


def eval_hellaswag(model, tokenizer, device: str, batch_size: int, limit: Optional[int]):
    ds_all = load_dataset('hellaswag')
    split = 'validation' if 'validation' in ds_all else 'test'
    ds = ds_all[split]
    items = []
    for ex in ds:
        ctx = ex.get('ctx', None) or (ex.get('ctx_a', '') + ' ' + ex.get('ctx_b', ''))
        endings = ex.get('endings') or ex.get('ending_options') or []
        label = ex.get('label', 0)
        items.append({'prompt': ctx.strip() + ' ', 'choices': list(endings), 'answer_idx': int(label)})
    return _eval_dataset_mc(model, tokenizer, items, device, batch_size, limit, desc="HellaSwag")


def eval_piqa(model, tokenizer, device: str, batch_size: int, limit: Optional[int]):
    # Prefer local copy if present under datasets/piqa
    if load_piqa_local is not None:
        try:
            local_items = load_piqa_local(split='validation')
            if local_items:
                if limit is not None:
                    local_items = local_items[:limit]
                return _eval_dataset_mc(model, tokenizer, local_items, device, batch_size, limit=None, desc="PIQA")
        except Exception:
            pass
    # Prefer hub-native dataset if available (datasets<3), otherwise use hub file
    ds = None
    try:
        ds_all = load_dataset('piqa')
        split = 'validation' if 'validation' in ds_all else ('test' if 'test' in ds_all else 'train')
        ds = ds_all[split]
    except Exception:
        # Fallback: download raw JSONL from the dataset hub and parse
        try:
            if hf_hub_download is None:
                raise RuntimeError('huggingface_hub not available')
            candidates = ['valid.jsonl', 'validation.jsonl', 'dev.jsonl', 'test.jsonl']
            local_path = None
            for fn in candidates:
                try:
                    local_path = hf_hub_download(repo_id='ybisk/piqa', repo_type='dataset', filename=fn)
                    break
                except Exception:
                    continue
            if local_path is None:
                raise RuntimeError('PIQA: could not download any valid split')
            import json
            with open(local_path, 'r', encoding='utf-8') as f:
                raw = [json.loads(l) for l in f if l.strip()]
            class _Wrap:
                def __iter__(self):
                    return iter(raw)
            ds = _Wrap()
        except Exception:
            print('[Eval] Skipping PIQA due to dataset load failure.')
            return float('nan')
    items = []
    for ex in ds:
        goal = ex.get('goal', '')
        sol1 = ex.get('sol1', '')
        sol2 = ex.get('sol2', '')
        label = ex.get('label', 0)
        try:
            idx = int(label)
        except Exception:
            idx = 0 if str(label).strip() in ('0', 'A', 'a') else 1
        items.append({'prompt': goal.strip() + '\nAnswer:', 'choices': [sol1, sol2], 'answer_idx': idx})
    return _eval_dataset_mc(model, tokenizer, items, device, batch_size, limit, desc="PIQA")


def _parse_mathqa_options(opt_str: str) -> Tuple[List[str], Dict[str, int]]:
    # options like "A) ... , B) ... , C) ..."
    choices, mapping = [], {}
    cur = ''
    labels = []
    buf = opt_str
    # Split on letter) occurrences
    import re
    parts = re.split(r"\s*([A-Ea-e])\)\s*", buf)
    # parts like ['', 'A', 'optA', 'B', 'optB', ...]
    for i in range(1, len(parts), 2):
        lab = parts[i].upper()
        text = parts[i + 1].strip()
        labels.append(lab)
        choices.append(text)
        mapping[lab] = len(choices) - 1
    return choices, mapping


def eval_mathqa(model, tokenizer, device: str, batch_size: int, limit: Optional[int]):
    # Prefer local copy if present under datasets/MathQA
    if load_mathqa_local is not None:
        try:
            local_items = load_mathqa_local(split='validation')
            if local_items:
                if limit is not None:
                    local_items = local_items[:limit]
                return _eval_dataset_mc(model, tokenizer, local_items, device, batch_size, limit=None, desc="MathQA")
        except Exception:
            pass
    # Try hub-native dataset; fallback to CSV from hub using hf_hub_download
    ds = None
    try:
        ds_all = load_dataset('math_qa')
        split = 'validation' if 'validation' in ds_all else ('test' if 'test' in ds_all else 'train')
        ds = ds_all[split]
    except Exception:
        # Try common file names for splits using hub download
        try:
            if hf_hub_download is None:
                raise RuntimeError('huggingface_hub not available')
            file_candidates = ['validation.csv', 'valid.csv', 'val.csv', 'dev.csv', 'test.csv']
            csv_path = None
            for fname in file_candidates:
                try:
                    csv_path = hf_hub_download(repo_id='math_qa', repo_type='dataset', filename=fname)
                    break
                except Exception:
                    continue
            if csv_path is None:
                raise RuntimeError('MathQA: could not download any split CSV')
            import csv
            rows = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            class _Wrap:
                def __iter__(self):
                    return iter(rows)
            ds = _Wrap()
        except Exception:
            print('[Eval] Skipping MathQA due to dataset load failure.')
            return float('nan')
    items = []
    for ex in ds:
        q = ex.get('Problem') or ex.get('problem') or ex.get('question', '')
        opt = ex.get('options', '')
        correct = ex.get('correct') or ex.get('label') or 'A'
        choices, map_l = _parse_mathqa_options(opt)
        idx = map_l.get(str(correct).upper(), 0)
        items.append({'prompt': q.strip() + '\nAnswer:', 'choices': choices, 'answer_idx': idx})
    return _eval_dataset_mc(model, tokenizer, items, device, batch_size, limit, desc="MathQA")


def eval_truthfulqa_mc1(model, tokenizer, device: str, batch_size: int, limit: Optional[int]):
    try:
        ds_all = load_dataset('truthful_qa', 'multiple_choice')
    except Exception:
        # Older naming variant
        ds_all = load_dataset('truthful_qa_mc')
    split = 'validation' if 'validation' in ds_all else ('test' if 'test' in ds_all else 'train')
    ds = ds_all[split]
    items = []
    for ex in ds:
        q = ex.get('question', '')
        # Attempt schema: a flat choices list with labels
        choices = ex.get('choices', None)
        label = ex.get('label', None)
        if isinstance(choices, list) and isinstance(label, int):
            idx = int(label)
            items.append({'prompt': q.strip() + '\nAnswer:', 'choices': choices, 'answer_idx': idx})
            continue
        # Fallback to mc1_targets with {choices: [...], labels: [...]}
        mc1 = ex.get('mc1_targets', None)
        if isinstance(mc1, dict) and 'choices' in mc1 and 'labels' in mc1:
            chs = list(mc1['choices'])
            labs = list(mc1['labels'])
            # pick the first label==1 as correct (MC1)
            idx = 0
            for i, lb in enumerate(labs):
                if int(lb) == 1:
                    idx = i
                    break
            items.append({'prompt': q.strip() + '\nAnswer:', 'choices': chs, 'answer_idx': idx})
    if not items:
        return float('nan')
    return _eval_dataset_mc(model, tokenizer, items, device, batch_size, limit, desc="TruthfulQA")


@torch.no_grad()
def eval_gsm8k(model, tokenizer, device: str, limit: Optional[int], max_new_tokens: int = 64, batch_size: int = 1) -> float:
    ds_all = load_dataset('gsm8k', 'main')
    split = 'test' if 'test' in ds_all else 'validation'
    ds = ds_all[split]
    # Greedy generation of short answers; parse final number
    def _gold_answer(s: str) -> str:
        import re
        m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", s)
        return m.group(1) if m else s.strip().split()[-1]

    def _pred_answer(text: str) -> str:
        import re
        # take last number-like token
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else text.strip().split()[-1]

    prompts = []
    golds = []
    for ex in ds:
        q = ex.get('question', '')
        a = ex.get('answer', '')
        golds.append(_gold_answer(a))
        prompt = (
            "Solve the following math problem. Give only the final answer as a number.\n"
            f"Question: {q}\nAnswer:"
        )
        prompts.append(prompt)
    if limit is not None:
        prompts = prompts[:limit]
        golds = golds[:limit]
    correct = 0
    # Batch generate
    i = 0
    _ensure_pad_token(tokenizer)
    total = len(prompts)
    pbar = tqdm(total=(total + batch_size - 1) // batch_size, desc="GSM8K", leave=False)
    while i < total:
        batch = prompts[i: i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = enc['input_ids'].to(device)
        attn = enc['attention_mask'].to(device)
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        outs = tokenizer.batch_decode(gen[:, input_ids.size(1):], skip_special_tokens=True)
        for j, text in enumerate(outs):
            pred = _pred_answer(text)
            if pred == golds[i + j]:
                correct += 1
        i += batch_size
        pbar.update(1)
    pbar.close()
    return 100.0 * correct / len(prompts) if prompts else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default=None, help='HF model id or path to local .pt checkpoint saved by this repo')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--hf_token', type=str, default=None)
    ap.add_argument('--tokenizer', type=str, default=None, help='Optional tokenizer id/path for HuggingFace/ASVD model loading')
    ap.add_argument('--revision', type=str, default=None, help='Optional HF revision for --model when loading from HuggingFace')
    ap.add_argument('--cache_dir', type=str, default=None, help='Optional HF cache dir for --model when loading from HuggingFace')
    ap.add_argument('--trust_remote_code', action='store_true', help='Enable transformers trust_remote_code for HuggingFace/ASVD loading')
    ap.add_argument('--no_trust_remote_code', action='store_true', help='Disable transformers trust_remote_code for HuggingFace/ASVD loading')
    ap.add_argument('--saes_svd', action='store_true',
                    help='Treat --model as SAES-SVD checkpoint (local dir or HF repo id)')
    ap.add_argument('--saes_base_model', type=str, default=None,
                    help='Base HF model id/path to apply SAES-SVD onto (required when using SAES-SVD).')
    ap.add_argument('--saes_revision', type=str, default=None)
    ap.add_argument('--saes_cache_dir', type=str, default=None)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None, help='Limit examples per task for quick runs')
    ap.add_argument('--dtype', type=str, default=None, help='float16/bfloat16/float32')
    ap.add_argument('--gsm8k_max_new_tokens', type=int, default=64)
    ap.add_argument('--skip_gsm8k', action='store_true')
    ap.add_argument('--skip_truthfulqa', action='store_true')
    ap.add_argument('--use_lm_eval', action='store_true', help='Run LM-Eval Harness tasks instead of custom benchmarks')
    ap.add_argument('--lm_eval_tasks', type=str, default='arc_easy,arc_challenge,hellaswag,piqa',
                    help='Comma-separated LM-Eval task names')
    ap.add_argument('--lm_eval_num_fewshot', type=int, default=0)
    ap.add_argument('--lm_eval_max_length', type=int, default=2048)
    ap.add_argument('--lm_eval_max_batch_size', type=int, default=64)
    ap.add_argument(
        '--lm_eval_add_bos_token',
        type=str,
        default='auto',
        choices=['auto', 'true', 'false'],
        help='Whether to add BOS for lm-eval tokenization (auto=avoid double BOS by preferring prefix token).',
    )
    ap.add_argument(
        '--lm_eval_prefix_token_id',
        type=int,
        default=None,
        help='Override lm-eval rolling prefix token id (e.g., set to BOS token id).',
    )
    ap.add_argument('--lm_eval_include_path', type=str, default=None,
                    help='Optional include_path for custom lm-eval tasks')
    ap.add_argument('--token_ppl', action='store_true', help='Run token-level PPL (eval_general_ppl style) and exit')
    ap.add_argument('--token_ppl_datasets', type=str, default='wikitext2,ptb,c4',
                    help='Comma-separated datasets for token PPL')
    ap.add_argument('--token_ppl_seqlen', type=int, default=2048)
    ap.add_argument('--token_ppl_batch_size', type=int, default=4)
    ap.add_argument('--token_ppl_max_batches', type=int, default=None)
    ap.add_argument('--force_right_padding', action='store_true', help='Force right padding (safer for FlashSVD)')
    ap.add_argument('--fix_pad_query_mask', action='store_true', help='Fix pad-query rows in FlashSVD attention')

    ap.add_argument('--output_json', type=str, default=None,
                    help='Write evaluation results to this JSON file (optional).')
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Directory to write JSON results (auto-named). If unset, no JSON is written unless --output_json is provided.')
    ap.add_argument('--run_name', type=str, default=None,
                    help='Optional run name used when auto-naming JSON under --output_dir.')
    ap.add_argument('--dfsvd_svd', action='store_true', help='Treat --model as DF-SVD checkpoint (local dir or HF repo id)')
    ap.add_argument('--dfsvd_base_model', type=str, default=None, help='Base HF model id/path to apply DF-SVD onto (required when using DF-SVD).')
    ap.add_argument('--dfsvd_revision', type=str, default=None)
    ap.add_argument('--dfsvd_cache_dir', type=str, default=None)
    ap.add_argument('--dobi_model', type=str, default=None,
                    help='Dobi-SVD checkpoint (HF repo id or local dir). If set, overrides --model for loading.')
    ap.add_argument('--dobi_revision', type=str, default=None)
    ap.add_argument('--dobi_cache_dir', type=str, default=None)
    ap.add_argument('--dobi_remapping', action='store_true', help='Force the Dobi remapping loader.')
    ap.add_argument('--dobi_unremapping', action='store_true', help='Force the Dobi unremapping loader.')
    args = ap.parse_args()

    if not args.model and not args.dobi_model:
        ap.error('Please provide --model or --dobi_model.')
    if args.dobi_remapping and args.dobi_unremapping:
        ap.error('Please choose at most one of --dobi_remapping / --dobi_unremapping.')
    model_ref = args.dobi_model or args.model

    trust_remote_code = True
    if args.no_trust_remote_code:
        trust_remote_code = False
    if args.trust_remote_code:
        trust_remote_code = True

    overall_t0 = time.perf_counter()
    overall_cpu_rss0 = _get_cpu_rss_bytes()
    overall_cpu_maxrss0 = _get_cpu_maxrss_bytes()
    overall_gpu0 = _cuda_mem_snapshot(args.device)
    perf: Dict[str, Any] = {"phases": {}, "tasks": {}}


    with PerfRecorder(args.device, label="load_model") as _perf_load:
        # Load model
        if args.dobi_model:
            remapping = None
            if args.dobi_remapping:
                remapping = True
            elif args.dobi_unremapping:
                remapping = False
            model, tokenizer, _ = _load_dobi_model(
                args.dobi_model,
                hf_token=args.hf_token,
                revision=args.dobi_revision,
                cache_dir=args.dobi_cache_dir,
                remapping=remapping,
            )
        elif args.model and os.path.exists(args.model) and args.model.endswith('.pt'):
            model, tokenizer = get_model_from_local(args.model)
        elif args.model and ((os.path.isdir(args.model) and looks_like_dfsvd_checkpoint(args.model)) or args.dfsvd_svd):
            if not args.dfsvd_base_model:
                raise ValueError("DF-SVD requires --dfsvd_base_model.")
            model, tokenizer, _ = load_dfsvd_model(
                args.model,
                base_model=args.dfsvd_base_model,
                hf_token=args.hf_token,
                revision=args.dfsvd_revision,
                cache_dir=args.dfsvd_cache_dir,
            )
        elif args.model and ((os.path.isdir(args.model) and looks_like_saes_svd_checkpoint(args.model)) or args.saes_svd):
            if not args.saes_base_model:
                raise ValueError("SAES-SVD requires --saes_base_model.")
            model, tokenizer, _ = load_saes_svd_model(
                args.model,
                base_model=args.saes_base_model,
                hf_token=args.hf_token,
                revision=args.saes_revision,
                cache_dir=args.saes_cache_dir,
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
        tokenizer = _ensure_tokenizer_compat(model, tokenizer, hf_token=args.hf_token, tokenizer_hint=args.tokenizer)
        if args.force_right_padding:
            _force_right_padding(tokenizer)
        else:
            _ensure_pad_token(tokenizer)
        model = _to_device(model, args.device, args.dtype)
        _set_pad_query_fix_flags(model, fix=args.fix_pad_query_mask)

        # Disable cache for deterministic eval and lower memory
        try:
            model.config.use_cache = False
        except Exception:
            pass

    perf["phases"]["load_model"] = _perf_load.to_dict()

    if args.use_lm_eval:
        # Resolve add_bos_token / prefix_token_id for lm-eval
        add_bos_token = None
        if args.lm_eval_add_bos_token == 'true':
            add_bos_token = True
        elif args.lm_eval_add_bos_token == 'false':
            add_bos_token = False
        # Resolve prefix token first
        prefix_token_id = args.lm_eval_prefix_token_id
        if prefix_token_id is None and tokenizer.bos_token_id is not None:
            prefix_token_id = tokenizer.bos_token_id
        # auto: avoid double BOS (prefer prefix token for rolling loglikelihood)
        if args.lm_eval_add_bos_token == 'auto':
            if tokenizer.bos_token_id is None:
                add_bos_token = None
            elif prefix_token_id == tokenizer.bos_token_id:
                add_bos_token = False
            else:
                add_bos_token = True
        with PerfRecorder(args.device, label="lm_eval") as _perf_lm_eval:
            lm_eval_res = _run_lm_eval_harness(
                model,
                tokenizer,
                device=args.device,
                tasks=args.lm_eval_tasks,
                limit=args.limit,
                batch_size=args.batch_size,
                max_batch_size=args.lm_eval_max_batch_size,
                max_length=args.lm_eval_max_length,
                num_fewshot=args.lm_eval_num_fewshot,
                include_path=args.lm_eval_include_path,
                add_bos_token=add_bos_token,
                prefix_token_id=prefix_token_id,
            )
        perf["phases"]["lm_eval"] = _perf_lm_eval.to_dict()

        out_json = _auto_output_json(args, 'lm_eval')
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

        args_dict = dict(vars(args))
        if args_dict.get("hf_token"):
            args_dict["hf_token"] = "<REDACTED>"

        payload = {
            'schema': 'svdllm_eval_v1',
            'script': os.path.basename(__file__),
            'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
            'cmd': ' '.join(sys.argv),
            'mode': 'lm_eval',
            "args": args_dict,
            "perf": perf,
            'model': model_ref,
            'results': lm_eval_res.get('results', lm_eval_res) if isinstance(lm_eval_res, dict) else lm_eval_res,
        }
        _write_json(out_json, payload)
        return
    if args.token_ppl:
        ds = [d.strip() for d in args.token_ppl_datasets.split(',') if d.strip()]
        with PerfRecorder(args.device, label="token_ppl") as _perf_token_ppl:
            token_ppl = _call_and_capture_dict(
                ppl_eval,
                want_keys=ds,
                model=model,
                tokenizer=tokenizer,
                datasets=ds,
                model_seq_len=args.token_ppl_seqlen,
                batch_size=args.token_ppl_batch_size,
                device=args.device,
                label='Token PPL',
                max_batches=args.token_ppl_max_batches,
            )
        perf["phases"]["token_ppl"] = _perf_token_ppl.to_dict()

        out_json = _auto_output_json(args, 'token_ppl')
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

        args_dict = dict(vars(args))
        if args_dict.get("hf_token"):
            args_dict["hf_token"] = "<REDACTED>"

        payload = {
            'schema': 'svdllm_eval_v1',
            'script': os.path.basename(__file__),
            'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
            'cmd': ' '.join(sys.argv),
            'mode': 'token_ppl',
            "args": args_dict,
            "perf": perf,
            'model': model_ref,
            'results': token_ppl,
        }
        _write_json(out_json, payload)
        return

    # Evaluate tasks
    results: Dict[str, float] = {}
    with PerfRecorder(args.device, label="Openb.") as _perf_task:
        results['Openb.'] = eval_openbookqa(model, tokenizer, args.device, args.batch_size, args.limit)
    perf["tasks"]["Openb."] = _perf_task.to_dict()
    with PerfRecorder(args.device, label="ARC_e") as _perf_task:
        results['ARC_e'] = eval_arc_easy(model, tokenizer, args.device, args.batch_size, args.limit)
    perf["tasks"]["ARC_e"] = _perf_task.to_dict()
    with PerfRecorder(args.device, label="WinoG.") as _perf_task:
        results['WinoG.'] = eval_winogrande(model, tokenizer, args.device, args.batch_size, args.limit)
    perf["tasks"]["WinoG."] = _perf_task.to_dict()
    with PerfRecorder(args.device, label="HellaS.") as _perf_task:
        results['HellaS.'] = eval_hellaswag(model, tokenizer, args.device, args.batch_size, args.limit)
    perf["tasks"]["HellaS."] = _perf_task.to_dict()
    with PerfRecorder(args.device, label="PIQA") as _perf_task:
        results['PIQA'] = eval_piqa(model, tokenizer, args.device, args.batch_size, args.limit)
    perf["tasks"]["PIQA"] = _perf_task.to_dict()
    with PerfRecorder(args.device, label="MathQA") as _perf_task:
        results['MathQA'] = eval_mathqa(model, tokenizer, args.device, args.batch_size, args.limit)
    perf["tasks"]["MathQA"] = _perf_task.to_dict()
    # Average over the six MC tasks
    mc_keys = ['Openb.', 'ARC_e', 'WinoG.', 'HellaS.', 'PIQA', 'MathQA']
    mc_vals = [v for k, v in results.items() if k in mc_keys and isinstance(v, float)]
    results['Average'] = sum(mc_vals) / len(mc_vals) if mc_vals else float('nan')
    # TruthfulQA MC1
    if not args.skip_truthfulqa:
        with PerfRecorder(args.device, label="TruthfulQA") as _perf_task:
            results['TruthfulQA'] = eval_truthfulqa_mc1(model, tokenizer, args.device, args.batch_size, args.limit)
        perf["tasks"]["TruthfulQA"] = _perf_task.to_dict()
    else:
        results['TruthfulQA'] = float('nan')
        perf["tasks"]["TruthfulQA"] = None
    # GSM8K
    if not args.skip_gsm8k:
        with PerfRecorder(args.device, label="GSM8K") as _perf_task:
            results['GSM8K'] = eval_gsm8k(
                model,
                tokenizer,
                args.device,
                args.limit,
                max_new_tokens=args.gsm8k_max_new_tokens,
                batch_size=max(1, args.batch_size // 2),
            )
        perf["tasks"]["GSM8K"] = _perf_task.to_dict()
    else:
        results['GSM8K'] = float('nan')
        perf["tasks"]["GSM8K"] = None

    # Pretty print
    order = ['Openb.', 'ARC_e', 'WinoG.', 'HellaS.', 'PIQA', 'MathQA', 'Average', 'TruthfulQA', 'GSM8K']
    print("\nResults (accuracy, %):")
    for k in order:
        v = results.get(k, float('nan'))
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")


    # Save JSON (optional)
    out_json = _auto_output_json(args, 'benchmark')
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

    args_dict = dict(vars(args))
    if args_dict.get("hf_token"):
        args_dict["hf_token"] = "<REDACTED>"

    payload = {
        'schema': 'svdllm_eval_v1',
        'script': os.path.basename(__file__),
        'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
        'cmd': ' '.join(sys.argv),
        'mode': 'benchmark',
        "args": args_dict,
        "perf": perf,
        'model': model_ref,
        'results': results,
    }
    _write_json(out_json, payload)


if __name__ == '__main__':
    main()
