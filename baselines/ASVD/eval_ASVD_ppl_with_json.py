"""eval_asvd_ppl.py

Perplexity (PPL) evaluator that works with **ASVD HuggingFace repos**.

Why this exists:
  - The provided `eval_general_ppl.py` is tied to another repo layout (utils.model_utils, etc.).
  - ASVD models are usually exported as a HuggingFace-style folder under `huggingface_repos/`.
  - This script evaluates *token-level* PPL on common LM datasets (wikitext2 / ptb / c4)
    and can load ASVD models via `transformers` with a robust fallback for your layout.

Main features:
  - Loads models from:
      * a local HuggingFace folder (e.g. ./huggingface_repos/Llama-2-7b-hf-asvd40)
      * a HuggingFace Hub repo id (downloads via snapshot_download)
  - Supports ASVD remote-code loading (`trust_remote_code=True`), plus a fallback when
    the model folder does NOT contain the python files but they live next to it
    (your screenshot shows `huggingface_repos/modeling_asvd_llama.py` etc).
  - Computes strict token-level PPL (recommended) or a legacy baseline-style PPL.
  - Optional lm-eval-harness integration for word/byte/bpb (if installed).

Example:
  CUDA_VISIBLE_DEVICES=0 python eval_ASVD_ppl.py \
    --model ./huggingface_repos/Llama-2-7b-hf-asvd40 \
    --datasets wikitext2,ptb,c4 --c4_stream --c4_docs 2000 \
    --device cuda --dtype bf16 --seqlen 4096 --batch_size 1

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import datetime as _dt
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

# making the output file

def _jsonify(obj):
    """Make common python/torch/numpy objects JSON-serializable."""
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


def _auto_output_json(args, suffix: str):
    """
    Priority:
      1) --output_json
      2) --output_dir + (<run_name>_<suffix>.json)
      3) None
    """
    if getattr(args, "output_json", None):
        return args.output_json

    out_dir = getattr(args, "output_dir", None)
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)

    run_name = getattr(args, "run_name", None)
    if not run_name:
        base = getattr(args, "model", None) or "model"
        run_name = f"{_safe_tag(base)}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return os.path.join(out_dir, f"{run_name}_{suffix}.json")


def _write_json(path, payload):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2)
    print(f"[Output] Wrote JSON -> {path}")

def _repo_root() -> str:
    # Script is expected to live in the ASVD repo root.
    return os.path.dirname(os.path.abspath(__file__))


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_semicolon_kv_list(s: Optional[str]) -> List[Tuple[str, str]]:
    """Parse items like:  name:path;name2:path2  or just  path1;path2."""
    if not s:
        return []
    out: List[Tuple[str, str]] = []
    for idx, chunk in enumerate([c for c in s.split(";") if c.strip()]):
        if ":" in chunk:
            name, val = chunk.split(":", 1)
            name = name.strip() or f"model_{idx+1}"
            val = val.strip()
        else:
            name, val = f"model_{idx+1}", chunk.strip()
        out.append((name, val))
    return out


def _get_torch_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    m = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = dtype_str.lower().strip()
    if key not in m:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from: {sorted(m.keys())}")
    return m[key]


def _move_to_device(model: torch.nn.Module, device: str, dtype: Optional[torch.dtype]) -> torch.nn.Module:
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device)


def _ensure_tokenizer_pad(tokenizer) -> None:
    # Llama-style tokenizers often have no pad token.
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------


def _resolve_model_path(model_id_or_path: str, hf_token: Optional[str], revision: Optional[str], cache_dir: Optional[str]) -> str:
    """Return a local directory path for a HF repo id or local path."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    # Treat as HF Hub id
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required to download from the Hub. "
            "Install with: pip install huggingface_hub\n"
            f"Original error: {e}"
        )
    return snapshot_download(repo_id=model_id_or_path, revision=revision, cache_dir=cache_dir, token=hf_token)


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _import_from_file(module_name: str, file_path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _find_module_file(module_name: str, search_roots: Sequence[str]) -> Optional[str]:
    """Find <module_name>.py (module_name can be dotted) under any root."""
    rel = module_name.replace(".", os.sep) + ".py"
    for root in search_roots:
        cand = os.path.join(root, rel)
        if os.path.isfile(cand):
            return cand
    # fallback: search by basename only (depth-limited)
    base = os.path.basename(rel)
    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            if base in filenames:
                return os.path.join(dirpath, base)
    return None


def load_model_and_tokenizer(
    model_id_or_path: str,
    *,
    device: str,
    dtype: Optional[torch.dtype],
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Tuple[torch.nn.Module, object, str]:
    """Load a causal LM + tokenizer.

    Returns: (model, tokenizer, resolved_local_path)

    The loader tries:
      1) transformers AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
      2) Fallback for your ASVD layout:
         - Reads config.json -> auto_map entries
         - Imports python files from *adjacent* `huggingface_repos/` folder (or repo root)
         - Instantiates config/model classes and loads weights from the model folder
    """

    local_path = _resolve_model_path(model_id_or_path, hf_token=hf_token, revision=revision, cache_dir=cache_dir)

    # Make sure repo root is importable (so modeling_asvd_llama.py can import `modules/`, etc.)
    rr = _repo_root()
    if rr not in sys.path:
        sys.path.insert(0, rr)
    hf_code_root = os.path.join(rr, "huggingface_repos")
    if os.path.isdir(hf_code_root) and hf_code_root not in sys.path:
        sys.path.insert(0, hf_code_root)

    # --- Try the straightforward HF way first ---
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        _ensure_tokenizer_pad(tok)

        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = _move_to_device(model, device=device, dtype=dtype)
        model.eval()
        # Disable cache to reduce memory and avoid custom cache incompatibilities
        try:
            model.config.use_cache = False
        except Exception:
            pass
        return model, tok, local_path
    except Exception as e_auto:
        # --- Fallback path ---
        # This is common when your model dir doesn't include the remote-code python files,
        # but they exist in ./huggingface_repos next to the model folders.
        print(f"[WARN] AutoModelForCausalLM failed: {type(e_auto).__name__}: {e_auto}")
        print("[WARN] Trying ASVD local-code fallback (auto_map-based import)...")

    config_json = os.path.join(local_path, "config.json")
    if not os.path.isfile(config_json):
        raise FileNotFoundError(
            f"Could not find config.json under {local_path}. "
            "If this is a HF Hub repo, ensure it is a model repo; "
            "if it's a local folder, ensure it contains HF files."
        )
    cfg_dict = _read_json(config_json)
    auto_map = cfg_dict.get("auto_map") or {}
    cfg_entry = auto_map.get("AutoConfig")
    model_entry = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel")
    if not cfg_entry or not model_entry:
        raise RuntimeError(
            "Fallback loader requires config.json to contain `auto_map` entries for AutoConfig and AutoModelForCausalLM. "
            "Your config.json has no/partial auto_map.\n"
            f"Found keys: {list(auto_map.keys())}"
        )

    def _split_entry(entry: str) -> Tuple[str, str]:
        if ":" in entry:
            # HF sometimes stores as "module.py:Class"; support both.
            mod, cls = entry.split(":", 1)
        else:
            parts = entry.split(".")
            if len(parts) < 2:
                raise ValueError(f"Invalid auto_map entry: {entry}")
            mod, cls = ".".join(parts[:-1]), parts[-1]
        return mod, cls

    cfg_mod_name, cfg_cls_name = _split_entry(cfg_entry)
    model_mod_name, model_cls_name = _split_entry(model_entry)

    search_roots = [
        local_path,
        os.path.dirname(local_path),
        hf_code_root,
        rr,
    ]
    cfg_file = _find_module_file(cfg_mod_name, search_roots)
    model_file = _find_module_file(model_mod_name, search_roots)
    if cfg_file is None or model_file is None:
        raise FileNotFoundError(
            "Could not locate ASVD remote-code python files for fallback loader.\n"
            f"  Wanted config module: {cfg_mod_name}  -> {cfg_file}\n"
            f"  Wanted model module:  {model_mod_name} -> {model_file}\n"
            "Searched roots:\n  - " + "\n  - ".join(search_roots)
        )

    cfg_mod = _import_from_file(cfg_mod_name, cfg_file)
    model_mod = _import_from_file(model_mod_name, model_file)

    cfg_cls = getattr(cfg_mod, cfg_cls_name)
    model_cls = getattr(model_mod, model_cls_name)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local_path, use_fast=True)
    _ensure_tokenizer_pad(tok)
    config = cfg_cls.from_pretrained(local_path)
    model = model_cls.from_pretrained(local_path, config=config, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = _move_to_device(model, device=device, dtype=dtype)
    model.eval()
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return model, tok, local_path


# ----------------------------------------------------------------------------
# Dataset loading / packing
# ----------------------------------------------------------------------------


@dataclass
class PackedDataset:
    name: str
    input_ids: torch.LongTensor  # shape (num_sequences, seq_len)


def _load_raw_texts(dataset_name: str, *, c4_stream: bool, c4_docs: int, c4_dataset: Optional[str]) -> List[str]:
    """Load raw texts for a named dataset."""

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "The `datasets` library is required for PPL evaluation. Install with: pip install datasets\n"
            f"Original error: {e}"
        )

    dataset_name = dataset_name.lower()
    if dataset_name in {"wikitext2", "wt2"}:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # filter out empty lines (common practice)
        return [t for t in ds["text"] if t and t.strip()]
    if dataset_name in {"wikitext2_val", "wikitext2_valid", "wt2_val", "wt2_valid"}:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        return [t for t in ds["text"] if t and t.strip()]
    if dataset_name in {"ptb", "penn_treebank"}:
        try:
            ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
            col = "sentence" if "sentence" in ds.column_names else ds.column_names[0]
            return [t for t in ds[col] if t and t.strip()]
        except Exception as e:
            # datasets>=4.0.0: scripts are blocked -> load parquet instead
            parquet_url = "https://huggingface.co/datasets/FALcon6/ptb_text_only/resolve/main/penn_treebank/test/0000.parquet"
            ds = load_dataset("parquet", data_files={"test": [parquet_url]}, split="test")
            col = "sentence" if "sentence" in ds.column_names else ds.column_names[0]
            return [t for t in ds[col] if t and t.strip()]
    if dataset_name == "c4":
        split = "validation"
        parquet_urls = [
            "https://huggingface.co/datasets/allenai/c4/resolve/refs%2Fconvert%2Fparquet/en/partial-validation/0000.parquet",
            "https://huggingface.co/datasets/allenai/c4/resolve/refs%2Fconvert%2Fparquet/en/partial-validation/0001.parquet",
        ]
        ds = load_dataset("parquet", data_files={split: parquet_urls}, split=split, streaming=c4_stream)
        texts = []
        if c4_stream:
            for ex in ds:
                if len(texts) >= int(c4_docs):
                    break
                t = ex.get("text", "")
                if t and t.strip():
                    texts.append(t)
        else:
            ds = ds.select(range(min(int(c4_docs), len(ds))))
            texts = [t for t in ds["text"] if t and t.strip()]
        return texts

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. Supported: wikitext2, wikitext2_val, ptb, c4"
    )


def pack_dataset(
    dataset_name: str,
    tokenizer,
    *,
    seq_len: int,
    c4_stream: bool,
    c4_docs: int,
    c4_dataset: Optional[str],
    add_bos: bool = False,
) -> PackedDataset:
    """Tokenize and pack a dataset into fixed-length sequences."""

    texts = _load_raw_texts(
        dataset_name,
        c4_stream=c4_stream,
        c4_docs=c4_docs,
        c4_dataset=c4_dataset,
    )

    # Join with double newlines (common evaluation convention)
    joined = "\n\n".join(texts)

    # Many tokenizers for causal LMs don't want special tokens here.
    # NOTE: For Llama-family tokenizers you may want a BOS token at the very beginning.
    # Set add_bos=True (via --add_bos) to include tokenizer special tokens.
    enc = tokenizer(joined, return_tensors="pt", add_special_tokens=add_bos)
    ids = enc["input_ids"][0]

    if ids.numel() < seq_len + 1:
        raise RuntimeError(
            f"Dataset {dataset_name} too small after tokenization: got {ids.numel()} tokens for seq_len={seq_len}"
        )

    n_seq = ids.numel() // seq_len
    ids = ids[: n_seq * seq_len]
    packed = ids.view(n_seq, seq_len).contiguous()
    return PackedDataset(name=dataset_name, input_ids=packed)


def make_dataloader(packed: PackedDataset, batch_size: int) -> DataLoader:
    ds = TensorDataset(packed.input_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


# ----------------------------------------------------------------------------
# PPL evaluation
# ----------------------------------------------------------------------------


@torch.no_grad()
def ppl_token_level(
    model,
    loader: DataLoader,
    *,
    device: str,
    max_batches: Optional[int] = None,
    desc: str = "ppl",
) -> float:
    """Strict token-level PPL: exp(total_nll / total_tokens)."""
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0

    for i, (batch_ids,) in enumerate(tqdm(loader, desc=desc)):
        if max_batches is not None and i >= max_batches:
            break
        batch_ids = batch_ids.to(device)

        out = model(
            batch_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = out.logits if hasattr(out, "logits") else out[0]
        if not torch.isfinite(logits).all():
            # skip pathological batch
            continue

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch_ids[:, 1:].contiguous()

        nll = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        total_nll += float(nll.item())
        total_tokens += int(shift_labels.numel())

    if total_tokens == 0:
        return float("nan")
    return float(math.exp(total_nll / total_tokens))


@torch.no_grad()
def ppl_legacy_sample_mean(
    model,
    loader: DataLoader,
    *,
    device: str,
    max_batches: Optional[int] = None,
    desc: str = "legacy_ppl",
) -> float:
    """Legacy baseline-style PPL.

    Mimics the common (but slightly wrong) baseline pattern:
      input_ids = batch[:, :-1]
      logits = model(input_ids)
      shift_logits = logits[:, :-1]
      shift_labels = input_ids[:, 1:]
      denom = num_samples * original_seq_len

    This underestimates PPL vs strict token-level averaging.
    """

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_sum = 0.0
    num_samples = 0
    orig_seq_len: Optional[int] = None

    for i, (batch_ids,) in enumerate(tqdm(loader, desc=desc)):
        if max_batches is not None and i >= max_batches:
            break
        batch_ids = batch_ids.to(device)
        if orig_seq_len is None:
            orig_seq_len = int(batch_ids.shape[1])

        input_ids = batch_ids[:, :-1]
        out = model(
            input_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = out.logits if hasattr(out, "logits") else out[0]
        if not torch.isfinite(logits).all():
            continue
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss_sum += float(loss.sum().item())
        num_samples += int(input_ids.shape[0])

    if num_samples == 0 or orig_seq_len is None:
        return float("nan")
    denom = float(num_samples * orig_seq_len)
    return float(math.exp(loss_sum / denom))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Perplexity evaluator for ASVD HuggingFace repos.")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path or HF repo id. Example: ./huggingface_repos/Llama-2-7b-hf-asvd40",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Alias for --model (kept for compatibility with older commands).",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Evaluate multiple models sequentially. Format: name:path;name2:path2 (or just path1;path2).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="wikitext2,ptb,c4",
        help="Comma-separated datasets: wikitext2, wikitext2_val, ptb, c4",
    )
    p.add_argument("--seqlen", type=int, default=2048, help="Sequence length")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    p.add_argument("--dtype", type=str, default=None, help="fp16/bf16/fp32")
    p.add_argument("--max_batches", type=int, default=None, help="Limit batches (smoke test)")
    p.add_argument(
        "--ppl_method",
        type=str,
        default="token",
        choices=["token", "legacy"],
        help="token=strict token-level PPL (recommended); legacy=baseline-style under-estimated PPL",
    )
    p.add_argument("--revision", type=str, default=None, help="Model revision (HF Hub)")
    p.add_argument("--cache_dir", type=str, default=None, help="HF cache dir")
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable transformers trust_remote_code (recommended for ASVD).",
    )
    p.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Disable trust_remote_code.",
    )

    # C4 options
    p.add_argument(
        "--c4_docs",
        type=int,
        default=None,
        help="Number of C4 validation docs to use (default: auto 200 if c4 is requested)",
    )
    p.add_argument(
        "--c4_stream",
        action="store_true",
        help="Use streaming mode for C4 (avoids huge downloads)",
    )
    p.add_argument(
        "--c4_dataset",
        type=str,
        default=None,
        help="Override C4 dataset name (e.g., stas/c4-en-10k)",
    )

    # Optional lm-eval
    p.add_argument(
        "--metrics",
        type=str,
        default="token",
        help="Comma-separated: token, word, byte, bpb (word/byte/bpb require lm-eval)",
    )
    p.add_argument(
        "--lm_eval_add_bos_token",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="lm-eval: add BOS token (auto avoids double-BOS)",
    )
    p.add_argument(
        "--lm_eval_prefix_token_id",
        type=int,
        default=None,
        help="lm-eval: prefix token id for rolling loglikelihood (defaults to BOS)",
    )
    p.add_argument(
        "--lm_eval_allow_c4_download",
        action="store_true",
        help="Allow lm-eval C4 task to download non-streaming shards (default: skip)",
    )

    p.add_argument(
        "--add_bos",
        action="store_true",
        help="Add BOS/special tokens when tokenizing datasets (sometimes used for Llama-style eval).",
    )
    p.add_argument("--hf_token", type=str, default=None, help="HuggingFace Hub token (if needed for private models)")
    p.add_argument("--output_json", type=str, default=None, help="Write JSON report to this path")
    p.add_argument("--output_dir", type=str, default=None, help="If set (and --output_json not set), auto-write JSON to <output_dir>/<run_name>_<suffix>.json")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name prefix for auto JSON naming (default: <model>_<timestamp>)")
    args = p.parse_args()

    # Backward-compatible alias
    if getattr(args, "checkpoint", None) and not args.model:
        args.model = args.checkpoint
    elif getattr(args, "checkpoint", None) and args.model and args.checkpoint != args.model:
        raise ValueError("Please pass only one of --model or --checkpoint.")


    

    trust_remote_code = True
    if args.no_trust_remote_code:
        trust_remote_code = False
    if args.trust_remote_code:
        trust_remote_code = True

    dtype = _get_torch_dtype(args.dtype)

    model_items: List[Tuple[str, str]] = []
    if args.models:
        model_items = _parse_semicolon_kv_list(args.models)
    elif args.model:
        model_items = [("model", args.model)]
    else:
        raise ValueError("Please provide --model or --models")

    datasets = _parse_csv(args.datasets)

    # Auto-small C4 behavior to match typical scripts:
    c4_docs = args.c4_docs
    c4_stream = bool(args.c4_stream)
    if "c4" in [d.lower() for d in datasets] and c4_docs is None:
        c4_docs = 200
        if not c4_stream:
            c4_stream = True
        print("[Info] C4 requested but --c4_docs not set -> using --c4_docs 200 and --c4_stream")
    if c4_docs is None:
        c4_docs = 2000

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    want_token = "token" in metrics
    want_lm_eval = any(m in metrics for m in ("word", "byte", "bpb"))

    results: Dict[str, Dict[str, float]] = {}

    models_info: Dict[str, Dict[str, object]] = {}

    for name, model_path in model_items:
        print("\n" + "=" * 80)
        print(f"[Model] {name}: {model_path}")
        t0 = time.time()
        model, tokenizer, resolved_path = load_model_and_tokenizer(
            model_path,
            device=args.device,
            dtype=dtype,
            hf_token=args.hf_token,
            revision=args.revision,
            cache_dir=args.cache_dir,
            trust_remote_code=trust_remote_code,
        )
        print(f"[Model] Resolved path: {resolved_path}")

        models_info[name] = {
            "model": model_path,
            "resolved_path": resolved_path,
        }

        # Safety: cap seqlen by model max_position_embeddings if present
        max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0 and args.seqlen > max_pos:
            print(f"[WARN] --seqlen {args.seqlen} > model max_position_embeddings {max_pos}. Capping to {max_pos}.")
            seqlen = max_pos
        else:
            seqlen = args.seqlen

        per_dataset: Dict[str, float] = {}

        if want_token:
            for ds_name in datasets:
                packed = pack_dataset(
                    ds_name,
                    tokenizer,
                    seq_len=seqlen,
                    c4_stream=c4_stream,
                    c4_docs=c4_docs,
                    c4_dataset=args.c4_dataset,
                    add_bos=bool(args.add_bos),
                )
                loader = make_dataloader(packed, batch_size=args.batch_size)
                if args.ppl_method == "legacy":
                    ppl = ppl_legacy_sample_mean(
                        model,
                        loader,
                        device=args.device,
                        max_batches=args.max_batches,
                        desc=f"legacy_ppl[{ds_name}]",
                    )
                else:
                    ppl = ppl_token_level(
                        model,
                        loader,
                        device=args.device,
                        max_batches=args.max_batches,
                        desc=f"ppl[{ds_name}]",
                    )
                per_dataset[f"{ds_name}_ppl"] = ppl
                print(f"[PPL] {name} {ds_name}: {ppl:.4f}")

        if want_lm_eval:
            
            from lm_eval import evaluator
            from lm_eval.models.huggingface import HFLM
                       
            # Disable KV cache
            try:
                model.config.use_cache = False
            except Exception:
                pass

            lm_eval_datasets = [d for d in datasets]
            if "c4" in [d.lower() for d in lm_eval_datasets] and not args.lm_eval_allow_c4_download:
                lm_eval_datasets = [d for d in lm_eval_datasets if d.lower() != "c4"]
                print(
                    "[LM-Eval] Skipping C4 for word/byte/bpb to avoid shard downloads. "
                    "Pass --lm_eval_allow_c4_download to enable."
                )

            add_bos_token = None
            if args.lm_eval_add_bos_token == "true":
                add_bos_token = True
            elif args.lm_eval_add_bos_token == "false":
                add_bos_token = False

            prefix_token_id = args.lm_eval_prefix_token_id
            if prefix_token_id is None and getattr(tokenizer, "bos_token_id", None) is not None:
                prefix_token_id = tokenizer.bos_token_id
            if args.lm_eval_add_bos_token == "auto":
                if getattr(tokenizer, "bos_token_id", None) is None:
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
                max_length=seqlen,
                trust_remote_code=True,
                add_bos_token=add_bos_token,
                prefix_token_id=prefix_token_id,
            )

            if lm_eval_datasets:
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
                    raise RuntimeError("lm-eval returned no results (not rank 0?)")
                # Store raw results for convenience
                per_dataset.update({f"lm_eval_{k}": v for k, v in res.get("results", {}).items()})
                print("\n[LM-Eval] results:")
                print(json.dumps(res.get("results", res), indent=2))
            else:
                print("[LM-Eval] No datasets to evaluate after filtering; skipping.")

        results[name] = per_dataset
        print(f"[Done] {name} in {time.time() - t0:.1f}s")

    print("\n" + "=" * 80)
    print("Final results:")
    print(json.dumps(results, indent=2))

    out_json = _auto_output_json(args, "ppl")
    payload = {
        "schema": "asvd_eval_v1",
        "script": os.path.basename(__file__),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join(sys.argv),
        "mode": "ppl",
        "args": vars(args),
        "models": models_info,
        "results": results,
    }
    _write_json(out_json, payload)


if __name__ == "__main__":
    main()
