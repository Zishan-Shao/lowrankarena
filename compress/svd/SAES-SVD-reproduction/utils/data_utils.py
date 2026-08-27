import os
import random
import torch
import sys
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


def _tokenizer_ok(tokenizer_obj) -> bool:
    try:
        return tokenizer_obj is not None and not isinstance(tokenizer_obj, bool) and callable(tokenizer_obj)
    except Exception:
        return False


def _tokenizer_total_vocab_hint(tokenizer_obj) -> int:
    try:
        n = len(tokenizer_obj)
        if isinstance(n, int) and n > 0:
            return int(n)
    except Exception:
        pass
    try:
        n = getattr(tokenizer_obj, "vocab_size", None)
        if isinstance(n, int) and n > 0:
            return int(n)
    except Exception:
        pass
    return 0


def _normalize_hf_token(hf_token: Optional[str]) -> Optional[str]:
    if hf_token is None:
        return None
    tok = str(hf_token).strip()
    return tok or None


def _load_tokenizer_from_hint(model_hint: str, hf_token: Optional[str] = None):
    hf_token = _normalize_hf_token(hf_token)
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
    # Fallback: direct Llama tokenizers for repos with broken auto-map
    try:
        from transformers import LlamaTokenizerFast, LlamaTokenizer
        for cls in (LlamaTokenizerFast, LlamaTokenizer):
            try:
                tok = cls.from_pretrained(model_hint, token=hf_token)
                if _tokenizer_ok(tok):
                    return tok
            except Exception:
                continue
    except Exception:
        pass
    return None


def _safe_cache_stem(name: str) -> str:
    """
    Turn an arbitrary dataset spec (may include '/', ':', paths, etc.) into a safe
    single path component for cache filenames.
    """
    import hashlib
    import re

    s = str(name or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", s):
        return s
    base = os.path.basename(s)
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_")
    if not base:
        base = "ds"
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{h}"


def _read_local_text_corpus(path: str) -> str:
    """
    Read a local corpus file and return a single concatenated text string.

    Supported:
      - .txt: one document per line (empty lines ignored)
      - .json: list of objects or {"data":[...]} where each object has a text-ish field
      - .jsonl: one JSON object per line with a text-ish field
    """
    import json

    def _obj_to_text(obj) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for k in ("text", "content", "document", "article", "body"):
                v = obj.get(k, None)
                if isinstance(v, str) and v.strip():
                    return v
            for v in obj.values():
                if isinstance(v, str) and v.strip():
                    return v
        return ""

    ext = os.path.splitext(path)[1].lower()
    texts = []
    if ext in (".txt", ".text"):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    texts.append(ln)
    elif ext in (".jsonl",):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                t = _obj_to_text(obj)
                if t and t.strip():
                    texts.append(t)
    elif ext in (".json",):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            obj = obj["data"]
        if isinstance(obj, list):
            for it in obj:
                t = _obj_to_text(it)
                if t and t.strip():
                    texts.append(t)
        else:
            t = _obj_to_text(obj)
            if t and t.strip():
                texts.append(t)
    else:
        # Unknown extension: treat as plain text
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    texts.append(ln)

    return "\n\n".join(texts)


def _split_hf_dataset_spec(spec: str) -> Tuple[str, Optional[str]]:
    s = str(spec or "").strip()
    if not s:
        return s, None
    if ":" in s:
        path, cfg = s.split(":", 1)
        path = path.strip()
        cfg = cfg.strip()
        if path and cfg:
            return path, cfg
    return s, None


def _extract_text_from_hf_example(ex: Any) -> str:
    if ex is None:
        return ""
    if isinstance(ex, str):
        return ex
    if isinstance(ex, dict):
        for key in ("text", "content", "document", "body", "article", "page", "raw_content", "raw"):
            value = ex.get(key, None)
            if isinstance(value, str) and value.strip():
                return value
        for key in ("texts", "contents", "documents"):
            value = ex.get(key, None)
            if isinstance(value, list) and value:
                try:
                    return "\n\n".join(str(x) for x in value if isinstance(x, (str, bytes)) and str(x).strip())
                except Exception:
                    continue
    return ""


def _is_datasets_script_blocked_error(err: BaseException) -> bool:
    msg = str(err or "")
    return "Dataset scripts are no longer supported" in msg or "dataset scripts are no longer supported" in msg


def _load_dataset_streaming_via_hub_datafiles(
    repo_id: str,
    *,
    split: str = "train",
    token: Optional[str] = None,
    config_filter: Optional[str] = None,
    max_files: int = 32,
):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub is required to load {repo_id} without scripts: {e}")

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    if not files:
        raise RuntimeError(f"No files found for dataset repo {repo_id!r}")

    exts = (".parquet", ".jsonl", ".json", ".csv", ".tsv")
    data_files_all = [f for f in files if str(f).lower().endswith(exts)]
    if not data_files_all:
        raise RuntimeError(
            f"No supported data files ({', '.join(exts)}) found in dataset repo {repo_id!r}. "
            "You may need a different mirror when dataset scripts are blocked."
        )

    preferred_ext_order = [".parquet", ".jsonl", ".json", ".csv", ".tsv"]
    chosen_ext = None
    for ext in preferred_ext_order:
        if any(str(f).lower().endswith(ext) for f in data_files_all):
            chosen_ext = ext
            break
    if chosen_ext is None:
        chosen_ext = os.path.splitext(str(data_files_all[0]))[1].lower()

    split_key = str(split or "train").strip().lower()
    if split_key in ("train", "training"):
        split_prefixes = ("train", "training")
    elif split_key in ("validation", "valid", "val", "dev"):
        split_prefixes = ("validation", "valid", "val", "dev")
    elif split_key in ("test",):
        split_prefixes = ("test",)
    else:
        split_prefixes = (split_key,)

    candidates = sorted([f for f in data_files_all if str(f).lower().endswith(chosen_ext)])
    if config_filter:
        cf = str(config_filter).strip().lower()
        if cf:
            filtered = [f for f in candidates if cf in str(f).lower()]
            if filtered:
                candidates = filtered

    split_files = [f for f in candidates if any((f.lower().startswith(p) or f"/{p}" in f.lower()) for p in split_prefixes)]
    chosen_files = split_files if split_files else candidates
    chosen_files = chosen_files[: max(1, int(max_files))]

    local_paths = []
    for f in chosen_files:
        try:
            local_paths.append(hf_hub_download(repo_id, f, repo_type="dataset", token=token))
        except Exception:
            continue
    if not local_paths:
        raise RuntimeError(f"Failed to download any usable data files for dataset repo {repo_id!r}")

    builder_map = {
        ".parquet": "parquet",
        ".jsonl": "json",
        ".json": "json",
        ".csv": "csv",
        ".tsv": "csv",
    }
    builder = builder_map.get(chosen_ext, "json")
    kwargs = {}
    if chosen_ext == ".tsv":
        kwargs["delimiter"] = "\t"
    return load_dataset(builder, data_files={"train": local_paths}, split="train", streaming=True, **kwargs)


def _load_dataset_streaming(
    path: str,
    config: Optional[str],
    *,
    split: str = "train",
    hf_token: Optional[str] = None,
):
    def _call(**auth):
        if config:
            return load_dataset(path, config, split=split, streaming=True, **auth)
        return load_dataset(path, split=split, streaming=True, **auth)

    try:
        if hf_token:
            try:
                return _call(token=hf_token)
            except TypeError:
                try:
                    return _call(use_auth_token=hf_token)
                except TypeError:
                    return _call()
        return _call()
    except Exception as e:
        if _is_datasets_script_blocked_error(e):
            try:
                return _load_dataset_streaming_via_hub_datafiles(
                    repo_id=path,
                    split=split,
                    token=hf_token,
                    config_filter=config,
                    max_files=32,
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to load {path!r} via hub datafiles fallback: {e2}") from e
        raise


def _resolve_streaming_lm_spec(name: str) -> Optional[Dict[str, Any]]:
    path, cfg = _split_hf_dataset_spec(str(name or "").strip())
    lpath = str(path or "").strip().lower()

    if lpath in ("c4", "c4_stream", "allenai/c4"):
        return {"kind": "c4", "repo_id": "allenai/c4", "config": "en", "split": "train", "pretty": "c4"}

    if lpath in ("pileval", "pile_val", "pile-val", "mit-han-lab/pile-val-backup"):
        repo_id = "mit-han-lab/pile-val-backup" if "/" not in lpath else path
        return {"kind": "pileval", "repo_id": repo_id, "config": cfg, "split": "validation", "pretty": "pileval"}

    if lpath in ("web", "web_stream", "fineweb_edu", "fineweb-edu", "finewebedu") or lpath.endswith("/fineweb-edu") or lpath.startswith("huggingfacefw/fineweb-edu"):
        repo_id = "HuggingFaceFW/fineweb-edu" if "/" not in lpath else path
        return {"kind": "fineweb_edu", "repo_id": repo_id, "config": cfg, "split": "train", "pretty": "fineweb_edu"}

    if lpath in ("refinedweb", "refined_web", "falcon-refinedweb", "tiiuae/falcon-refinedweb") or lpath.endswith("/falcon-refinedweb"):
        repo_id = "tiiuae/falcon-refinedweb" if "/" not in lpath else path
        return {"kind": "refinedweb", "repo_id": repo_id, "config": cfg, "split": "train", "pretty": "refinedweb"}

    if lpath in (
        "redpajama_cc",
        "redpajama-cc",
        "red_cc",
        "redpajama_cc_2023",
        "redpajama-cc-2023",
        "datajuicer/redpajama-cc-2023-06-refined-by-data-juicer",
    ):
        repo_id = "datajuicer/redpajama-cc-2023-06-refined-by-data-juicer" if "/" not in lpath else path
        return {"kind": "redpajama_cc", "repo_id": repo_id, "config": cfg, "split": "train", "pretty": "redpajama_cc"}

    if lpath in (
        "redpajama_cc_2022",
        "redpajama-cc-2022",
        "datajuicer/redpajama-cc-2022-05-refined-by-data-juicer",
    ):
        repo_id = "datajuicer/redpajama-cc-2022-05-refined-by-data-juicer" if "/" not in lpath else path
        return {"kind": "redpajama_cc_2022", "repo_id": repo_id, "config": cfg, "split": "train", "pretty": "redpajama_cc_2022"}

    if lpath in ("dolma", "allenai/dolma") or lpath.endswith("/dolma") or lpath.startswith("allenai/dolma"):
        repo_id = "allenai/dolma" if "/" not in lpath else path
        return {"kind": "dolma", "repo_id": repo_id, "config": cfg, "split": "train", "pretty": "dolma"}

    if "/" in lpath:
        return {"kind": "generic", "repo_id": path, "config": cfg, "split": "train", "pretty": path}

    return None


def _sample_streaming_lm_windows(
    name: str,
    *,
    tokenizer,
    seqlen: int,
    budget: int,
    max_docs: int,
    seed: int,
    hf_token: Optional[str] = None,
) -> List[torch.Tensor]:
    info = _resolve_streaming_lm_spec(name)
    if info is None:
        raise ValueError(f"Unknown streaming LM dataset spec: {name!r}")

    stream = _load_dataset_streaming(
        str(info["repo_id"]),
        info.get("config", None),
        split=str(info.get("split", "train")),
        hf_token=hf_token,
    )

    try:
        buf_sz = int(min(10_000, max(1, int(max_docs) if int(max_docs) > 0 else 10_000)))
        stream = stream.shuffle(seed=int(seed), buffer_size=buf_sz)
    except Exception:
        pass

    eos_id = getattr(tokenizer, "eos_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    md = int(max(0, int(max_docs)))
    bgt = int(max(0, int(budget)))
    sl = int(max(1, int(seqlen)))
    if bgt <= 0:
        return []

    stride = int(max(1, sl // 4))
    out: List[torch.Tensor] = []
    token_buf: List[int] = []
    buf_pos = 0
    saw_any_tokens = False

    def _emit() -> None:
        nonlocal token_buf, buf_pos
        while (len(token_buf) - int(buf_pos)) >= sl and len(out) < bgt:
            win = token_buf[int(buf_pos): int(buf_pos) + sl]
            out.append(torch.tensor(win, dtype=torch.long).unsqueeze(0))
            buf_pos += stride
            if buf_pos >= sl * 16:
                token_buf = token_buf[int(buf_pos):]
                buf_pos = 0

    import itertools

    for ex in itertools.islice(iter(stream), md if md > 0 else None):
        text = _extract_text_from_hf_example(ex)
        if not text:
            continue
        try:
            enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_tensors=None)
            ids = enc.get("input_ids", None) if isinstance(enc, dict) else None
            if ids is None:
                ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            continue
        if isinstance(ids, (tuple, list)) and ids and isinstance(ids[0], (tuple, list)):
            ids = ids[0]
        if not ids:
            continue

        if not saw_any_tokens and bos_id is not None:
            try:
                token_buf.append(int(bos_id))
            except Exception:
                pass
        saw_any_tokens = True

        try:
            token_buf.extend(int(x) for x in ids)
        except Exception:
            continue
        if eos_id is not None:
            try:
                token_buf.append(int(eos_id))
            except Exception:
                pass

        _emit()
        if len(out) >= bgt:
            break

    if not out:
        return []
    if len(out) < bgt:
        rng = random.Random(int(seed))
        need = int(bgt) - int(len(out))
        for _ in range(need):
            out.append(out[rng.randrange(len(out))])
    return out


def _load_social_iqa_calib_rows(
    *,
    split: str = "train",
    cache_root: str = "cache/social_iqa",
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load SocialIQA without relying on a Hub dataset script.
    """
    import json
    import pathlib
    import urllib.request
    import zipfile

    split_key = str(split or "train").strip().lower()
    if split_key in ("valid", "val", "validation", "dev"):
        split_key = "dev"
    elif split_key not in ("train", "dev"):
        split_key = "train"

    cache_dir = pathlib.Path(cache_root)
    inner_dir = cache_dir / "socialiqa-train-dev"
    data_path = inner_dir / f"{split_key}.jsonl"
    label_path = inner_dir / f"{split_key}-labels.lst"
    if not (data_path.exists() and label_path.exists()):
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
        zip_path = cache_dir / "socialiqa-train-dev.zip"
        if not zip_path.exists():
            if not quiet:
                print(f"[SocialIQA] Downloading {url} -> {zip_path}")
            urllib.request.urlretrieve(url, zip_path)
        if not quiet:
            print(f"[SocialIQA] Extracting {zip_path} -> {cache_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)

    if not (data_path.exists() and label_path.exists()):
        raise RuntimeError(f"SocialIQA cache missing expected files: {data_path} / {label_path}")

    labels: List[int] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                labels.append(int(s))
            except Exception:
                continue

    rows: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            dp = json.loads(ln)
            if idx < len(labels):
                dp["label"] = int(labels[idx])
            rows.append(dp)
    return rows

def get_calib_train_data(
    name,
    tokenizer,
    nsamples,
    seqlen=2048,
    seed=3,
    batch_size=1,
    dataset_cache_dir=None,
    c4_stream: Optional[bool] = None,
):
    import random
    random.seed(seed)
    tot_text = None
    try:
        name_norm = str(name).lower()
    except Exception:
        name_norm = str(name)
    is_c4_like = name_norm in ("c4", "c4_stream")
    # Ensure we have a callable HF tokenizer. Some older checkpoints or
    # environments may pass a placeholder (e.g., bool). Reload a sane tokenizer
    # if needed using a best-effort model hint from env.
    if not _tokenizer_ok(tokenizer):
        model_hint = os.getenv('SVDLLM_TOKENIZER_MODEL', None)
        if model_hint is None:
            # Fall back to a generic LLaMA tokenizer if model is unknown.
            # This is only used to build calibration text batches.
            model_hint = 'openlm-research/open_llama_7b'
        hf_token = (
            os.getenv('HF_TOKEN')
            or os.getenv('HUGGINGFACE_TOKEN')
            or os.getenv('HUGGINGFACE_HUB_TOKEN')
        )
        tokenizer = _load_tokenizer_from_hint(model_hint, hf_token=hf_token)
    if not _tokenizer_ok(tokenizer):
        raise TypeError(
            "Tokenizer object is not callable and could not be reconstructed; "
            "set SVDLLM_TOKENIZER_MODEL to a cached/local tokenizer or pass a valid tokenizer."
        )
    # Include tokenizer vocab size to avoid cache collisions across different tokenizers/models.
    # Be robust if tokenizer is missing or not a HF tokenizer (e.g., older pickled checkpoints).
    vocab_hint = _tokenizer_total_vocab_hint(tokenizer)
    cache_stem = _safe_cache_stem(name)
    cache_file = f"cache/{cache_stem}_{vocab_hint}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    nsamples += 1 #############################
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        # Guard against cache/tokenizer mismatch: rebuild if any token id exceeds vocab.
        try:
            max_token = max(batch["input_ids"].max().item() for batch in traindataset)
            vsize = _tokenizer_total_vocab_hint(tokenizer)
            if isinstance(vsize, int) and vsize > 0 and max_token >= vsize:
                print(f"[Cache] Discarding cached calib data at {cache_file} (max token {max_token} >= vocab {vsize}); regenerating.")
                traindataset = None
        except Exception:
            traindataset = None
        if traindataset is not None:
            return traindataset

    # Local corpus adapter (no HF datasets needed):
    #   - local_text:/path/to/file.txt
    #   - local_jsonl:/path/to/file.jsonl
    #   - local_json:/path/to/file.json
    #   - file:/path (auto-detect by extension)
    try:
        spec = str(name or "").strip()
    except Exception:
        spec = ""
    spec_l = spec.lower()
    local_path = None
    if spec_l.startswith(("local_text:", "local_text=", "local_txt:", "local_txt=")):
        local_path = spec.split(":", 1)[1] if ":" in spec else spec.split("=", 1)[1]
    elif spec_l.startswith(("local_jsonl:", "local_jsonl=", "jsonl:", "jsonl=")):
        local_path = spec.split(":", 1)[1] if ":" in spec else spec.split("=", 1)[1]
    elif spec_l.startswith(("local_json:", "local_json=", "json:", "json=")):
        local_path = spec.split(":", 1)[1] if ":" in spec else spec.split("=", 1)[1]
    elif spec_l.startswith(("local:", "local=", "file:", "file=")):
        local_path = spec.split(":", 1)[1] if ":" in spec else spec.split("=", 1)[1]

    if local_path is not None:
        local_path = str(local_path).strip()
        if not os.path.isabs(local_path):
            local_path = os.path.join(parent_path, local_path)
        local_path = os.path.abspath(local_path)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local calib corpus not found: {local_path}")
        tot_text = _read_local_text_corpus(local_path)
        if not tot_text.strip():
            raise ValueError(f"Local calib corpus is empty after parsing: {local_path}")

        enc = tokenizer(tot_text, return_tensors="pt")
        ids = enc.input_ids
        if ids is None or ids.numel() < (int(seqlen) + 1):
            raise ValueError(
                f"Local calib corpus too small for seqlen={seqlen}: tokens={0 if ids is None else int(ids.numel())} path={local_path}"
            )

        traindataset = []
        buf = []
        for _ in tqdm(range(int(nsamples)), desc=f"[local:{os.path.basename(local_path)}] build calib", leave=False):
            start = random.randint(0, ids.shape[1] - int(seqlen) - 1)
            window = ids[:, start:start + int(seqlen)]
            buf.append(window)
            if len(buf) >= int(batch_size):
                inp = torch.cat(buf, dim=0)
                traindataset.append({"input_ids": inp, "attention_mask": torch.ones_like(inp)})
                buf = []
        if buf:
            inp = torch.cat(buf, dim=0)
            traindataset.append({"input_ids": inp, "attention_mask": torch.ones_like(inp)})
        torch.save(traindataset, cache_file)
        return traindataset
    # Generic streaming LM corpora (PileVal / FineWeb-Edu / RefinedWeb / RedPajama-CC / generic HF repos).
    # C4 keeps its dedicated path below because we have custom fallback behavior for it.
    stream_info = _resolve_streaming_lm_spec(spec) if spec else None
    if stream_info is not None and not is_c4_like:
        hf_token = (
            os.getenv('HF_TOKEN')
            or os.getenv('HUGGINGFACE_TOKEN')
            or os.getenv('HUGGINGFACE_HUB_TOKEN')
        )
        stream_seed = int(seed)
        max_docs = max(int(nsamples) * 8, int(nsamples) + 256)
        windows = _sample_streaming_lm_windows(
            spec,
            tokenizer=tokenizer,
            seqlen=int(seqlen),
            budget=int(nsamples),
            max_docs=int(max_docs),
            seed=stream_seed,
            hf_token=hf_token,
        )
        if not windows:
            raise RuntimeError(
                f"Streaming LM calibration produced 0 windows for dataset '{name}' "
                f"(seqlen={seqlen}, nsamples={nsamples})."
            )
        traindataset = []
        buf = []
        for window in windows:
            buf.append(window)
            if len(buf) >= int(batch_size):
                inp = torch.cat(buf, dim=0)
                traindataset.append({"input_ids": inp, "attention_mask": torch.ones_like(inp)})
                buf = []
        if buf:
            inp = torch.cat(buf, dim=0)
            traindataset.append({"input_ids": inp, "attention_mask": torch.ones_like(inp)})
        torch.save(traindataset, cache_file)
        return traindataset
    if is_c4_like:
        # Streaming by default to avoid enumerating all 1,024 shards.
        # Override via c4_stream=False or env SVDLLM_C4_STREAM=0 if needed.
        if c4_stream is None:
            try:
                env_flag = os.getenv('SVDLLM_C4_STREAM', '').strip()
                if env_flag == '':
                    # Default to streaming ON
                    prefer_streaming = True
                else:
                    prefer_streaming = env_flag not in ('0', 'false', 'False', 'no', 'NO')
            except Exception:
                prefer_streaming = True
        else:
            prefer_streaming = bool(c4_stream)
        small_c4_loaded = False
        use_streaming_c4 = False
        c4_stream = None
        # Priority order:
        # 1) Streaming (default)
        # 2) Local JSON if exists
        # 3) Small curated subset (stas/c4-en-10k)
        # 4) Official builder tiny slice (slowest; avoid when possible)
        if prefer_streaming:
            try:
                print("[C4] Using streaming: allenai/c4 'en' (train, streaming=True; scanning limited docs).")
                traindata = load_dataset("allenai/c4", "en", split="train", streaming=True, cache_dir=dataset_cache_dir)
                use_streaming_c4 = True
            except Exception:
                # Fall through to non-stream candidates
                use_streaming_c4 = False
        if not use_streaming_c4:
            try:
                traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
                small_c4_loaded = True
                print("[C4] Using local utils/c4-train.json for calibration.")
            except Exception:
                try:
                    print("[C4] Using small subset: stas/c4-en-10k (train[:1000]) for calibration.")
                    traindata = load_dataset("stas/c4-en-10k", split="train[:1000]", cache_dir=dataset_cache_dir)
                    small_c4_loaded = True
                except Exception:
                    # As a last resort, fall back to the official builder (may enumerate many files)
                    print("[C4] Falling back to HuggingFace allenai/c4 'en' (train[:200]).")
                    try:
                        traindata = load_dataset("allenai/c4", "en", split="train[:200]", cache_dir=dataset_cache_dir)
                    except Exception:
                        traindata = load_dataset("c4", "en", split="train[:200]", cache_dir=dataset_cache_dir)
                    tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        try:
            traindata = load_dataset(
                'ptb_text_only',
                'penn_treebank',
                split='train',
                cache_dir=dataset_cache_dir,
                trust_remote_code=True,
            )
            tot_text = "\n\n".join(traindata["sentence"])
        except Exception as e:
            # Fallback to raw PTB files (wojzaremba/lstm repo) when dataset scripts are not supported
            import urllib.request
            import pathlib
            cache_dir = pathlib.Path('cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt'
            ptb_path = cache_dir / 'ptb_train.txt'
            if not ptb_path.exists():
                print(f"[PTB] Falling back to raw URL for train split: {url}")
                urllib.request.urlretrieve(url, ptb_path)
            with open(ptb_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            tot_text = "\n\n".join(lines)
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
        tot_text = "\n\n".join(traindata["text"])
    elif name.lower() in ("yahma/alpaca-cleaned", "tatsu-lab/alpaca", "alpaca", "alpaca-cleaned"):
        # Alpaca-style instruction data (instruction/input/output)
        try:
            from utils.Prompter import Prompter
        except Exception:
            Prompter = None
        try:
            ds = load_dataset(name, cache_dir=dataset_cache_dir)
            try:
                split = ds["train"]
            except Exception:
                split = ds
        except Exception:
            split = load_dataset(name, split="train", cache_dir=dataset_cache_dir)
        prompter = Prompter("alpaca") if Prompter is not None else None
        chunks = []
        for item in split:
            instruction = str(item.get("instruction", ""))
            inp = item.get("input", None)
            output = str(item.get("output", ""))
            if not instruction and not output and not inp:
                continue
            if prompter is not None:
                text = prompter.generate_prompt(instruction, inp, output)
            else:
                if inp:
                    text = f"Instruction: {instruction}\nInput: {inp}\nResponse: {output}"
                else:
                    text = f"Instruction: {instruction}\nResponse: {output}"
            chunks.append(text)
        if not chunks:
            raise ValueError("No valid data found in Alpaca dataset")
        tot_text = "\n\n".join(chunks)
    elif name.lower() in ("hellaswag", "piqa", "winogrande", "winogrande/xl", "winogrande/l", "winogrande/m", "winogrande/s"):
        # MCQ-style datasets mapped to instruction text
        lname = name.lower()
        try:
            if lname == "hellaswag":
                ds = load_dataset("hellaswag", split="train", cache_dir=dataset_cache_dir)
                chunks = []
                letters = ["A", "B", "C", "D"]
                for item in ds:
                    ctx = item.get("ctx", "") or item.get("context", "")
                    endings = item.get("endings") or []
                    label = item.get("label")
                    opts = [f"{letters[i]}) {endings[i]}" for i in range(min(4, len(endings)))]
                    ans = letters[int(label)] if isinstance(label, int) and int(label) < len(letters) else str(label)
                    chunks.append(f"Instruction: Choose the most plausible ending.\nInput: {ctx}\nOptions: " + " ".join(opts) + f"\nAnswer: {ans}")
            elif lname == "piqa":
                ds = load_dataset("piqa", split="train", cache_dir=dataset_cache_dir)
                chunks = []
                for item in ds:
                    goal = item.get("goal", "")
                    sol1 = item.get("sol1", "")
                    sol2 = item.get("sol2", "")
                    label = item.get("label", 0)
                    ans = "A" if str(label) == "0" else "B"
                    chunks.append(f"Instruction: Pick the more sensible solution.\nInput: Goal: {goal}\nOptions: A) {sol1} B) {sol2}\nAnswer: {ans}")
            else:
                cfg = lname.split("/")[-1] if "/" in lname else "xl"
                try:
                    ds = load_dataset("winogrande", cfg, split="train", cache_dir=dataset_cache_dir)
                except Exception:
                    ds = load_dataset("winogrande", split="train", cache_dir=dataset_cache_dir)
                chunks = []
                for item in ds:
                    sent = item.get("sentence", "")
                    o1 = item.get("option1", "")
                    o2 = item.get("option2", "")
                    ans = item.get("answer", "")
                    label = "1" if str(ans) in ("1", "A") else "2"
                    chunks.append(f"Instruction: Fill in the blank with the correct option.\nInput: {sent}\nOptions: 1) {o1} 2) {o2}\nAnswer: {label}")
            if not chunks:
                raise ValueError("No valid data found in MCQ dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load MCQ dataset '{name}': {e}")
    elif name.lower() in ("ai2_arc_easy", "arc_easy", "ai2_arc/arc-easy", "ai2_arc_challenge", "arc_challenge", "ai2_arc/arc-challenge"):
        # ARC Easy/Challenge (MCQ)
        lname = name.lower()
        config = "ARC-Easy" if "easy" in lname else "ARC-Challenge"
        try:
            ds = load_dataset("ai2_arc", config, split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for item in ds:
                q = item.get("question", {})
                stem = q.get("stem") if isinstance(q, dict) else item.get("question", "")
                choices = q.get("choices") if isinstance(q, dict) else item.get("choices", {})
                texts = choices.get("text") if isinstance(choices, dict) else None
                labels = choices.get("label") if isinstance(choices, dict) else None
                opts = []
                if texts and labels:
                    for i in range(min(4, len(texts))):
                        opts.append(f"{labels[i]}) {texts[i]}")
                ans = item.get("answerKey", "")
                chunks.append(f"Instruction: Select the correct option.\nInput: {stem}\nOptions: " + " ".join(opts) + f"\nAnswer: {ans}")
            if not chunks:
                raise ValueError("No valid data found in ARC dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load ARC dataset '{name}': {e}")
    elif name.lower() in ("openbookqa", "openbookqa/main"):
        # OpenBookQA (MCQ)
        try:
            try:
                ds = load_dataset("openbookqa", "main", split="train", cache_dir=dataset_cache_dir)
            except Exception:
                ds = load_dataset("openbookqa", split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for item in ds:
                stem = item.get("question_stem") or item.get("question", "")
                choices = item.get("choices") or {}
                texts = (choices.get("text") if isinstance(choices, dict) else None) or []
                labels = (choices.get("label") if isinstance(choices, dict) else None) or []
                opts = [f"{labels[i]}) {texts[i]}" for i in range(min(4, len(texts)))]
                ans = item.get("answerKey", "")
                chunks.append(f"Instruction: Answer the science question.\nInput: {stem}\nOptions: " + " ".join(opts) + f"\nAnswer: {ans}")
            if not chunks:
                raise ValueError("No valid data found in OpenBookQA dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load OpenBookQA dataset '{name}': {e}")
    elif name.lower() in ("race", "race/all", "race:all"):
        # RACE reading-comprehension MCQ
        try:
            try:
                ds = load_dataset("race", "all", split="train", cache_dir=dataset_cache_dir)
            except Exception:
                try:
                    ds_h = load_dataset("race", "high", split="train", cache_dir=dataset_cache_dir)
                    ds_m = load_dataset("race", "middle", split="train", cache_dir=dataset_cache_dir)
                    ds = list(ds_h) + list(ds_m)
                except Exception:
                    ds = load_dataset("race", split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for item in ds:
                art = str(item.get("article", "") or item.get("passage", "")).strip()
                q = str(item.get("question", "")).strip()
                opts = item.get("options") or item.get("choices") or []
                if not isinstance(opts, list):
                    opts = []
                opts = [str(x).strip() for x in opts if str(x).strip()]
                if len(opts) < 2:
                    continue
                labels = [chr(ord("A") + i) for i in range(len(opts))]
                ans = str(item.get("answer", item.get("label", ""))).strip().upper()
                if ans not in labels:
                    continue
                ctx = f"Article: {art}\n" if art else ""
                opt_text = " ".join([f"{labels[i]}) {opts[i]}" for i in range(len(opts))])
                chunks.append(
                    f"Instruction: Answer the reading comprehension question.\n"
                    f"{ctx}Question: {q}\nOptions: {opt_text}\nAnswer: {ans}"
                )
            if not chunks:
                raise ValueError("No valid data found in RACE dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load RACE dataset '{name}': {e}")
    elif name.lower() in ("sciq", "allenai/sciq"):
        # SciQ science QA (MCQ)
        try:
            try:
                ds = load_dataset("sciq", split="train", cache_dir=dataset_cache_dir)
            except Exception:
                ds = load_dataset("allenai/sciq", split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for idx, item in enumerate(ds):
                q = str(item.get("question", "")).strip()
                support = str(item.get("support", "") or item.get("context", "")).strip()
                correct = str(item.get("correct_answer", "") or item.get("answer", "")).strip()
                distractors = item.get("distractors")
                if not isinstance(distractors, list):
                    distractors = [
                        item.get("distractor1", ""),
                        item.get("distractor2", ""),
                        item.get("distractor3", ""),
                    ]
                distractors = [str(x).strip() for x in distractors if str(x).strip()]
                if not q or not correct or len(distractors) < 1:
                    continue
                opts = [correct] + distractors
                rr = random.Random(int(seed) + 1337 + int(idx))
                rr.shuffle(opts)
                ans = chr(ord("A") + opts.index(correct))
                labels = [chr(ord("A") + i) for i in range(len(opts))]
                opt_text = " ".join([f"{labels[i]}) {opts[i]}" for i in range(len(opts))])
                ctx = f"Context: {support}\n" if support else ""
                chunks.append(
                    f"Instruction: Answer the science question.\n"
                    f"{ctx}Question: {q}\nOptions: {opt_text}\nAnswer: {ans}"
                )
            if not chunks:
                raise ValueError("No valid data found in SciQ dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load SciQ dataset '{name}': {e}")
    elif name.lower() in ("social_iqa", "social_i_qa", "allenai/social_i_qa"):
        # SocialIQA (MCQ)
        try:
            try:
                ds = load_dataset("allenai/social_i_qa", split="train", cache_dir=dataset_cache_dir)
                rows = list(ds)
            except Exception:
                rows = _load_social_iqa_calib_rows(split="train")
            chunks = []
            for item in rows:
                ctx = str(item.get("context", "") or item.get("ctx", "")).strip()
                q = str(item.get("question", "")).strip()
                opts = [
                    str(item.get("answerA", "")).strip(),
                    str(item.get("answerB", "")).strip(),
                    str(item.get("answerC", "")).strip(),
                ]
                if len([x for x in opts if x]) < 2:
                    continue
                lab = item.get("label", item.get("answer", ""))
                try:
                    corr = int(lab)
                except Exception:
                    corr = -1
                if corr in (1, 2, 3):
                    corr -= 1
                if corr < 0 or corr >= len(opts):
                    continue
                labels = [chr(ord("A") + i) for i in range(len(opts))]
                ans = labels[corr]
                opt_text = " ".join([f"{labels[i]}) {opts[i]}" for i in range(len(opts))])
                ctx_txt = f"Context: {ctx}\n" if ctx else ""
                chunks.append(
                    f"Instruction: Choose the most socially appropriate answer.\n"
                    f"{ctx_txt}Question: {q}\nOptions: {opt_text}\nAnswer: {ans}"
                )
            if not chunks:
                raise ValueError("No valid data found in SocialIQA dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load SocialIQA dataset '{name}': {e}")
    elif name.lower() in ("cola", "glue/cola", "glue_cola", "glue-cola"):
        # GLUE CoLA: linguistic acceptability
        try:
            ds = load_dataset("glue", "cola", split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for item in ds:
                sent = str(item.get("sentence", "") or "")
                lab = item.get("label", 0)
                try:
                    lab_i = int(lab)
                except Exception:
                    lab_i = 1 if str(lab).strip().lower() in ("1", "true", "yes") else 0
                out = "acceptable" if lab_i == 1 else "unacceptable"
                chunks.append(
                    "Instruction: Determine whether the following English sentence is grammatically acceptable.\n"
                    f"Input: {sent}\n"
                    f"Answer: {out}"
                )
            if not chunks:
                raise ValueError("No valid data found in GLUE CoLA dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load GLUE CoLA: {e}")
    elif name.lower() in ("sst2", "glue/sst2", "glue_sst2", "glue-sst2"):
        # GLUE SST-2: sentiment classification
        try:
            ds = load_dataset("glue", "sst2", split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for item in ds:
                sent = str(item.get("sentence", "") or "")
                lab = item.get("label", 0)
                try:
                    lab_i = int(lab)
                except Exception:
                    lab_i = 1 if str(lab).strip().lower() in ("1", "true", "yes", "pos", "positive") else 0
                out = "positive" if lab_i == 1 else "negative"
                chunks.append(
                    "Instruction: Classify the sentiment of the sentence as positive or negative.\n"
                    f"Input: {sent}\n"
                    f"Answer: {out}"
                )
            if not chunks:
                raise ValueError("No valid data found in GLUE SST-2 dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load GLUE SST-2: {e}")
    elif name.lower() == "gsm8k":
        # GSM8K: math reasoning dataset
        try:
            traindata = load_dataset("gsm8k", "main", split="train", cache_dir=dataset_cache_dir)
            # GSM8K has 'question' and 'answer' fields
            chunks = []
            for item in traindata:
                question = str(item.get('question', item.get('Question', '')))
                answer = str(item.get('answer', item.get('Answer', '')))
                if question or answer:
                    chunks.append(f"Question: {question}\nAnswer: {answer}")
            if not chunks:
                raise ValueError("No valid data found in GSM8K dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load GSM8K: {e}")
    elif name.lower() == "commonsenseqa":
        # CommonsenseQA: commonsense reasoning
        try:
            traindata = load_dataset("commonsense_qa", "default", split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for item in traindata:
                question = str(item.get('question', item.get('Question', '')))
                choices = item.get('choices', {})
                if isinstance(choices, dict) and 'text' in choices:
                    choice_text = ' '.join([str(c) for c in choices['text']])
                elif isinstance(choices, list):
                    choice_text = ' '.join([str(c) for c in choices])
                else:
                    choice_text = str(choices) if choices else ''
                answer = str(item.get('answerKey', item.get('answer', '')))
                if question:
                    chunks.append(f"Question: {question}\nChoices: {choice_text}\nAnswer: {answer}")
            if not chunks:
                raise ValueError("No valid data found in CommonsenseQA dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load CommonsenseQA: {e}")
    elif name.lower() == "humaneval":
        # HumanEval: code generation dataset
        try:
            traindata = load_dataset("openai/humaneval", split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for item in traindata:
                prompt = str(item.get('prompt', item.get('Prompt', '')))
                task_id = str(item.get('task_id', item.get('task_id', '')))
                if prompt:
                    chunks.append(f"Task {task_id}:\n{prompt}")
            if not chunks:
                raise ValueError("No valid data found in HumanEval dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load HumanEval: {e}")
    elif name.lower() == "aqua":
        # AQuA: algebraic word problems
        try:
            traindata = load_dataset("aqua_rat", split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for item in traindata:
                question = str(item.get('question', item.get('Question', '')))
                options = str(item.get('options', item.get('Options', '')))
                correct = str(item.get('correct', item.get('correct', '')))
                if question:
                    chunks.append(f"Question: {question}\nOptions: {options}\nCorrect: {correct}")
            if not chunks:
                raise ValueError("No valid data found in AQuA dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load AQuA: {e}")
    elif name.lower() == "strategyqa":
        # StrategyQA: strategic reasoning
        try:
            traindata = load_dataset("metaeval/strategyqa", split="train", cache_dir=dataset_cache_dir)
            chunks = []
            for item in traindata:
                question = str(item.get('question', item.get('Question', '')))
                answer = str(item.get('answer', item.get('Answer', '')))
                facts = item.get('facts', item.get('Facts', []))
                facts_text = ' '.join([str(f) for f in facts]) if isinstance(facts, list) else str(facts)
                if question:
                    chunks.append(f"Question: {question}\nFacts: {facts_text}\nAnswer: {answer}")
            if not chunks:
                raise ValueError("No valid data found in StrategyQA dataset")
            tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load StrategyQA: {e}")
    elif name.lower() == "multiarith":
        # MultiArith: arithmetic word problems
        # MultiArith is often included in math reasoning benchmarks
        # Try loading from common sources or use GSM8K as similar alternative
        try:
            # Try loading from a math reasoning collection if available
            try:
                # Some collections include MultiArith
                traindata = load_dataset("lighteval/MultiArith", split="train", cache_dir=dataset_cache_dir)
                chunks = []
                for item in traindata:
                    question = str(item.get('question', item.get('input', '')))
                    answer = str(item.get('answer', item.get('output', '')))
                    chunks.append(f"Question: {question}\nAnswer: {answer}")
                tot_text = "\n\n".join(chunks)
            except Exception:
                # Fallback: use GSM8K which has similar arithmetic problems
                print("[MultiArith] Falling back to GSM8K dataset (similar arithmetic problems)")
                traindata = load_dataset("gsm8k", "main", split="train", cache_dir=dataset_cache_dir)
                chunks = []
                # Use a subset to match MultiArith's smaller size
                for item in traindata[:min(len(traindata), nsamples * 3)]:
                    question = str(item.get('question', ''))
                    answer = str(item.get('answer', ''))
                    chunks.append(f"Question: {question}\nAnswer: {answer}")
                tot_text = "\n\n".join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load MultiArith: {e}")
    elif name.lower() in ("piqa", "mathqa", "math_qa", "arc_challenge", "arc-challenge", "arc-easy", "arc_easy", "arc"):
        # Handle expressivity datasets (PIQA, MathQA, ARC) using local loaders
        try:
            # Import here to avoid circular dependencies and conflict with HuggingFace's 'datasets' package
            import importlib.util
            load_data_path = os.path.join(parent_path, 'datasets', 'load_data.py')
            if not os.path.isfile(load_data_path):
                raise FileNotFoundError(f"Could not find load_data.py at {load_data_path}")
            spec = importlib.util.spec_from_file_location('local_datasets_load_data', load_data_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not create spec for {load_data_path}")
            load_data_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(load_data_mod)
            get_local_dataset = getattr(load_data_mod, 'get_local_dataset', None)
            if not get_local_dataset:
                raise AttributeError("get_local_dataset not found in load_data module")
            items = get_local_dataset(name, split='validation')
            if not items:
                items = get_local_dataset(name, split='train')
            if not items:
                raise ValueError(f"No data found for dataset '{name}'")
            # Build a long text corpus from prompts and choices
            chunks = []
            for it in items:
                prompt = str(it.get('prompt', ''))
                choices = it.get('choices', [])
                ch_text = ' '.join([str(c) for c in choices]) if isinstance(choices, (list, tuple)) else ''
                chunks.append(prompt + '\n' + ch_text)
            tot_text = '\n\n'.join(chunks)
        except Exception as e:
            raise NotImplementedError(f"Failed to load dataset '{name}': {e}")
    else:
        raise NotImplementedError
    traindataset = []
    if is_c4_like and small_c4_loaded:
        # Sample nsamples random documents and take a random seqlen window from each
        # Build 1-sample batches to match expected structure
        for _ in tqdm(range(nsamples), desc=f"[c4] build calib", leave=False):
            # Keep drawing until we find a doc with enough tokens
            for _retry in range(20):
                idx = random.randint(0, len(traindata) - 1)
                text = traindata[idx]['text'] if isinstance(traindata[idx], dict) else traindata[idx]['text']
                enc = tokenizer(text, return_tensors="pt")
                T = enc.input_ids.shape[1]
                if T >= seqlen + 1:
                    start = random.randint(0, T - seqlen - 1)
                    window = enc.input_ids[:, start:start + seqlen]
                    attn = torch.ones_like(window)
                    traindataset.append({"input_ids": window, "attention_mask": attn})
                    break
        # No caching concat batching; each entry is already a batch of size 1
    elif is_c4_like and use_streaming_c4:
        # Pack streamed docs into a token buffer so short C4 documents can still
        # contribute to full-length calibration windows.
        taken = 0
        docs_seen = 0
        max_docs = max(int(nsamples) * 8, int(nsamples) + 256)
        token_buffer = torch.empty((0,), dtype=torch.long)
        for item in tqdm(traindata, desc="[c4(stream)] scan docs", leave=False):
            docs_seen += 1
            try:
                text = item.get('text', '') if isinstance(item, dict) else item['text']
            except Exception:
                continue
            if not text:
                continue
            enc = tokenizer(text, return_tensors="pt")
            ids = getattr(enc, "input_ids", None)
            if ids is None or ids.numel() == 0:
                continue
            ids_1d = ids[0]
            if ids_1d.numel() == 0:
                continue
            token_buffer = torch.cat((token_buffer, ids_1d.to(dtype=torch.long, device="cpu")), dim=0)
            while token_buffer.numel() >= int(seqlen) and taken < int(nsamples):
                window = token_buffer[:int(seqlen)].unsqueeze(0)
                token_buffer = token_buffer[int(seqlen):]
                attn = torch.ones_like(window)
                traindataset.append({"input_ids": window, "attention_mask": attn})
                taken += 1
            if taken >= int(nsamples):
                break
            if docs_seen >= max_docs and token_buffer.numel() < int(seqlen):
                break
        if not traindataset:
            raise RuntimeError(
                f"C4 streaming calibration produced 0 windows for seqlen={seqlen} after scanning {docs_seen} docs."
            )
        torch.save(traindataset, cache_file)
        return traindataset
    else:
        # Original behavior (used for non-C4 or when official c4 fallback used)
        if tot_text is None:
            raise RuntimeError(
                f"Calibration corpus text is unavailable for dataset '{name}'. "
                "This usually means the dataset loader path did not materialize text or streaming sampling failed."
            )
        for s in tqdm(range(nsamples), desc=f"[{name}] build calib", leave=False):
            i = random.randint(0, len(tot_text) - seqlen - 1)
            j = i + seqlen * 10
            trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
            if trainenc.input_ids.shape[1] < seqlen:
                s = s - 1
                continue
            if s % batch_size == 0:
                if s != 0:
                    attention_mask = torch.ones_like(inp)
                    traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
                inp = trainenc.input_ids[:, :seqlen]
            else:
                inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)
    torch.save(traindataset, cache_file)
    return traindataset


def get_mixed_calib_train_data(
    tokenizer,
    nsamples,
    seqlen=2048,
    seed=3,
    bucket_props: str = "LM:0.4,INST:0.4,MCQ:0.0,MATH:0.2",
    bucket_lm_datasets: Optional[str] = "wikitext2,ptb,c4",
    bucket_inst_datasets: Optional[str] = "yahma/alpaca-cleaned",
    bucket_mcq_datasets: Optional[str] = "",
    bucket_math_datasets: Optional[str] = "gsm8k",
    dataset_cache_dir=None,
    c4_stream: Optional[bool] = None,
    per_bucket: bool = False,
    dump_bucket_debug: bool = False,
):
    import random
    total = int(nsamples)
    if total <= 0:
        return []

    def _parse_props(s: str) -> dict:
        out = {"LM": 0.4, "INST": 0.4, "MCQ": 0.0, "MATH": 0.2}
        try:
            for seg in (s or "").split(','):
                if not seg.strip():
                    continue
                k, v = seg.split(':')
                out[k.strip().upper()] = float(v)
        except Exception:
            pass
        sm = sum(out.values())
        if sm > 0:
            for k in out:
                out[k] = out[k] / sm
        return out

    def _split_names(s: Optional[str]) -> list:
        return [n.strip() for n in (s or "").split(',') if n.strip()]

    def _alloc(names: list, budget: int) -> dict:
        if not names or budget <= 0:
            return {}
        per = budget // len(names)
        rem = budget % len(names)
        return {name: per + (1 if i < rem else 0) for i, name in enumerate(names)}

    lm_names = _split_names(bucket_lm_datasets)
    inst_names = _split_names(bucket_inst_datasets)
    mcq_names = _split_names(bucket_mcq_datasets)
    math_names = _split_names(bucket_math_datasets)
    if per_bucket:
        props = _parse_props(bucket_props)
        bucket_budget = {}
        if lm_names and props.get("LM", 0.0) > 0:
            bucket_budget["LM"] = total
        if inst_names and props.get("INST", 0.0) > 0:
            bucket_budget["INST"] = total
        if mcq_names and props.get("MCQ", 0.0) > 0:
            bucket_budget["MCQ"] = total
        if math_names and props.get("MATH", 0.0) > 0:
            bucket_budget["MATH"] = total
        total = sum(bucket_budget.values())
        if total <= 0:
            return []
    else:
        props = _parse_props(bucket_props)
        bucket_budget = {k: int(round(total * props.get(k, 0.0))) for k in ("LM", "INST", "MCQ", "MATH")}
        diff = total - sum(bucket_budget.values())
        if diff != 0:
            bucket_budget["LM"] = max(0, bucket_budget.get("LM", 0) + diff)

    mixed = []
    counts = {}
    seed_offset = 0

    def _load(name: str, budget: int, offset: int):
        if budget <= 0:
            return []
        try:
            data = get_calib_train_data(
                name,
                tokenizer,
                budget,
                seqlen=seqlen,
                seed=seed + offset,
                dataset_cache_dir=dataset_cache_dir,
                c4_stream=c4_stream,
            )
        except Exception as e:
            msg = str(e).strip()
            if msg:
                print(f"[MixCalib] Skip dataset {name}: {type(e).__name__}: {msg}")
            else:
                print(f"[MixCalib] Skip dataset {name}: {type(e).__name__}: {e!r}")
            return []
        if len(data) > budget:
            data = data[:budget]
        counts[name] = counts.get(name, 0) + len(data)
        return data

    for name, budget in _alloc(lm_names, bucket_budget.get("LM", 0)).items():
        mixed.extend(_load(name, budget, seed_offset))
        seed_offset += 1
    for name, budget in _alloc(inst_names, bucket_budget.get("INST", 0)).items():
        mixed.extend(_load(name, budget, seed_offset))
        seed_offset += 1
    for name, budget in _alloc(mcq_names, bucket_budget.get("MCQ", 0)).items():
        mixed.extend(_load(name, budget, seed_offset))
        seed_offset += 1
    for name, budget in _alloc(math_names, bucket_budget.get("MATH", 0)).items():
        mixed.extend(_load(name, budget, seed_offset))
        seed_offset += 1

    # Top up if any bucket failed or was empty.
    missing = total - len(mixed)
    if missing > 0:
        fallback = (lm_names or inst_names or mcq_names or math_names or ["wikitext2"])[0]
        mixed.extend(_load(fallback, missing, seed_offset))

    random.seed(seed)
    random.shuffle(mixed)

    if dump_bucket_debug:
        print("[MixCalib] Dataset counts:", counts)
        print("[MixCalib] Total samples:", len(mixed))

    return mixed



def get_wikitext2(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    try:
        traindata = load_dataset(
            'ptb_text_only',
            'penn_treebank',
            split='train',
            cache_dir=dataset_cache_dir,
            trust_remote_code=True,
        )
        valdata = load_dataset(
            'ptb_text_only',
            'penn_treebank',
            split='validation',
            cache_dir=dataset_cache_dir,
            trust_remote_code=True,
        )
        train_text = "\n\n".join(traindata['sentence'])
        val_text = "\n\n".join(valdata['sentence'])
    except Exception as e:
        # Fallback to raw PTB URLs if datasets scripts are disabled
        import urllib.request
        import pathlib
        cache_dir = pathlib.Path('cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        urls = {
            'train': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'valid': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
        }
        paths = {k: cache_dir / f'ptb_{k}.txt' for k in urls}
        for k,u in urls.items():
            if not paths[k].exists():
                print(f"[PTB] Falling back to raw URL for {k} split: {u}")
                urllib.request.urlretrieve(u, paths[k])
        with open(paths['train'], 'r', encoding='utf-8') as f:
            train_lines = [ln.strip() for ln in f if ln.strip()]
        with open(paths['valid'], 'r', encoding='utf-8') as f:
            val_lines = [ln.strip() for ln in f if ln.strip()]
        train_text = "\n\n".join(train_lines)
        val_text = "\n\n".join(val_lines)

    trainenc = tokenizer(train_text, return_tensors='pt')
    testenc = tokenizer(val_text, return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, tokenizer):
    # Try local JSON shards; else use HF c4 'en' small slices
    try:
        traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
        valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']
        use_hf = False
    except Exception:
        print("[C4] Falling back to HuggingFace allenai/c4 'en' (train[:2000], validation[:2000]).")
        try:
            traindata = load_dataset("allenai/c4", "en", split="train[:2000]")
            valdata = load_dataset("allenai/c4", "en", split="validation[:2000]")
        except Exception:
            traindata = load_dataset("c4", "en", split="train[:2000]")
            valdata = load_dataset("c4", "en", split="validation[:2000]")
        use_hf = True

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text_i = traindata[i]['text'] if not use_hf else traindata[i]['text']
            trainenc = tokenizer(text_i, return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            text_i = valdata[i]['text'] if not use_hf else valdata[i]['text']
            tmp = tokenizer(text_i, return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc



def get_ptb_new(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    try:
        traindata = load_dataset(
            'ptb_text_only',
            'penn_treebank',
            split='train',
            cache_dir=dataset_cache_dir,
            trust_remote_code=True,
        )
        testdata = load_dataset(
            'ptb_text_only',
            'penn_treebank',
            split='test',
            cache_dir=dataset_cache_dir,
            trust_remote_code=True,
        )
        train_text = " ".join(traindata['sentence'])
        test_text = " ".join(testdata['sentence'])
    except Exception as e:
        # Fallback to raw PTB URLs
        import urllib.request
        import pathlib
        cache_dir = pathlib.Path('cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        urls = {
            'train': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'test': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
        }
        paths = {k: cache_dir / f'ptb_{k}.txt' for k in urls}
        for k,u in urls.items():
            if not paths[k].exists():
                print(f"[PTB] Falling back to raw URL for {k} split: {u}")
                urllib.request.urlretrieve(u, paths[k])
        with open(paths['train'], 'r', encoding='utf-8') as f:
            train_lines = [ln.strip() for ln in f if ln.strip()]
        with open(paths['test'], 'r', encoding='utf-8') as f:
            test_lines = [ln.strip() for ln in f if ln.strip()]
        train_text = " ".join(train_lines)
        test_text = " ".join(test_lines)

    trainenc = tokenizer(train_text, return_tensors='pt')
    testenc = tokenizer(test_text, return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, tokenizer):
    # Same as get_c4 but with a contiguous validation encoding
    try:
        traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
        valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']
        use_hf = False
    except Exception:
        print("[C4] Falling back to HuggingFace allenai/c4 'en' (train[:2000], validation[:2000]).")
        try:
            traindata = load_dataset("allenai/c4", "en", split="train[:2000]")
            valdata = load_dataset("allenai/c4", "en", split="validation[:2000]")
        except Exception:
            traindata = load_dataset("c4", "en", split="train[:2000]")
            valdata = load_dataset("c4", "en", split="validation[:2000]")
        use_hf = True

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text_i = traindata[i]['text'] if not use_hf else traindata[i]['text']
            trainenc = tokenizer(text_i, return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Build a contiguous validation buffer from first ~1100 docs
    if not use_hf:
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    else:
        # HF dataset doesn't support slicing by dict-like [:1100]['text'] directly in this code path
        texts = [valdata[i]['text'] for i in range(min(1100, len(valdata)))]
        valenc = tokenizer(' '.join(texts), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    # Ensure tokenizer is callable (some cached checkpoints store placeholders)
    if not _tokenizer_ok(tokenizer):
        model_hint = os.getenv('SVDLLM_TOKENIZER_MODEL', None)
        if model_hint is None:
            model_hint = 'openlm-research/open_llama_7b'
        hf_token = (
            os.getenv('HF_TOKEN')
            or os.getenv('HUGGINGFACE_TOKEN')
            or os.getenv('HUGGINGFACE_HUB_TOKEN')
        )
        tokenizer = _load_tokenizer_from_hint(model_hint, hf_token=hf_token)
    if not _tokenizer_ok(tokenizer):
        raise TypeError("Tokenizer object is not callable and could not be reconstructed; set SVDLLM_TOKENIZER_MODEL or pass a valid tokenizer.")
    # Local corpus adapter (mirrors get_calib_train_data's spec handling):
    #   - jsonl:/path/to/file.jsonl
    #   - file:/path (auto-detect by extension)
    #   - local_text:/path/to/file.txt
    # These return a list of (inp, tar) windows like the standard loaders.
    try:
        spec = str(name or "").strip()
    except Exception:
        spec = ""
    spec_l = spec.lower()
    if spec_l.startswith(
        (
            "jsonl:",
            "jsonl=",
            "file:",
            "file=",
            "local:",
            "local=",
            "local_text:",
            "local_text=",
            "local_txt:",
            "local_txt=",
            "local_jsonl:",
            "local_jsonl=",
            "local_json:",
            "local_json=",
            "json:",
            "json=",
        )
    ) or (os.path.isfile(spec) and os.path.splitext(spec)[1].lower() in (".txt", ".jsonl", ".json")):
        local_spec = spec
        if os.path.isfile(spec) and not spec_l.startswith(("jsonl:", "jsonl=", "file:", "file=", "local:", "local=")):
            local_spec = f"file:{spec}"
        batches = get_calib_train_data(
            local_spec,
            tokenizer,
            int(nsamples),
            seqlen=int(seqlen),
            seed=int(seed),
            batch_size=1,
        )
        trainloader = []
        for b in batches:
            try:
                inp = b["input_ids"]
            except Exception:
                continue
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader, None

    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, tokenizer)
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, tokenizer)
        return get_c4(nsamples, seed, seqlen, tokenizer)



def get_test_data(name, tokenizer, seq_len=2048, batch_size = 4):
    """
    Build a DataLoader over tokenized evaluation windows for a given dataset name.
    Be robust to environments where `tokenizer` is not a callable HF tokenizer
    (e.g., older pickled checkpoints may store a placeholder). In that case,
    reconstruct a usable tokenizer from an env hint or a generic LLaMA tokenizer.
    """
    # Normalize dataset name
    name = str(name).lower().strip()
    if name in ("wikitext", "wikitext2", "wiki2"):
        name = "wikitext2"
    # Ensure dataset loader is available in this scope
    from datasets import load_dataset
    # Ensure we have a callable tokenizer
    if not _tokenizer_ok(tokenizer):
        model_hint = os.getenv('SVDLLM_TOKENIZER_MODEL', None)
        if model_hint is None:
            model_hint = 'openlm-research/open_llama_7b'
        hf_token = (
            os.getenv('HF_TOKEN')
            or os.getenv('HUGGINGFACE_TOKEN')
            or os.getenv('HUGGINGFACE_HUB_TOKEN')
        )
        tokenizer = _load_tokenizer_from_hint(model_hint, hf_token=hf_token)
    if not _tokenizer_ok(tokenizer):
        # As a last resort, raise a clear error
        raise TypeError("Tokenizer object is not callable and could not be reconstructed; set SVDLLM_TOKENIZER_MODEL or pass a valid tokenizer.")
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    if 'wikitext2_val' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        try:
            test_data = load_dataset(
                'ptb_text_only',
                'penn_treebank',
                split='test',
                trust_remote_code=True,
            )
            test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
        except Exception as e:
            # Fallback: fetch canonical PTB test split text if dataset scripts are unsupported
            # Avoid extra deps by using urllib
            import urllib.request
            import pathlib
            cache_dir = pathlib.Path('cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            ptb_test_path = cache_dir / 'ptb_test.txt'
            if not ptb_test_path.exists():
                try:
                    url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
                    print(f"[PTB] Falling back to raw URL: {url}")
                    urllib.request.urlretrieve(url, ptb_test_path)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load PTB test split via datasets and fallback download: {e}; {e2}")
            with open(ptb_test_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # Build a minimal samples-like mapping expected by process_data
            samples = {'sentence': lines}
            test_dataset = process_data(samples, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        try:
            c4_docs = int(os.getenv("SVDLLM_C4_VAL_DOCS", "2000"))
        except Exception:
            c4_docs = 2000
        if c4_docs <= 0:
            c4_docs = 2000
        # Optional overrides to avoid large downloads
        c4_dataset = os.getenv("SVDLLM_C4_VAL_DATASET", "").strip() or "allenai/c4"
        c4_stream = os.getenv("SVDLLM_C4_VAL_STREAM", "").strip() not in ("", "0", "false", "False", "no", "NO")
        if c4_stream:
            try:
                from datasets import load_dataset
                import itertools
                # Use streaming to avoid full shard downloads
                if c4_dataset in ("allenai/c4", "c4"):
                    stream = load_dataset(c4_dataset, "en", split="validation", streaming=True)
                else:
                    # For small curated sets that require scripts, fall back to official C4 streaming
                    try:
                        stream = load_dataset(c4_dataset, split="train", streaming=True)
                    except Exception:
                        c4_dataset = "allenai/c4"
                        stream = load_dataset(c4_dataset, "en", split="validation", streaming=True)
                texts = []
                for ex in itertools.islice(iter(stream), int(c4_docs)):
                    t = ex.get("text") or ex.get("content") or ""
                    if t:
                        texts.append(t)
                samples = {"text": texts}
                test_dataset = process_data(samples, tokenizer, seq_len, "text")
            except Exception:
                c4_stream = False
        if not c4_stream:
            try:
                test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
                test_dataset = process_data(test_data[0:c4_docs], tokenizer, seq_len, 'text')
            except FileNotFoundError:
                # Fallback to HF C4 validation subset if local file is missing
                try:
                    if c4_dataset in ("allenai/c4", "c4"):
                        test_data = load_dataset(c4_dataset, "en", split=f"validation[:{c4_docs}]")
                    else:
                        test_data = load_dataset(c4_dataset, split=f"train[:{c4_docs}]")
                    test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
                except Exception:
                    # Final fallback: stream official C4 validation
                    try:
                        from datasets import load_dataset as _ld
                        import itertools as _it
                        stream = _ld("allenai/c4", "en", split="validation", streaming=True)
                        texts = []
                        for ex in _it.islice(iter(stream), int(c4_docs)):
                            t = ex.get("text") or ex.get("content") or ""
                            if t:
                                texts.append(t)
                        samples = {"text": texts}
                        test_dataset = process_data(samples, tokenizer, seq_len, "text")
                    except Exception:
                        raise
    else:
        raise ValueError(f"Unknown dataset name: {name}. Supported: wikitext2, wikitext2_val, ptb, c4.")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
