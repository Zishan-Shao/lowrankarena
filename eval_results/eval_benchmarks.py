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

import torch
from torch.nn import functional as F
from datasets import load_dataset
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
        base = getattr(args, "model", None) or "model"
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
from utils.df_svd_loader import looks_like_dfsvd_checkpoint, load_dfsvd_model
from evaluater import ppl_eval

# ----------------------------------------------------------------------------
# Dobi-SVD loader (mirrors eval_general_ppl.py)
# ----------------------------------------------------------------------------

def _looks_like_dobi_svd_model(model_id: str) -> bool:
    """Heuristic: detect a Dobi-SVD checkpoint to route loading to baselines/Dobi-SVD."""
    if not model_id:
        return False
    s = str(model_id).lower()
    if "dobisvd" in s or "dobi-svd" in s:
        return True
    if os.path.isdir(model_id):
        for fn in ("remapping_weight.pt", "DobiSVD_Model.pt"):
            if os.path.exists(os.path.join(model_id, fn)):
                return True
    return False


def _resolve_dobi_path(
    model_id: str,
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Return local directory for a Dobi-SVD model id (HF repo or local dir)."""
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        raise ImportError(
            "huggingface_hub is required to load Dobi-SVD HF repos. Please `pip install huggingface_hub`."
        ) from e
    return snapshot_download(repo_id=model_id, token=hf_token, revision=revision, cache_dir=cache_dir)


def _load_dobi_model(
    model_id: str,
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    remapping: Optional[bool] = None,
):
    """Load Dobi-SVD model via the official loader functions.

    remapping:
      - True  -> load_remapping_model (keeps remapped/packed weights; avoids dequantization)
      - False -> load_unremapping_model (unremap/dequantize to regular weights)
      - None  -> auto-detect based on files in the checkpoint dir.
    """
    dobi_root = os.path.join(_REPO_ROOT, "baselines", "Dobi-SVD")
    if dobi_root not in sys.path:
        sys.path.insert(0, dobi_root)
    try:
        from modelutils import load_remapping_model, load_unremapping_model  # type: ignore
    except Exception as e:
        raise ImportError(
            f"Failed to import Dobi-SVD loader from {dobi_root}. "
            "Make sure baselines/Dobi-SVD is present and its dependencies are installed."
        ) from e

    local_path = _resolve_dobi_path(model_id, hf_token=hf_token, revision=revision, cache_dir=cache_dir)

    if remapping is None:
        if os.path.exists(os.path.join(local_path, "remapping_weight.pt")):
            remapping = True
        elif os.path.exists(os.path.join(local_path, "DobiSVD_Model.pt")):
            remapping = False
        else:
            raise FileNotFoundError(
                f"Could not infer Dobi-SVD mode from checkpoint dir: {local_path}. "
                "Expected remapping_weight.pt (remapping) or DobiSVD_Model.pt (unremapping). "
                "If this is a standard HF model, do NOT use the Dobi loader."
            )

    loader = load_remapping_model if remapping else load_unremapping_model
    model, tokenizer = loader(local_path)
    return model, tokenizer, local_path, bool(remapping)


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


def _tokenizer_ok(tokenizer) -> bool:
    try:
        return tokenizer is not None and not isinstance(tokenizer, bool) and callable(tokenizer)
    except Exception:
        return False


def _load_tokenizer_from_hint(model_hint: str, hf_token: Optional[str] = None):
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
                tok = cls.from_pretrained(model_hint, token=hf_token)
                if _tokenizer_ok(tok):
                    return tok
            except Exception:
                continue
    except Exception:
        pass
    return None


def _ensure_tokenizer_compat(model, tokenizer, hf_token: Optional[str] = None):
    override = os.getenv("SVDLLM_TOKENIZER_MODEL", "").strip()
    model_hint = None
    try:
        model_hint = getattr(getattr(model, "config", None), "_name_or_path", None)
    except Exception:
        model_hint = None
    # Prefer explicit override
    if override:
        tok = _load_tokenizer_from_hint(override, hf_token=hf_token)
        if tok is not None:
            tokenizer = tok
    # Repair invalid tokenizer
    if not _tokenizer_ok(tokenizer):
        hint = override or model_hint
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
        hint = override or model_hint
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
    ap.add_argument('--model', type=str, required=True, help='HF model id or path to local .pt checkpoint saved by this repo')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--hf_token', type=str, default=None)
    ap.add_argument('--saes_svd', action='store_true',
                    help='Treat --model as SAES-SVD checkpoint (local dir or HF repo id)')
    ap.add_argument('--saes_base_model', type=str, default=None,
                    help='Base HF model id/path to apply SAES-SVD onto (required when using SAES-SVD).')
    ap.add_argument('--saes_revision', type=str, default=None)
    ap.add_argument('--saes_cache_dir', type=str, default=None)
    ap.add_argument('--dfsvd_svd', action='store_true',
                    help='Treat --model as DF-SVD checkpoint (local dir or HF repo id)')
    ap.add_argument('--dfsvd_base_model', type=str, default=None,
                    help='Base HF model id/path to apply DF-SVD onto (required when using DF-SVD).')
    ap.add_argument('--dfsvd_revision', type=str, default=None)
    ap.add_argument('--dfsvd_cache_dir', type=str, default=None)

    ap.add_argument('--dobi_svd', action='store_true',
                    help='Treat --model as a Dobi-SVD checkpoint (HF repo id or local dir) and load via baselines/Dobi-SVD.')
    ap.add_argument('--dobi_remapping', action='store_true',
                    help='Force Dobi *remapping* loader (keeps remapped/packed weights; avoids dequantization).')
    ap.add_argument('--dobi_unremapping', action='store_true',
                    help='Force Dobi *unremapping* loader (unremap/dequantize to regular weights).')
    ap.add_argument('--dobi_revision', type=str, default=None)
    ap.add_argument('--dobi_cache_dir', type=str, default=None)

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

    args = ap.parse_args()

    # Load model
    if args.dobi_remapping and args.dobi_unremapping:
        raise ValueError("Only one of --dobi_remapping / --dobi_unremapping can be set.")
    use_dobi = bool(args.dobi_svd or args.dobi_remapping or args.dobi_unremapping or _looks_like_dobi_svd_model(args.model))

    if use_dobi:
        remap_flag = True if args.dobi_remapping else (False if args.dobi_unremapping else None)
        model, tokenizer, _, is_dobi_remapping = _load_dobi_model(
            args.model,
            hf_token=args.hf_token,
            revision=args.dobi_revision,
            cache_dir=args.dobi_cache_dir,
            remapping=remap_flag,
        )
    elif os.path.exists(args.model) and args.model.endswith('.pt'):
        model, tokenizer = get_model_from_local(args.model)
        is_dobi_remapping = False
    elif (os.path.isdir(args.model) and looks_like_dfsvd_checkpoint(args.model)) or args.dfsvd_svd:
        if not args.dfsvd_base_model:
            raise ValueError("DF-SVD requires --dfsvd_base_model.")
        model, tokenizer, _ = load_dfsvd_model(
            args.model,
            base_model=args.dfsvd_base_model,
            hf_token=args.hf_token,
            revision=args.dfsvd_revision,
            cache_dir=args.dfsvd_cache_dir,
        )
        is_dobi_remapping = False
    elif (os.path.isdir(args.model) and looks_like_saes_svd_checkpoint(args.model)) or args.saes_svd:
        if not args.saes_base_model:
            raise ValueError("SAES-SVD requires --saes_base_model.")
        model, tokenizer, _ = load_saes_svd_model(
            args.model,
            base_model=args.saes_base_model,
            hf_token=args.hf_token,
            revision=args.saes_revision,
            cache_dir=args.saes_cache_dir,
        )
        is_dobi_remapping = False
    else:
        model, tokenizer = get_model_from_huggingface(args.model, hf_token=args.hf_token)
        is_dobi_remapping = False
    tokenizer = _ensure_tokenizer_compat(model, tokenizer, hf_token=args.hf_token)
    if args.force_right_padding:
        _force_right_padding(tokenizer)
    else:
        _ensure_pad_token(tokenizer)
    # NOTE: Dobi remapping checkpoints may store packed/int weights; avoid dtype casting in that case.
    if 'is_dobi_remapping' in locals() and is_dobi_remapping:
        model = model.to(args.device).eval()
    else:
        model = _to_device(model, args.device, args.dtype)
    _set_pad_query_fix_flags(model, fix=args.fix_pad_query_mask)

    # Disable cache for deterministic eval and lower memory
    try:
        model.config.use_cache = False
    except Exception:
        pass

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

        out_json = _auto_output_json(args, 'lm_eval')
        payload = {
            'schema': 'svdllm_eval_v1',
            'script': os.path.basename(__file__),
            'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
            'cmd': ' '.join(sys.argv),
            'mode': 'lm_eval',
            'args': vars(args),
            'model': args.model,
            'results': lm_eval_res.get('results', lm_eval_res) if isinstance(lm_eval_res, dict) else lm_eval_res,
        }
        _write_json(out_json, payload)
        return
    if args.token_ppl:
        ds = [d.strip() for d in args.token_ppl_datasets.split(',') if d.strip()]
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

        out_json = _auto_output_json(args, 'token_ppl')
        payload = {
            'schema': 'svdllm_eval_v1',
            'script': os.path.basename(__file__),
            'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
            'cmd': ' '.join(sys.argv),
            'mode': 'token_ppl',
            'args': vars(args),
            'model': args.model,
            'results': token_ppl,
        }
        _write_json(out_json, payload)
        return

    # Evaluate tasks
    results: Dict[str, float] = {}
    results['Openb.'] = eval_openbookqa(model, tokenizer, args.device, args.batch_size, args.limit)
    results['ARC_e'] = eval_arc_easy(model, tokenizer, args.device, args.batch_size, args.limit)
    results['WinoG.'] = eval_winogrande(model, tokenizer, args.device, args.batch_size, args.limit)
    results['HellaS.'] = eval_hellaswag(model, tokenizer, args.device, args.batch_size, args.limit)
    results['PIQA'] = eval_piqa(model, tokenizer, args.device, args.batch_size, args.limit)
    results['MathQA'] = eval_mathqa(model, tokenizer, args.device, args.batch_size, args.limit)
    # Average over the six MC tasks
    mc_keys = ['Openb.', 'ARC_e', 'WinoG.', 'HellaS.', 'PIQA', 'MathQA']
    mc_vals = [v for k, v in results.items() if k in mc_keys and isinstance(v, float)]
    results['Average'] = sum(mc_vals) / len(mc_vals) if mc_vals else float('nan')
    # TruthfulQA MC1
    if not args.skip_truthfulqa:
        results['TruthfulQA'] = eval_truthfulqa_mc1(model, tokenizer, args.device, args.batch_size, args.limit)
    else:
        results['TruthfulQA'] = float('nan')
    # GSM8K
    if not args.skip_gsm8k:
        results['GSM8K'] = eval_gsm8k(
            model,
            tokenizer,
            args.device,
            args.limit,
            max_new_tokens=args.gsm8k_max_new_tokens,
            batch_size=max(1, args.batch_size // 2),
        )
    else:
        results['GSM8K'] = float('nan')

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
    payload = {
        'schema': 'svdllm_eval_v1',
        'script': os.path.basename(__file__),
        'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
        'cmd': ' '.join(sys.argv),
        'mode': 'benchmark',
        'args': vars(args),
        'model': args.model,
        'results': results,
    }
    _write_json(out_json, payload)


if __name__ == '__main__':
    main()
