import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F
from datasets import load_dataset
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None
from tqdm import tqdm

# Ensure repo root on PYTHONPATH
# Ensure repo root is on PYTHONPATH
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.model_utils import get_model_from_huggingface, get_model_from_local

# Optional local dataset loaders (piqa/mathqa)
load_piqa_local = None
load_mathqa_local = None
try:
    from datasets.load_data import load_piqa_local as _lp_local, load_mathqa_local as _lm_local  # type: ignore
    load_piqa_local = _lp_local
    load_mathqa_local = _lm_local
except Exception:
    try:
        import importlib.util as _ilu
        _base = os.path.join(_THIS_DIR, 'datasets', 'load_data.py')
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

# Default padding side used during scoring; can be overridden by args
PADDING_SIDE = "left"


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    try:
        if PADDING_SIDE:
            tokenizer.padding_side = PADDING_SIDE
    except Exception:
        pass


def _force_right_padding(tokenizer):
    global PADDING_SIDE
    PADDING_SIDE = "right"
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})


def _set_padding_side(tokenizer, side: Optional[str]) -> None:
    global PADDING_SIDE
    if side:
        PADDING_SIDE = side
        try:
            tokenizer.padding_side = side
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
    if override:
        tok = _load_tokenizer_from_hint(override, hf_token=hf_token)
        if tok is not None:
            tokenizer = tok
    if not _tokenizer_ok(tokenizer):
        hint = override or model_hint
        if hint:
            tok = _load_tokenizer_from_hint(hint, hf_token=hf_token)
            if tok is not None:
                tokenizer = tok
    if not _tokenizer_ok(tokenizer):
        raise TypeError(
            "Tokenizer object is not callable and could not be reconstructed; "
            "set SVDLLM_TOKENIZER_MODEL to a valid local tokenizer."
        )
    return tokenizer


def _apply_fix_pad_query_mask(model, enabled: bool) -> None:
    for module in model.modules():
        if module.__class__.__name__ == "SVD_LlamaAttention":
            module.fix_pad_query_mask = enabled


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


def _letters(n: int) -> List[str]:
    base = [chr(ord('A') + i) for i in range(26)]
    if n <= len(base):
        return base[:n]
    # Fallback: repeat letters with suffix
    out = []
    k = 0
    while len(out) < n:
        out.append(base[k % 26] + str(k // 26))
        k += 1
    return out


def _build_prompt_from_cfg(
    question: str,
    choices: List[str],
    cfg: Dict[str, Any],
) -> Tuple[str, List[str]]:
    lines: List[str] = []
    preface = cfg.get("preface")
    if preface:
        lines.append(str(preface))
    q_label = cfg.get("q_label")
    if q_label:
        lines.append(f"{q_label} {question}")
    else:
        lines.append(str(question))
    include_options = bool(cfg.get("include_options", False))
    use_letters = bool(cfg.get("use_letters", False))
    out_choices = choices
    if include_options:
        options_label = cfg.get("options_label", "Options:")
        if options_label:
            lines.append(str(options_label))
        if use_letters:
            letters = _letters(len(choices))
            lines.extend([f"{letters[i]}) {choices[i]}" for i in range(len(choices))])
            out_choices = letters
        else:
            lines.extend([f"- {c}" for c in choices])
    answer_label = cfg.get("answer_label")
    if answer_label:
        lines.append(str(answer_label))
    prompt = "\n".join(lines)
    return prompt, out_choices


def _strip_prompt(prompt: str) -> str:
    text = str(prompt).strip()
    text = re.sub(r"\n?\s*Answer:\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\s*Question:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def _normalize_local_items(local_items: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for ex in local_items:
        q = ex.get('question')
        if not q and ex.get('prompt'):
            q = _strip_prompt(ex.get('prompt'))
        choices = ex.get('choices') or ex.get('options')
        if not choices and ex.get('sol1') is not None and ex.get('sol2') is not None:
            choices = [ex.get('sol1'), ex.get('sol2')]
        ans = ex.get('answer_idx')
        if ans is None:
            lab = ex.get('label') or ex.get('answer') or ex.get('correct')
            if lab is not None:
                try:
                    ans = int(lab)
                except Exception:
                    lab_s = str(lab).strip().upper()
                    if lab_s and 'A' <= lab_s[0] <= 'E':
                        ans = ord(lab_s[0]) - ord('A')
        if q and choices:
            try:
                ans_i = int(ans) if ans is not None else 0
            except Exception:
                ans_i = 0
            norm.append({'task': task, 'question': str(q).strip(), 'choices': list(choices), 'answer_idx': ans_i})
    return norm


def _tmpl_plain(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    q = item['question']
    if item['task'] == 'winogrande':
        if '_' in q:
            left = q.split('_', 1)[0]
        else:
            left = q + ' '
        return left, item['choices']
    return q, item['choices']


def _tmpl_qa(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    q = item['question']
    return f"Question: {q}\nAnswer:", item['choices']


def _tmpl_mc_letters(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    q = item['question']
    letters = _letters(len(item['choices']))
    opts = [f"{letters[i]}) {item['choices'][i]}" for i in range(len(item['choices']))]
    prompt = f"Question: {q}\nOptions: " + " \n".join(opts) + "\nAnswer:"
    return prompt, letters


def _tmpl_instruction(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    q = item['question']
    letters = _letters(len(item['choices']))
    opts = [f"{letters[i]}) {item['choices'][i]}" for i in range(len(item['choices']))]
    prompt = "Choose the correct answer.\n" + f"Question: {q}\nOptions: " + " \n".join(opts) + "\nAnswer:"
    return prompt, letters


_TEMPLATE_FALLBACKS = {
    "plain": _tmpl_plain,
    "qa": _tmpl_qa,
    "mc_letters": _tmpl_mc_letters,
    "instruction": _tmpl_instruction,
}

# Task-specific templates (intentionally distinct from lm-eval prompt strings)
DEFAULT_TEMPLATE_PROFILE = "realistic"
TEMPLATE_PROFILE = DEFAULT_TEMPLATE_PROFILE

_REALISTIC_ARC_CFG = {
    "plain": {
        "preface": "Task: ARC",
        "q_label": "Question:",
        "include_options": False,
        "answer_label": "Answer:",
    },
    "qa": {
        "preface": "Task: ARC",
        "q_label": "Question:",
        "include_options": True,
        "options_label": "Options:",
        "answer_label": "Answer:",
    },
    "mc_letters": {
        "preface": "Task: ARC (multiple choice)",
        "q_label": "Question:",
        "include_options": True,
        "use_letters": True,
        "options_label": "Options:",
        "answer_label": "Answer (letter):",
    },
    "instruction": {
        "preface": "Please select the correct option.",
        "q_label": "Question:",
        "include_options": True,
        "options_label": "Options:",
        "answer_label": "Final answer:",
    },
}

_REBUTTAL_ARC_CFG = {
    "plain": {
        "preface": "ARC Science",
        "q_label": "Science question:",
        "include_options": False,
        "answer_label": "Answer:",
    },
    "qa": {
        "preface": "ARC Science",
        "q_label": "Science question:",
        "include_options": True,
        "options_label": "Choices:",
        "answer_label": "Best choice:",
    },
    "mc_letters": {
        "preface": "ARC Science (multiple choice)",
        "q_label": "Science question:",
        "include_options": True,
        "use_letters": True,
        "options_label": "Choices:",
        "answer_label": "Choose the letter:",
    },
    "instruction": {
        "preface": "Instruction: choose the best answer to the science question.",
        "q_label": "Science question:",
        "include_options": True,
        "options_label": "Choices:",
        "answer_label": "Final answer:",
    },
}

TASK_TEMPLATE_CONFIGS: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "realistic": {
        "openbookqa": {
            "plain": {
                "preface": "Task: OpenBookQA",
                "q_label": "Question:",
                "include_options": False,
                "answer_label": "Answer:",
            },
            "qa": {
                "preface": "Task: OpenBookQA",
                "q_label": "Question:",
                "include_options": True,
                "options_label": "Options:",
                "answer_label": "Answer:",
            },
            "mc_letters": {
                "preface": "Task: OpenBookQA (multiple choice)",
                "q_label": "Question:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Options:",
                "answer_label": "Answer (letter):",
            },
            "instruction": {
                "preface": "Please choose the best option for this question.",
                "q_label": "Question:",
                "include_options": True,
                "options_label": "Options:",
                "answer_label": "Final answer:",
            },
        },
        "arc_easy": {k: {**v, "preface": "Task: ARC-Easy"} for k, v in _REALISTIC_ARC_CFG.items()},
        "arc_challenge": {
            k: {**v, "preface": "Task: ARC-Challenge"} for k, v in _REALISTIC_ARC_CFG.items()
        },
        "winogrande": {
            "plain": {
                "preface": "Task: WinoGrande (fill in the blank)",
                "q_label": "Sentence:",
                "include_options": False,
                "answer_label": "Blank:",
            },
            "qa": {
                "preface": "Task: WinoGrande",
                "q_label": "Sentence:",
                "include_options": True,
                "options_label": "Choices:",
                "answer_label": "Answer:",
            },
            "mc_letters": {
                "preface": "Task: WinoGrande (multiple choice)",
                "q_label": "Sentence:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Choices:",
                "answer_label": "Answer (letter):",
            },
            "instruction": {
                "preface": "Choose the word that best fills the blank.",
                "q_label": "Sentence:",
                "include_options": True,
                "options_label": "Choices:",
                "answer_label": "Final answer:",
            },
        },
        "hellaswag": {
            "plain": {
                "preface": "Task: HellaSwag",
                "q_label": "Context:",
                "include_options": False,
                "answer_label": "Continuation:",
            },
            "qa": {
                "preface": "Task: HellaSwag",
                "q_label": "Context:",
                "include_options": True,
                "options_label": "Endings:",
                "answer_label": "Best ending:",
            },
            "mc_letters": {
                "preface": "Task: HellaSwag (multiple choice)",
                "q_label": "Context:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Endings:",
                "answer_label": "Answer (letter):",
            },
            "instruction": {
                "preface": "Pick the most plausible continuation.",
                "q_label": "Context:",
                "include_options": True,
                "options_label": "Endings:",
                "answer_label": "Final answer:",
            },
        },
        "piqa": {
            "plain": {
                "preface": "Task: PIQA",
                "q_label": "Goal:",
                "include_options": False,
                "answer_label": "Better solution:",
            },
            "qa": {
                "preface": "Task: PIQA",
                "q_label": "Goal:",
                "include_options": True,
                "options_label": "Solutions:",
                "answer_label": "Best solution:",
            },
            "mc_letters": {
                "preface": "Task: PIQA (multiple choice)",
                "q_label": "Goal:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Solutions:",
                "answer_label": "Answer (letter):",
            },
            "instruction": {
                "preface": "Choose the most appropriate solution.",
                "q_label": "Goal:",
                "include_options": True,
                "options_label": "Solutions:",
                "answer_label": "Final answer:",
            },
        },
        "mathqa": {
            "plain": {
                "preface": "Task: MathQA",
                "q_label": "Problem:",
                "include_options": False,
                "answer_label": "Answer:",
            },
            "qa": {
                "preface": "Task: MathQA",
                "q_label": "Problem:",
                "include_options": True,
                "options_label": "Choices:",
                "answer_label": "Answer:",
            },
            "mc_letters": {
                "preface": "Task: MathQA (multiple choice)",
                "q_label": "Problem:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Choices:",
                "answer_label": "Answer (letter):",
            },
            "instruction": {
                "preface": "Compute the answer and select the correct choice.",
                "q_label": "Problem:",
                "include_options": True,
                "options_label": "Choices:",
                "answer_label": "Final answer:",
            },
        },
    },
    "rebuttal": {
        "openbookqa": {
            "plain": {
                "preface": "OpenBookQA",
                "q_label": "Q:",
                "include_options": False,
                "answer_label": "Answer:",
            },
            "qa": {
                "preface": "OpenBookQA",
                "q_label": "Q:",
                "include_options": True,
                "options_label": "Options:",
                "answer_label": "Answer:",
            },
            "mc_letters": {
                "preface": "OpenBookQA (multiple choice)",
                "q_label": "Q:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Options:",
                "answer_label": "Choose the letter:",
            },
            "instruction": {
                "preface": "Instruction: choose the best answer for this OpenBookQA item.",
                "q_label": "Q:",
                "include_options": True,
                "options_label": "Options:",
                "answer_label": "Final answer:",
            },
        },
        "arc_easy": {k: {**v, "preface": "ARC-Easy"} for k, v in _REBUTTAL_ARC_CFG.items()},
        "arc_challenge": {
            k: {**v, "preface": "ARC-Challenge"} for k, v in _REBUTTAL_ARC_CFG.items()
        },
        "winogrande": {
            "plain": {
                "preface": "WinoGrande (fill the blank)",
                "q_label": "Sentence:",
                "include_options": False,
                "answer_label": "Blank:",
            },
            "qa": {
                "preface": "WinoGrande",
                "q_label": "Sentence:",
                "include_options": True,
                "options_label": "Choices:",
                "answer_label": "Answer:",
            },
            "mc_letters": {
                "preface": "WinoGrande (multiple choice)",
                "q_label": "Sentence:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Choices:",
                "answer_label": "Letter:",
            },
            "instruction": {
                "preface": "Instruction: choose the word that best fills the blank.",
                "q_label": "Sentence:",
                "include_options": True,
                "options_label": "Choices:",
                "answer_label": "Final answer:",
            },
        },
        "hellaswag": {
            "plain": {
                "preface": "HellaSwag",
                "q_label": "Context:",
                "include_options": False,
                "answer_label": "Continuation:",
            },
            "qa": {
                "preface": "HellaSwag",
                "q_label": "Context:",
                "include_options": True,
                "options_label": "Endings:",
                "answer_label": "Best ending:",
            },
            "mc_letters": {
                "preface": "HellaSwag (multiple choice)",
                "q_label": "Context:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Endings:",
                "answer_label": "Letter:",
            },
            "instruction": {
                "preface": "Instruction: pick the most plausible continuation.",
                "q_label": "Context:",
                "include_options": True,
                "options_label": "Endings:",
                "answer_label": "Final answer:",
            },
        },
        "piqa": {
            "plain": {
                "preface": "PIQA",
                "q_label": "Goal:",
                "include_options": False,
                "answer_label": "Better solution:",
            },
            "qa": {
                "preface": "PIQA",
                "q_label": "Goal:",
                "include_options": True,
                "options_label": "Solutions:",
                "answer_label": "Best solution:",
            },
            "mc_letters": {
                "preface": "PIQA (multiple choice)",
                "q_label": "Goal:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Solutions:",
                "answer_label": "Letter:",
            },
            "instruction": {
                "preface": "Instruction: choose the most appropriate solution.",
                "q_label": "Goal:",
                "include_options": True,
                "options_label": "Solutions:",
                "answer_label": "Final answer:",
            },
        },
        "mathqa": {
            "plain": {
                "preface": "MathQA",
                "q_label": "Problem:",
                "include_options": False,
                "answer_label": "Answer:",
            },
            "qa": {
                "preface": "MathQA",
                "q_label": "Problem:",
                "include_options": True,
                "options_label": "Choices:",
                "answer_label": "Best choice:",
            },
            "mc_letters": {
                "preface": "MathQA (multiple choice)",
                "q_label": "Problem:",
                "include_options": True,
                "use_letters": True,
                "options_label": "Choices:",
                "answer_label": "Letter:",
            },
            "instruction": {
                "preface": "Instruction: compute the answer and select the correct choice.",
                "q_label": "Problem:",
                "include_options": True,
                "options_label": "Choices:",
                "answer_label": "Final answer:",
            },
        },
    },
}


def _tmpl_task_specific(item: Dict[str, Any], tmpl_name: str) -> Tuple[str, List[str]]:
    task = item.get("task", "")
    prof_cfg = TASK_TEMPLATE_CONFIGS.get(TEMPLATE_PROFILE) or TASK_TEMPLATE_CONFIGS.get(DEFAULT_TEMPLATE_PROFILE, {})
    cfg = prof_cfg.get(task, {}).get(tmpl_name)
    if cfg is None:
        return _TEMPLATE_FALLBACKS[tmpl_name](item)
    q = item.get("question", "")
    return _build_prompt_from_cfg(q, item['choices'], cfg)


def _make_task_template_fn(tmpl_name: str):
    def _fn(item: Dict[str, Any]) -> Tuple[str, List[str]]:
        return _tmpl_task_specific(item, tmpl_name)
    return _fn


TEMPLATES = {
    "plain": _make_task_template_fn("plain"),
    "qa": _make_task_template_fn("qa"),
    "mc_letters": _make_task_template_fn("mc_letters"),
    "instruction": _make_task_template_fn("instruction"),
}


def _score_mc_batch(
    model,
    tokenizer,
    batch_samples: List[Dict[str, Any]],
    device: str,
) -> Tuple[List[int], List[int]]:
    _ensure_pad_token(tokenizer)
    input_ids_list = []
    labels_list = []
    group_sizes = []
    for s in batch_samples:
        prompt_ids = tokenizer.encode(s['prompt'], add_special_tokens=False)
        k = 0
        for ch in s['choices']:
            ans_ids = tokenizer.encode(" " + ch, add_special_tokens=False)
            ids = prompt_ids + ans_ids
            labels = [-100] * len(prompt_ids) + ans_ids
            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            k += 1
        group_sizes.append(k)
    max_len = max(x.size(0) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    for i, (ids, lbs) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : ids.size(0)] = ids
        labels[i, : lbs.size(0)] = lbs
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    out = model(input_ids=input_ids, use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
    logits = out.logits
    logprobs = F.log_softmax(logits, dim=-1)
    lp = logprobs[:, :-1, :]
    lbl = labels[:, 1:]
    mask = (lbl != -100)
    lbl_clamped = torch.where(mask, lbl, torch.zeros_like(lbl))
    token_lp = lp.gather(dim=-1, index=lbl_clamped.unsqueeze(-1)).squeeze(-1)
    token_lp = torch.where(mask, token_lp, torch.zeros_like(token_lp))
    seq_scores_sum = token_lp.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1)
    seq_scores_avg = seq_scores_sum / lengths
    preds_sum = []
    preds_avg = []
    idx = 0
    for g in group_sizes:
        group_sum = seq_scores_sum[idx: idx + g]
        group_avg = seq_scores_avg[idx: idx + g]
        preds_sum.append(int(torch.argmax(group_sum).item()))
        preds_avg.append(int(torch.argmax(group_avg).item()))
        idx += g
    return preds_sum, preds_avg


def _accuracy(preds: List[int], golds: List[int]) -> float:
    if not preds:
        return float('nan')
    correct = sum(int(p == g) for p, g in zip(preds, golds))
    return 100.0 * correct / len(preds)


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float('nan'), float('nan')
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def _parse_mathqa_options(opt_str: str) -> Tuple[List[str], Dict[str, int]]:
    choices, mapping = [], {}
    import re
    parts = re.split(r"\s*([A-Ea-e])\)\s*", opt_str or "")
    for i in range(1, len(parts), 2):
        lab = parts[i].upper()
        text = parts[i + 1].strip()
        mapping[lab] = len(choices)
        choices.append(text)
    return choices, mapping


def _load_openbookqa() -> List[Dict[str, Any]]:
    ds_all = load_dataset('openbookqa', 'main')
    split = 'validation' if 'validation' in ds_all else ('test' if 'test' in ds_all else 'train')
    ds = ds_all[split]
    items = []
    for ex in ds:
        stem = ex.get('question_stem') or ex.get('question') or ''
        ch_texts = ex['choices']['text'] if isinstance(ex.get('choices'), dict) else ex.get('choices', [])
        ch_labels = ex['choices']['label'] if isinstance(ex.get('choices'), dict) else None
        key = ex.get('answerKey')
        idx = 0
        if key is not None and ch_labels is not None:
            try:
                idx = list(ch_labels).index(key)
            except Exception:
                idx = 0
        elif isinstance(ex.get('label'), int):
            idx = int(ex['label'])
        items.append({'task': 'openbookqa', 'question': stem, 'choices': list(ch_texts), 'answer_idx': idx})
    return items


def _load_arc(split_name: str) -> List[Dict[str, Any]]:
    ds_all = load_dataset('ai2_arc', split_name)
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
                ch_texts = list(ch)
                ch_labels = [None] * len(ch_texts)
        idx = 0
        if key is not None and ch_labels:
            try:
                idx = list(ch_labels).index(key)
            except Exception:
                idx = 0
        items.append({'task': split_name.lower(), 'question': q, 'choices': ch_texts, 'answer_idx': idx})
    return items


def _load_winogrande() -> List[Dict[str, Any]]:
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
        items.append({'task': 'winogrande', 'question': sent, 'choices': [opt1, opt2], 'answer_idx': ans_idx})
    return items


def _load_hellaswag() -> List[Dict[str, Any]]:
    ds_all = load_dataset('hellaswag')
    split = 'validation' if 'validation' in ds_all else 'test'
    ds = ds_all[split]
    items = []
    for ex in ds:
        ctx = ex.get('ctx', None) or (ex.get('ctx_a', '') + ' ' + ex.get('ctx_b', ''))
        endings = ex.get('endings') or ex.get('ending_options') or []
        label = ex.get('label', 0)
        items.append({'task': 'hellaswag', 'question': ctx.strip(), 'choices': list(endings), 'answer_idx': int(label)})
    return items


def _load_piqa() -> List[Dict[str, Any]]:
    if load_piqa_local is not None:
        try:
            local_items = load_piqa_local(split='validation')
            if local_items:
                norm = _normalize_local_items(local_items, 'piqa')
                if norm:
                    return norm
        except Exception:
            pass
    ds = None
    try:
        ds_all = load_dataset('piqa')
        split = 'validation' if 'validation' in ds_all else ('test' if 'test' in ds_all else 'train')
        ds = ds_all[split]
    except Exception:
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
            return []
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
        items.append({'task': 'piqa', 'question': goal.strip(), 'choices': [sol1, sol2], 'answer_idx': idx})
    return items


def _load_mathqa() -> List[Dict[str, Any]]:
    if load_mathqa_local is not None:
        try:
            local_items = load_mathqa_local(split='validation')
            if local_items:
                norm = _normalize_local_items(local_items, 'mathqa')
                if norm:
                    return norm
        except Exception:
            pass
    ds = None
    try:
        ds_all = load_dataset('math_qa')
        split = 'validation' if 'validation' in ds_all else ('test' if 'test' in ds_all else 'train')
        ds = ds_all[split]
    except Exception:
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
            return []
    items = []
    for ex in ds:
        q = ex.get('Problem') or ex.get('problem') or ex.get('question', '')
        opt = ex.get('options', '')
        correct = ex.get('correct') or ex.get('label') or 'A'
        choices, map_l = _parse_mathqa_options(opt)
        idx = map_l.get(str(correct).upper(), 0)
        items.append({'task': 'mathqa', 'question': q.strip(), 'choices': choices, 'answer_idx': idx})
    return items


TASK_LOADERS = {
    'openbookqa': _load_openbookqa,
    'arc_easy': lambda: _load_arc('ARC-Easy'),
    'arc_challenge': lambda: _load_arc('ARC-Challenge'),
    'winogrande': _load_winogrande,
    'hellaswag': _load_hellaswag,
    'piqa': _load_piqa,
    'mathqa': _load_mathqa,
}


def _eval_task(
    model,
    tokenizer,
    device: str,
    task_name: str,
    template_fn,
    batch_size: int,
    limit: Optional[int],
) -> Dict[str, float]:
    items = TASK_LOADERS[task_name]()
    if limit is not None:
        items = items[:limit]
    samples = []
    for it in items:
        if not it.get('choices'):
            continue
        if not it.get('question') and it.get('prompt'):
            it['question'] = _strip_prompt(it.get('prompt'))
        if not it.get('question'):
            continue
        prompt, choices = template_fn(it)
        samples.append({'prompt': prompt, 'choices': choices, 'answer_idx': it['answer_idx']})
    preds_sum, preds_avg, golds = [], [], []
    total = len(samples)
    if total == 0:
        return {'acc': float('nan'), 'acc_norm': float('nan')}
    n_batches = (total + max(1, batch_size) - 1) // max(1, batch_size)
    pbar = tqdm(total=n_batches, desc=f"{task_name}:{template_fn.__name__}", leave=False)
    i = 0
    while i < total:
        batch = samples[i: i + batch_size]
        ps, pa = _score_mc_batch(model, tokenizer, batch, device)
        preds_sum.extend(ps)
        preds_avg.extend(pa)
        golds.extend([s['answer_idx'] for s in batch])
        i += batch_size
        pbar.update(1)
    pbar.close()
    return {
        'acc': _accuracy(preds_sum, golds),
        'acc_norm': _accuracy(preds_avg, golds),
    }


def _evaluate_templates(
    model,
    tokenizer,
    args,
    task_list: List[str],
    tmpl_list: List[str],
    padding_side: Optional[str],
    fix_pad_query_mask: bool,
    pad_variant: Optional[str] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        'model': args.model,
        'device': args.device,
        'batch_size': args.batch_size,
        'dtype': args.dtype,
        'limit': args.limit,
        'tasks': task_list,
        'template_profile': TEMPLATE_PROFILE,
        'templates': {},
    }
    if padding_side:
        results['padding_side'] = padding_side
    results['fix_pad_query_mask'] = bool(fix_pad_query_mask)
    if pad_variant:
        results['pad_variant'] = pad_variant

    for tmpl_name in tmpl_list:
        tmpl_fn = TEMPLATES[tmpl_name]
        per_task = {}
        accs, acc_norms = [], []
        for task in task_list:
            metrics = _eval_task(model, tokenizer, args.device, task, tmpl_fn, args.batch_size, args.limit)
            per_task[task] = metrics
            if isinstance(metrics.get('acc'), float) and not math.isnan(metrics['acc']):
                accs.append(metrics['acc'])
            if isinstance(metrics.get('acc_norm'), float) and not math.isnan(metrics['acc_norm']):
                acc_norms.append(metrics['acc_norm'])
        mean_acc, std_acc = _mean_std(accs)
        mean_acc_norm, std_acc_norm = _mean_std(acc_norms)
        results['templates'][tmpl_name] = {
            'tasks': per_task,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_acc_norm': mean_acc_norm,
            'std_acc_norm': std_acc_norm,
        }
    return results


def _render_md(results: Dict[str, Any]) -> str:
    def _render_tables(res: Dict[str, Any]) -> List[str]:
        out = []
        out.append("## Summary")
        out.append("| Template | Mean Acc | Std Acc | Mean Acc Norm | Std Acc Norm |")
        out.append("|---|---:|---:|---:|---:|")
        for tname, tres in res.get('templates', {}).items():
            out.append(
                f"| {tname} | {tres['mean_acc']:.2f} | {tres['std_acc']:.2f} | "
                f"{tres['mean_acc_norm']:.2f} | {tres['std_acc_norm']:.2f} |"
            )
        for tname, tres in res.get('templates', {}).items():
            out.append("")
            out.append(f"## Template: {tname}")
            out.append("| Task | Acc | Acc Norm |")
            out.append("|---|---:|---:|")
            for task, vals in tres.get('tasks', {}).items():
                out.append(f"| {task} | {vals['acc']:.2f} | {vals['acc_norm']:.2f} |")
        out.append("")
        return out

    lines = []
    lines.append("# Template Stability Evaluation")
    lines.append("")
    lines.append(f"Model: {results.get('model')}")
    lines.append(f"Device: {results.get('device')}")
    lines.append(f"Batch size: {results.get('batch_size')}")
    if results.get('template_profile'):
        lines.append(f"Template profile: {results.get('template_profile')}")
    if results.get('dtype'):
        lines.append(f"Dtype: {results.get('dtype')}")
    if results.get('limit') is not None:
        lines.append(f"Limit: {results.get('limit')}")
    if results.get('pad_ablation'):
        lines.append(f"Pad ablation: {results.get('pad_ablation')}")
    lines.append("")

    if results.get('pad_variants'):
        for vname, vres in results.get('pad_variants', {}).items():
            lines.append(f"## Pad variant: {vname}")
            if vres.get('padding_side'):
                lines.append(f"Padding side: {vres.get('padding_side')}")
            if vres.get('fix_pad_query_mask') is not None:
                lines.append(f"fix_pad_query_mask: {bool(vres.get('fix_pad_query_mask'))}")
            lines.append("")
            lines.extend(_render_tables(vres))
    else:
        if results.get('padding_side'):
            lines.append(f"Padding side: {results.get('padding_side')}")
        if results.get('fix_pad_query_mask') is not None:
            lines.append(f"fix_pad_query_mask: {bool(results.get('fix_pad_query_mask'))}")
        lines.append("")
        lines.extend(_render_tables(results))

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True, help='HF model id or path to local .pt checkpoint saved by this repo')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--hf_token', type=str, default=None)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None, help='Limit examples per task for quick runs')
    ap.add_argument('--dtype', type=str, default=None, help='float16/bfloat16/float32')
    ap.add_argument('--tasks', type=str, default='openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa')
    ap.add_argument('--templates', type=str, default='plain,qa,mc_letters,instruction')
    ap.add_argument('--template_profile', type=str, default=DEFAULT_TEMPLATE_PROFILE, choices=sorted(TASK_TEMPLATE_CONFIGS.keys()))
    ap.add_argument('--output_json', type=str, default='template_eval_results.json')
    ap.add_argument('--output_md', type=str, default='template_eval_results.md')
    ap.add_argument('--force_right_padding', action='store_true')
    ap.add_argument('--fix_pad_query_mask', action='store_true')
    ap.add_argument('--pad_ablation', type=str, default='none', choices=['none', 'compare', 'full'])
    args = ap.parse_args()

    # Load model
    if os.path.exists(args.model) and args.model.endswith('.pt'):
        model, tokenizer = get_model_from_local(args.model)
    else:
        model, tokenizer = get_model_from_huggingface(args.model, hf_token=args.hf_token)
    tokenizer = _ensure_tokenizer_compat(model, tokenizer, hf_token=args.hf_token)
    model = _to_device(model, args.device, args.dtype)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    global TEMPLATE_PROFILE
    TEMPLATE_PROFILE = args.template_profile

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tmpl_list = [t.strip() for t in args.templates.split(",") if t.strip()]
    for t in tmpl_list:
        if t not in TEMPLATES:
            raise ValueError(f"Unknown template '{t}'. Available: {', '.join(TEMPLATES.keys())}")
    for t in task_list:
        if t not in TASK_LOADERS:
            raise ValueError(f"Unknown task '{t}'. Available: {', '.join(TASK_LOADERS.keys())}")

    if args.pad_ablation == 'none':
        padding_side = "right" if args.force_right_padding else "left"
        _set_padding_side(tokenizer, padding_side)
        _ensure_pad_token(tokenizer)
        _apply_fix_pad_query_mask(model, bool(args.fix_pad_query_mask))
        results = _evaluate_templates(
            model,
            tokenizer,
            args,
            task_list,
            tmpl_list,
            padding_side=padding_side,
            fix_pad_query_mask=bool(args.fix_pad_query_mask),
        )
    else:
        if args.pad_ablation == 'compare':
            variants = [
                ("no_padfix", "left", False),
                ("with_flags", "right" if args.force_right_padding else "left", bool(args.fix_pad_query_mask)),
            ]
        else:
            variants = [
                ("left_no_fix", "left", False),
                ("left_fix", "left", True),
                ("right_no_fix", "right", False),
                ("right_fix", "right", True),
            ]
        results = {
            'model': args.model,
            'device': args.device,
            'batch_size': args.batch_size,
            'dtype': args.dtype,
            'limit': args.limit,
            'tasks': task_list,
            'template_profile': TEMPLATE_PROFILE,
            'pad_ablation': args.pad_ablation,
            'pad_variants': {},
        }
        for vname, side, fix in variants:
            _set_padding_side(tokenizer, side)
            _ensure_pad_token(tokenizer)
            _apply_fix_pad_query_mask(model, bool(fix))
            results['pad_variants'][vname] = _evaluate_templates(
                model,
                tokenizer,
                args,
                task_list,
                tmpl_list,
                padding_side=side,
                fix_pad_query_mask=bool(fix),
                pad_variant=vname,
            )

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    md = _render_md(results)
    with open(args.output_md, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"[Done] Wrote {args.output_json} and {args.output_md}")


if __name__ == '__main__':
    main()
