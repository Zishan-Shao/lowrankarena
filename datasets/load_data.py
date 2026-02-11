import os
import json
import csv
import re
from typing import List, Dict, Any, Optional


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Some dumps use a top-level object with a key like 'data'
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            return data['data']
        if isinstance(data, list):
            return data
    return []


def _parse_mathqa_options_text(opt_str: str) -> (List[str], Dict[str, int]):
    # Parse strings like "A) ... , B) ... , C) ..."
    choices, mapping = [], {}
    # accept optional whitespace between letter and ')', e.g., 'a ) 190'
    parts = re.split(r"\s*([A-Ea-e])\s*\)\s*", opt_str)
    for i in range(1, len(parts), 2):
        lab = parts[i].upper()
        text = parts[i + 1].strip()
        mapping[lab] = len(choices)
        choices.append(text)
    return choices, mapping


def _mathqa_choices_from_field(opt_field: Any) -> (List[str], Dict[str, int]):
    # Handle dict/list/string variants
    if isinstance(opt_field, dict):
        # e.g., {"A": "...", "B": "...", ...}
        out = []
        mapping = {}
        for lab in ['A', 'B', 'C', 'D', 'E']:
            if lab in opt_field:
                mapping[lab] = len(out)
                out.append(opt_field[lab])
        return out, mapping
    if isinstance(opt_field, list):
        out = [str(x) for x in opt_field]
        mapping = {chr(ord('A') + i): i for i in range(len(out))}
        return out, mapping
    if isinstance(opt_field, str):
        return _parse_mathqa_options_text(opt_field)
    return [], {}


def load_piqa_local(base_dir: Optional[str] = None, split: str = 'validation') -> List[Dict[str, Any]]:
    """
    Load PIQA from local files under the repo's datasets/ directory.
    Expected locations (first available is used):
      datasets/piqa/{valid.jsonl, validation.jsonl, dev.jsonl, test.jsonl, train.jsonl}
    Returns a list of items with keys: prompt, choices, answer_idx
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'piqa')
    if not os.path.isdir(base_dir):
        return []
    # Map split to candidate file names
    if split.lower() in ('validation', 'valid', 'dev'):
        candidates = ['valid.jsonl', 'validation.jsonl', 'dev.jsonl']
    elif split.lower() in ('test',):
        candidates = ['test.jsonl']
    else:
        candidates = ['train.jsonl']
    # Also check nested 'data' subdir commonly used by PIQA repos
    search_dirs = [base_dir, os.path.join(base_dir, 'data')]
    path = None
    for fn in candidates:
        for d in search_dirs:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                path = p
                break
        if path is not None:
            break
    if path is None:
        return []
    rows = _read_json_or_jsonl(path)
    # If labels aren't embedded, look for sidecar label list in same directory
    labels_sidecar = None
    base = os.path.dirname(path)
    if os.path.basename(path).startswith('valid'):
        cand_lab = ['valid-labels.lst', 'validation-labels.lst', 'dev-labels.lst']
    elif os.path.basename(path).startswith('train'):
        cand_lab = ['train-labels.lst']
    else:
        cand_lab = []
    for fn in cand_lab:
        lp = os.path.join(base, fn)
        if os.path.isfile(lp):
            labels_sidecar = lp
            break
    labels_list: Optional[List[int]] = None
    if labels_sidecar is not None:
        with open(labels_sidecar, 'r', encoding='utf-8') as f:
            labels_list = [int(l.strip()) for l in f if l.strip()]
    items: List[Dict[str, Any]] = []
    for idx_row, ex in enumerate(rows):
        goal = ex.get('goal', '')
        sol1 = ex.get('sol1', '')
        sol2 = ex.get('sol2', '')
        label = ex.get('label', None)
        if label is None and labels_list is not None and idx_row < len(labels_list):
            label = labels_list[idx_row]
        if label is None:
            # Some test splits lack labels; skip those
            continue
        try:
            idx = int(label)
        except Exception:
            idx = 0 if str(label).strip() in ('0', 'A', 'a') else 1
        items.append({'prompt': goal.strip() + '\nAnswer:', 'choices': [sol1, sol2], 'answer_idx': idx})
    return items


def load_mathqa_local(base_dir: Optional[str] = None, split: str = 'validation') -> List[Dict[str, Any]]:
    """
    Load MathQA from local files under datasets/MathQA.
    Accepts JSON/JSONL/CSV with common field names: Problem/options/correct or problem/options/label.
    Returns a list of items with keys: prompt, choices, answer_idx
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'MathQA')
    if not os.path.isdir(base_dir):
        return []
    # Candidate filenames per split
    split_l = split.lower()
    cand_json = [f'{split_l}.jsonl', f'{split_l}.json']
    cand_csv = [f'{split_l}.csv']
    # Also try common aliases
    alias_map = {
        'validation': ['valid', 'val', 'dev', 'test'],
        'test': ['test', 'validation', 'valid', 'val', 'dev'],
        'train': ['train'],
    }
    for alias in alias_map.get(split_l, [split_l]):
        cand_json.extend([f'{alias}.jsonl', f'{alias}.json'])
        cand_csv.append(f'{alias}.csv')
    path = None
    # Prefer JSON/JSONL
    for fn in cand_json + cand_csv:
        p = os.path.join(base_dir, fn)
        if os.path.isfile(p):
            path = p
            break
    if path is None:
        return []
    rows: List[Dict[str, Any]] = []
    if path.endswith('.csv'):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    else:
        rows = _read_json_or_jsonl(path)
    items: List[Dict[str, Any]] = []
    for ex in rows:
        q = ex.get('Problem') or ex.get('problem') or ex.get('question', '')
        opt_field = ex.get('options') or ex.get('Options') or ex.get('choices')
        choices, mapping = _mathqa_choices_from_field(opt_field if opt_field is not None else '')
        corr = ex.get('correct') or ex.get('label') or ex.get('answer') or 'A'
        idx = mapping.get(str(corr).strip().upper(), 0)
        if choices:
            items.append({'prompt': str(q).strip() + '\nAnswer:', 'choices': choices, 'answer_idx': idx})
    return items


def get_local_dataset(name: str, split: str = 'validation') -> List[Dict[str, Any]]:
    name_l = name.strip().lower()
    if name_l in ('piqa',):
        return load_piqa_local(split=split)
    if name_l in ('mathqa', 'math_qa'):
        return load_mathqa_local(split=split)
    if name_l in ('arc', 'arc_e', 'arc-e', 'arc_easy', 'arc-easy', 'arc_c', 'arc-c', 'arc_challenge', 'arc-challenge'):
        # Prefer local loader if added in the future; for now, use HF ai2_arc.
        try:
            return load_arc(split=split, variant=name_l)
        except Exception:
            return []
    return []


def load_arc(split: str = 'validation', variant: str = 'arc') -> List[Dict[str, Any]]:
    """
    Load ARC (AI2 Reasoning Challenge) via HuggingFace datasets 'ai2_arc'.
    variant: one of arc/arc_easy/arc_challenge/arc_e/arc_c
    Returns a list of {'prompt','choices','answer_idx'}
    """
    # Resolve config
    v = (variant or 'arc').lower()
    if any(x in v for x in ['challenge', 'arc_c', 'arc-c', 'arc_c', 'arc-challenge', 'c']):
        config = 'ARC-Challenge'
    elif any(x in v for x in ['easy', 'arc_e', 'arc-e', 'e']):
        config = 'ARC-Easy'
    else:
        config = 'ARC-Challenge'
    # Map split
    split_l = split.lower()
    hf_split = 'validation' if split_l in ('validation', 'valid', 'val', 'dev') else ('test' if split_l == 'test' else 'train')
    # Lazy import to avoid hard dependency at module import time
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset('ai2_arc', config, split=hf_split)
    items: List[Dict[str, Any]] = []
    for ex in ds:
        q = ex.get('question', '')
        ch = ex.get('choices', {}) or {}
        labels = list(ch.get('label', []))
        texts = [str(t) for t in ch.get('text', [])]
        ans = str(ex.get('answerKey', 'A')).strip().upper()
        try:
            idx = labels.index(ans)
        except Exception:
            # Fallback mapping A=0, B=1, ...
            idx = max(0, min(len(texts) - 1, ord(ans) - ord('A')))
        if texts:
            items.append({'prompt': str(q).strip() + '\nAnswer:', 'choices': texts, 'answer_idx': idx})
    return items
