import json
import logging
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import LlamaTokenizer, LlamaTokenizerFast
except Exception:
    LlamaTokenizer = None
    LlamaTokenizerFast = None
from tqdm import tqdm

from modules.remapping import DOBI_dequantize
from modules.module import *


def _normalize_tokenizer_hint(hint):
    if hint is None:
        return None
    hint = str(hint).strip()
    if not hint:
        return None
    if os.path.isfile(hint):
        hint = os.path.dirname(hint)
    return hint


def _tokenizer_ok(tokenizer) -> bool:
    return tokenizer is not None and not isinstance(tokenizer, bool) and callable(tokenizer)


def _maybe_fix_padding(tokenizer):
    if tokenizer is None:
        return tokenizer
    try:
        if getattr(tokenizer, 'pad_token', None) is None and getattr(tokenizer, 'eos_token', None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass
    try:
        tokenizer.padding_side = 'right'
    except Exception:
        pass
    return tokenizer


def _read_dobi_metadata(model_dir: str):
    model_dir = _normalize_tokenizer_hint(model_dir)
    if not model_dir:
        return {}
    meta_path = os.path.join(model_dir, 'dobi_metadata.json')
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _load_tokenizer_from_hint(hint):
    hint = _normalize_tokenizer_hint(hint)
    if not hint:
        return None

    lower = hint.lower()
    if 'llama' in lower:
        for cls in (LlamaTokenizerFast, LlamaTokenizer):
            if cls is None:
                continue
            try:
                kwargs = {}
                if cls is LlamaTokenizer:
                    kwargs['legacy'] = True
                tok = cls.from_pretrained(hint, **kwargs)
                if _tokenizer_ok(tok):
                    return _maybe_fix_padding(tok)
            except Exception:
                continue

    for use_fast in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(hint, use_fast=use_fast)
            if _tokenizer_ok(tok):
                return _maybe_fix_padding(tok)
        except Exception:
            continue
    return None


def _load_tokenizer_robust(primary_hint, model=None):
    hints = []
    primary_hint = _normalize_tokenizer_hint(primary_hint)
    if primary_hint:
        hints.append(primary_hint)

    metadata = _read_dobi_metadata(primary_hint) if primary_hint else {}
    for meta_key in ('tokenizer_hint', 'base_model_id'):
        meta_hint = _normalize_tokenizer_hint(metadata.get(meta_key))
        if meta_hint and meta_hint not in hints:
            hints.append(meta_hint)

    model_hint = None
    try:
        model_hint = _normalize_tokenizer_hint(getattr(getattr(model, 'config', None), '_name_or_path', None))
    except Exception:
        model_hint = None
    if model_hint and model_hint not in hints:
        hints.append(model_hint)

    for hint in hints:
        tok = _load_tokenizer_from_hint(hint)
        if tok is not None:
            return tok
    return None


def _set_model_name_or_path(model, model_path):
    try:
        model.config._name_or_path = str(_normalize_tokenizer_hint(model_path) or model_path)
    except Exception:
        pass
    try:
        model.name_or_path = str(_normalize_tokenizer_hint(model_path) or model_path)
    except Exception:
        pass


def _torch_load_allow_pickle(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_remapping_model(updated_model_path):
    logging.getLogger('transformers').setLevel(logging.ERROR)

    model_id = updated_model_path
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(config)
    state_dict = torch.load(f"{model_id}/pytorch_model.bin", map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.float16)
    _set_model_name_or_path(model, model_id)

    tokenizer = _load_tokenizer_robust(model_id, model=model)
    if tokenizer is None:
        raise TypeError(
            f'Could not load a tokenizer for remapping checkpoint at {model_id}. '
            'Make sure tokenizer files exist in the checkpoint directory or set SVDLLM_TOKENIZER_MODEL.'
        )

    mapping_info = torch.load(f"{model_id}/remapping_weight.pt", map_location='cpu')
    new_layer_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for name, module in tqdm(model.named_modules(), desc='Dequantize the model after remaping.'):
        if isinstance(module, nn.Linear) and all(x not in name for x in ['lm_head']):
            us_quan = mapping_info[name]['us_quan']
            vt_quan = mapping_info[name]['vt_quan']
            us_absmax = mapping_info[name]['us_absmax']
            vt_absmax = mapping_info[name]['vt_absmax']
            tuple_info = mapping_info[name]['tuple_info']
            dequan_us, dequan_vt = DOBI_dequantize(us_quan, vt_quan, us_absmax, vt_absmax, tuple_info, code=None)

            compress_size = dequan_vt.size(0) * dequan_vt.size(1) + dequan_us.size(0) * dequan_us.size(1)
            ori_size = module.in_features * module.out_features
            if ori_size > compress_size:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                NewLayer = SVDTransformLayer_remapping(
                    weight1=dequan_vt.T,
                    weight2=dequan_us.T,
                    bias=module.bias,
                    name=name,
                    device=new_layer_device,
                )
                setattr(parent, attr_name, NewLayer)
                del module
            else:
                new_weight = dequan_us @ dequan_vt
                module.weight.data = new_weight.detach()

            mapping_info[name] = {}

    model.eval()
    return model, tokenizer


def load_unremapping_model(model_id):
    checkpoint_path = f"{model_id}/DobiSVD_Model.pt"
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Could not find unremapping checkpoint: {checkpoint_path}')

    pruned_dict = _torch_load_allow_pickle(checkpoint_path, map_location='cpu')
    model = pruned_dict['model']
    tokenizer = pruned_dict.get('tokenizer')
    _set_model_name_or_path(model, model_id)

    if not _tokenizer_ok(tokenizer):
        tokenizer = _load_tokenizer_robust(model_id, model=model)

    if not _tokenizer_ok(tokenizer):
        raise TypeError(
            'Tokenizer object is not callable and could not be reconstructed for the unremapping checkpoint; '
            'make sure tokenizer files exist next to DobiSVD_Model.pt.'
        )

    _maybe_fix_padding(tokenizer)
    model.eval()
    return model, tokenizer
