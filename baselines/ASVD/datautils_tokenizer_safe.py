import os
import io
import json
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset

"""Tokenizer-safe data utils.

This module is a drop-in replacement for the original ASVD `datautils.py`.
It adds a small guard so that if a caller accidentally passes a boolean
instead of a tokenizer object, we reload the tokenizer from `model_id`
instead of crashing with:

    TypeError: 'bool' object is not callable
"""

# Added for PTB dataset of new version 4.4.0
PTB_URL_BASE = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
PTB_FILES = {
    "train": PTB_URL_BASE + "ptb.train.txt",
    "validation": PTB_URL_BASE + "ptb.valid.txt",
    "test": PTB_URL_BASE + "ptb.test.txt",
}


# ----------------------------------------------------------------------
# Tokenizer safety guard
# ----------------------------------------------------------------------
_TOKENIZER_CACHE: Dict[str, Any] = {}


def _resolve_hf_token() -> Optional[str]:
    """Best-effort Hugging Face token from common env vars."""
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_ACCESS_TOKEN"):
        v = os.getenv(k)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _ensure_tokenizer(tokenizer: Any, model_id: Optional[str] = None) -> Any:
    """Return a callable Hugging Face tokenizer.

    Some call sites accidentally pass a boolean (e.g., False) instead of a tokenizer.
    This guard prevents: TypeError: 'bool' object is not callable.

    If `tokenizer` is invalid, we reload from `model_id`.
    """

    # Happy path: already a callable tokenizer object.
    try:
        if tokenizer is not None and not isinstance(tokenizer, bool) and callable(tokenizer):
            # Ensure pad token exists for LLaMA-like tokenizers.
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
    except Exception:
        pass

    model_key = str(model_id).strip() if model_id is not None else ""

    # Cache to avoid repeatedly re-loading tokenizers.
    if model_key and model_key in _TOKENIZER_CACHE:
        cached = _TOKENIZER_CACHE.get(model_key)
        try:
            if cached is not None and not isinstance(cached, bool) and callable(cached):
                return cached
        except Exception:
            pass
        # Drop invalid cache entry and rebuild.
        try:
            del _TOKENIZER_CACHE[model_key]
        except Exception:
            pass

    if not model_key:
        raise TypeError(
            "Tokenizer is not callable (likely a bool). "
            "Pass a Hugging Face tokenizer object, or provide model_id so it can be reloaded."
        )

    # Lazy import to avoid importing transformers at module import time.
    from transformers import AutoTokenizer  # type: ignore

    hf_token = _resolve_hf_token()
    kwargs: Dict[str, Any] = {"trust_remote_code": True, "use_fast": True}
    if hf_token:
        kwargs["token"] = hf_token

    # Try fast tokenizer first; fall back to slow. Also support older transformers token arg.
    try:
        tok = AutoTokenizer.from_pretrained(model_key, **kwargs)
    except TypeError:
        # Older transformers versions use `use_auth_token` instead of `token`.
        if "token" in kwargs:
            kwargs.pop("token", None)
            if hf_token:
                kwargs["use_auth_token"] = hf_token
        tok = AutoTokenizer.from_pretrained(model_key, **kwargs)
    except Exception:
        kwargs["use_fast"] = False
        try:
            tok = AutoTokenizer.from_pretrained(model_key, **kwargs)
        except TypeError:
            if "token" in kwargs:
                kwargs.pop("token", None)
                if hf_token:
                    kwargs["use_auth_token"] = hf_token
            tok = AutoTokenizer.from_pretrained(model_key, **kwargs)

    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    try:
        if tok is not None and not isinstance(tok, bool) and callable(tok):
            _TOKENIZER_CACHE[model_key] = tok
    except Exception:
        pass

    return tok


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# Calibration / loaders
# ----------------------------------------------------------------------
def sample_train_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048, model_id: Optional[str] = None):
    set_seed(seed)
    tokenizer = _ensure_tokenizer(tokenizer, model_id)

    if "wikitext2" in name:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        traindata = "\n\n".join(traindata["text"])
    elif "c4" in name:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - seqlen * 2 - 1)
        j = i + seqlen * 2
        if not callable(tokenizer):
            tokenizer = _ensure_tokenizer(tokenizer, model_id)
        # just before trainenc = tokenizer(...)
        print("before tokenize:", __file__, type(tokenizer), callable(tokenizer), repr(tokenizer))
        assert not isinstance(tokenizer, bool), f"tokenizer turned bool: {tokenizer!r}"
        trainenc = tokenizer(traindata[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        trainloader.append(inp)
    return trainloader


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048, model_id: Optional[str] = None):
    tokenizer = _ensure_tokenizer(tokenizer, model_id)

    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000 * percent / 100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)
    processed_dataset = dataset.map(tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count())
    return processed_dataset


def get_english_quote(dataset_name, tokenizer, model_id: Optional[str] = None):
    tokenizer = _ensure_tokenizer(tokenizer, model_id)
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent, model_id: Optional[str] = None):
    tokenizer = _ensure_tokenizer(tokenizer, model_id)
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent, model_id=model_id)
    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer, model_id=model_id)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data


llama_chat_format = """<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{{ instruction }} [/INST] {{ response }} </s>
"""


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3, use_bos=False):
    # IMPORTANT: enforce callable tokenizer here to avoid bool-tokenizer crashes.
    tokenizer = _ensure_tokenizer(tokenizer, model_id)
    set_seed(seed)

    print(f" get_ptq_calib_data {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = f"cache/{name}_{str(model_id).replace('/', '_')}_{nsamples}_{seqlen}_{seed}_bos{use_bos}.pt"
    print(f"cache_file={cache_file}")

    if not os.path.exists("cache"):
        os.makedirs("cache")

    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset

    if name == "c4":
        traindata = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        traindata = load_dataset("text", data_files=PTB_FILES, split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name == "alpaca":
        # this is for chat models
        data_path = "data/alpaca_data.json"
        list_data_dict = jload(data_path)
        traindataset = []
        selected_data_dict = random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s = llama_chat_format.format(instruction=example["instruction"], response=example["output"])
                if not callable(tokenizer):
                    tokenizer = _ensure_tokenizer(tokenizer, model_id)
                trainenc = tokenizer(s, return_tensors="pt")
                inp = trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        return traindataset
    elif name == "selfgen":
        raise NotImplementedError
    else:
        raise NotImplementedError

    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        txt = tot_text[i:j]
        ind = txt.find(".")
        txt = txt[ind + 1 :].strip()

        if use_bos:
            bos = getattr(tokenizer, "bos_token", None) or getattr(tokenizer, "eos_token", None)
            if bos is not None:
                txt = str(bos) + txt

        if not callable(tokenizer):
            tokenizer = _ensure_tokenizer(tokenizer, model_id)
        trainenc = tokenizer(txt, return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

    torch.save(traindataset, cache_file)
    return traindataset


def get_eval_loaders(name, tokenizer, model_id: Optional[str] = None):
    tokenizer = _ensure_tokenizer(tokenizer, model_id)

    if "wikitext2" in name:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc

    if "ptb" in name:
        valdata = load_dataset("text", data_files=PTB_FILES, split="validation")
        testenc = tokenizer("\n\n".join(valdata["text"]), return_tensors="pt")
        return testenc

    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc

    raise NotImplementedError
