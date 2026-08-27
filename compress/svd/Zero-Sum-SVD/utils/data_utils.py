import json
import hashlib
import os
import random
import torch
import sys
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


def _resolve_c4_json(split):
    """Resolve pinned C4 JSON without assuming it was copied into this repo."""
    split = str(split).lower()
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported C4 split: {split}")
    env_name = f"LOWRANKARENA_C4_{split.upper()}_FILE"
    candidates = [
        os.environ.get(env_name, "").strip(),
        os.path.join(current_path, f"c4-{split}.json"),
    ]
    path = next((candidate for candidate in candidates if candidate and os.path.isfile(candidate)), None)
    if path is None:
        raise FileNotFoundError(
            f"C4 {split} JSON not found; set {env_name} to the pinned local file"
        )
    return path


def _build_text_from_example(ex, text_fields=None):
    """Build text string from a dataset example using specified or guessed fields."""
    if text_fields:
        parts = []
        for f in text_fields:
            v = ex.get(f, None)
            if v is None:
                continue
            if isinstance(v, list):
                v = " ".join([str(x) for x in v])
            parts.append(str(v))
        return "\n\n".join([p for p in parts if p])

    # heuristic fallback
    for k in ["text", "question", "prompt", "query", "context", "caption", "title"]:
        if k in ex and ex[k] is not None:
            v = ex[k]
            if isinstance(v, list):
                v = " ".join([str(x) for x in v])
            return str(v)

    return ""


def _load_local_jsonl(path):
    """Load rows from a local JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_calib_train_data(
    name,
    tokenizer,
    nsamples,
    seqlen=2048,
    seed=3,
    batch_size=1,
    dataset_cache_dir=None,
    split="train",
    text_fields=None,
):
    """
    Load calibration/training data from various sources:
    - Existing special datasets: c4, ptb, wikitext2
    - Any HuggingFace dataset name
    - Local .jsonl files
    """
    shared_path = os.environ.get("LOWRANKARENA_CALIBRATION_FILE", "").strip()
    if shared_path:
        payload = torch.load(shared_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "lowrankarena_shared_calibration_v1":
            raise ValueError(f"Unsupported shared calibration format in {shared_path}")
        input_ids = payload["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[1] != seqlen:
            raise ValueError(
                f"Shared calibration shape {tuple(input_ids.shape)} does not match seqlen={seqlen}"
            )
        required = nsamples * batch_size
        if input_ids.shape[0] < required:
            raise ValueError(f"Shared calibration has {input_ids.shape[0]} samples, requested {required}")
        recorded_vocab = payload.get("tokenizer_vocab_size")
        if recorded_vocab is not None and int(recorded_vocab) != int(tokenizer.vocab_size):
            raise ValueError(
                f"Shared calibration tokenizer vocab {recorded_vocab} != active vocab {tokenizer.vocab_size}"
            )
        selected = input_ids[:required].to(torch.long)
        print(
            f"Using shared calibration {shared_path}: dataset={payload.get('dataset')} "
            f"samples={required} seqlen={seqlen} sha256={payload.get('input_ids_sha256')}"
        )
        return [
            {
                "input_ids": selected[start:start + batch_size],
                "attention_mask": torch.ones_like(selected[start:start + batch_size]),
                "sample_id": start // batch_size,
            }
            for start in range(0, required, batch_size)
        ]

    tokenizer_identity = {
        "class": tokenizer.__class__.__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    tokenizer_tag = hashlib.sha256(
        json.dumps(tokenizer_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    cache_file = (
        f"cache/{str(name).replace('/','__')}_{nsamples}_{seqlen}_{seed}_{batch_size}"
        f"_tok{tokenizer_tag}.pt"
    )
    print(f"\n🔍 get_calib_train_data called:")
    print(f"   name={name}, text_fields={text_fields}, split={split}")
    print(f"   nsamples={nsamples}, seqlen={seqlen}, seed={seed}")
    print(f"   cache_file={cache_file}")

    if not os.path.exists("cache"):
        os.makedirs("cache")

    if os.path.exists(cache_file):
        print(f"   ✓ Loading from cache: {cache_file}")
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "zssvd_calibration_cache_v2":
            raise ValueError(f"Unsupported tokenizer-aware calibration cache: {cache_file}")
        if payload.get("tokenizer_identity") != tokenizer_identity:
            raise ValueError(f"Tokenizer identity mismatch in calibration cache: {cache_file}")
        return payload["data"]

    texts = []

    # Debug: print calibration data loading info
    print(f"\n🔍 Calibration data loading debug:")
    print(f"   name={name}, type={type(name)}")
    print(f"   isinstance(name, str)={isinstance(name, str)}")
    print(f"   os.path.isfile(name)={os.path.isfile(name) if isinstance(name, str) else 'N/A'}")
    print(f"   name.endswith('.jsonl')={name.endswith('.jsonl') if isinstance(name, str) else 'N/A'}")
    print(f"   text_fields={text_fields}")
    print(f"   split={split}, nsamples={nsamples}, seqlen={seqlen}")

    # Case A: local jsonl path
    if isinstance(name, str) and os.path.isfile(name) and name.endswith(".jsonl"):
        print(f"📂 [Case A] Loading local JSONL file: {name}")
        rows = _load_local_jsonl(name)
        parsed_fields = text_fields.split(",") if isinstance(text_fields, str) else text_fields
        for ex in rows:
            t = _build_text_from_example(ex, text_fields=parsed_fields)
            if t:
                texts.append(t)

    # Case B: existing special datasets
    elif name == "c4":
        print(f"📂 [Case B] Loading special dataset: c4")
        traindata = load_dataset("json", data_files=_resolve_c4_json("train"))["train"]
        texts = list(traindata["text"])
    elif name == "ptb":
        print(f"📂 [Case B] Loading special dataset: ptb")
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train", cache_dir=dataset_cache_dir)
        texts = list(traindata["sentence"])
    elif name == "wikitext2":
        print(f"📂 [Case B] Loading special dataset: wikitext2")
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
        texts = list(traindata["text"])

    # Case C: any HuggingFace dataset
    else:
        print(f"📦 [Case C] Loading HuggingFace dataset: {name} (split={split})")
        ds = load_dataset(name, split=split, cache_dir=dataset_cache_dir)
        parsed_fields = text_fields.split(",") if isinstance(text_fields, str) else text_fields
        for ex in ds:
            t = _build_text_from_example(ex, text_fields=parsed_fields)
            if t:
                texts.append(t)

    print(f"   ✓ Loaded {len(texts)} text samples")

    if len(texts) == 0:
        raise ValueError(f"No text could be built from dataset={name}")

    # For standard datasets (c4, ptb, wikitext2): use the join-and-slice
    # approach that produces full-length sequences with no padding.
    # For everything else (JSONL, HuggingFace) fall back to per-text tokenization.
    is_standard = name in ("c4", "ptb", "wikitext2")

    if is_standard:
        tot_text = "\n\n".join(texts)
        random.seed(seed)
        nsamples_iter = nsamples + 1

        out = []
        sample_counter = 0
        inp = None
        for s in range(nsamples_iter):
            i = random.randint(0, len(tot_text) - seqlen - 1)
            j = i + seqlen * 10
            trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
            if trainenc.input_ids.shape[1] < seqlen:
                continue
            if s % batch_size == 0:
                if s != 0:
                    attention_mask = torch.ones_like(inp)
                    out.append({
                        "input_ids": inp,
                        "attention_mask": attention_mask,
                        "sample_id": sample_counter,
                    })
                    sample_counter += 1
                inp = trainenc.input_ids[:, :seqlen]
            else:
                inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)
    else:
        # Fallback: per-text tokenization for non-standard datasets
        rng = random.Random(seed)
        n_total = nsamples * batch_size
        idxs = [rng.randrange(0, len(texts)) for _ in range(n_total)]

        out = []
        for i in range(nsamples):
            batch_ids = []
            batch_mask = []
            for j in range(batch_size):
                idx = idxs[i * batch_size + j]
                t = texts[idx]
                enc = tokenizer(
                    t,
                    truncation=True,
                    max_length=seqlen,
                    padding="max_length",
                    return_tensors="pt",
                )
                batch_ids.append(enc["input_ids"])
                batch_mask.append(enc["attention_mask"])

            batch_dict = {
                "input_ids": torch.cat(batch_ids, dim=0).long(),
                "attention_mask": torch.cat(batch_mask, dim=0).long(),
            }
            out.append(batch_dict)

    torch.save(
        {
            "format": "zssvd_calibration_cache_v2",
            "tokenizer_identity": tokenizer_identity,
            "data": out,
        },
        cache_file,
    )
    return out



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
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

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
    traindata = load_dataset("json", data_files=_resolve_c4_json("train"))['train']
    valdata = load_dataset("json", data_files=_resolve_c4_json("validation"))['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
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
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
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
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

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
    traindata = load_dataset("json", data_files=_resolve_c4_json("train"))['train']
    valdata = load_dataset("json", data_files=_resolve_c4_json("validation"))['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
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
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    # Helper: check if tokenizer has large model_max_length (Llama/Llama 2 style)
    def _should_use_legacy_join(tok):
        return getattr(tok, 'model_max_length', 0) > 200000

    # Helper: probe if tokenizer adds BOS token automatically
    def _tokenizer_adds_bos(tok):
        probe = tok("a", add_special_tokens=True).input_ids
        bos_id = getattr(tok, 'bos_token_id', None)
        return bos_id is not None and len(probe) > 0 and probe[0] == bos_id

    # Legacy approach: join all text, tokenize once, chunk
    def process_data_legacy(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer(
            "\n\n".join(samples[field_name]),
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    # Streaming approach: tokenize texts one by one, concatenate tokens
    def process_data_streaming(samples, tokenizer, seq_len, field_name):
        texts = samples[field_name]
        sep = "\n\n"
        adds_bos = _tokenizer_adds_bos(tokenizer)
        bos_id = getattr(tokenizer, 'bos_token_id', None)

        all_ids = []
        for idx, text in enumerate(texts):
            if idx == 0:
                # First text: tokenize normally (includes BOS if tokenizer adds it)
                ids = tokenizer(text, add_special_tokens=True).input_ids
            else:
                # Subsequent texts: prepend separator, tokenize, strip BOS if added
                ids = tokenizer(sep + text, add_special_tokens=True).input_ids
                if adds_bos and len(ids) > 0 and ids[0] == bos_id:
                    ids = ids[1:]
            all_ids.extend(ids)

        test_ids = torch.tensor(all_ids, dtype=torch.long)
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    # Dispatcher: choose legacy or streaming based on tokenizer
    def process_data(samples, tokenizer, seq_len, field_name):
        if _should_use_legacy_join(tokenizer):
            return process_data_legacy(samples, tokenizer, seq_len, field_name)
        else:
            return process_data_streaming(samples, tokenizer, seq_len, field_name)

    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files=_resolve_c4_json("validation"))['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
