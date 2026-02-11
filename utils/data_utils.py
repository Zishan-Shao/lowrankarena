import os
import random
import torch
import sys
from typing import Optional
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

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
    # Ensure we have a callable HF tokenizer. Some older checkpoints or
    # environments may pass a placeholder (e.g., bool). Reload a sane tokenizer
    # if needed using a best-effort model hint from env.
    try:
        _ok = callable(tokenizer)
    except Exception:
        _ok = False
    if not _ok:
        try:
            from transformers import AutoTokenizer
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
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_hint, trust_remote_code=True, use_fast=True, token=hf_token)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_hint, trust_remote_code=True, use_fast=False, token=hf_token)
        except Exception:
            pass
    # Include tokenizer vocab size to avoid cache collisions across different tokenizers/models.
    # Be robust if tokenizer is missing or not a HF tokenizer (e.g., older pickled checkpoints).
    try:
        vocab_hint = int(getattr(tokenizer, "vocab_size", 0) or 0)
    except Exception:
        vocab_hint = 0
    cache_file = f"cache/{name}_{vocab_hint}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    nsamples += 1 #############################
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        # Guard against cache/tokenizer mismatch: rebuild if any token id exceeds vocab.
        try:
            max_token = max(batch["input_ids"].max().item() for batch in traindataset)
            vsize = getattr(tokenizer, "vocab_size", None)
            if isinstance(vsize, int) and vsize > 0 and max_token >= vsize:
                print(f"[Cache] Discarding cached calib data at {cache_file} (max token {max_token} >= vocab {vsize}); regenerating.")
                traindataset = None
        except Exception:
            traindataset = None
        if traindataset is not None:
            return traindataset
    if name == "c4":
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
            traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
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
    if name == "c4" and small_c4_loaded:
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
    elif name == "c4" and use_streaming_c4:
        # Iterate the streaming dataset and take the first ~max_docs that yield
        # a seqlen window. This avoids large downloads and 1024-file resolution.
        taken = 0
        docs_seen = 0
        max_docs = max(nsamples * 3, nsamples + 32)
        for item in tqdm(traindata, desc="[c4(stream)] scan docs", leave=False):
            docs_seen += 1
            try:
                text = item.get('text', '') if isinstance(item, dict) else item['text']
            except Exception:
                continue
            enc = tokenizer(text, return_tensors="pt")
            T = enc.input_ids.shape[1]
            if T >= seqlen + 1:
                start = random.randint(0, T - seqlen - 1)
                window = enc.input_ids[:, start:start + seqlen]
                attn = torch.ones_like(window)
                traindataset.append({"input_ids": window, "attention_mask": attn})
                taken += 1
                if taken >= nsamples:
                    break
            if docs_seen >= max_docs and taken >= nsamples:
                break
    else:
        # Original behavior (used for non-C4 or when official c4 fallback used)
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
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
        valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', cache_dir=dataset_cache_dir)
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
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', cache_dir=dataset_cache_dir)
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
    # Ensure we have a callable tokenizer
    try:
        _ok = callable(tokenizer)
    except Exception:
        _ok = False
    if not _ok:
        try:
            from transformers import AutoTokenizer
            model_hint = os.getenv('SVDLLM_TOKENIZER_MODEL', None)
            if model_hint is None:
                model_hint = 'openlm-research/open_llama_7b'
            hf_token = (
                os.getenv('HF_TOKEN')
                or os.getenv('HUGGINGFACE_TOKEN')
                or os.getenv('HUGGINGFACE_HUB_TOKEN')
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_hint, trust_remote_code=True, use_fast=True, token=hf_token)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_hint, trust_remote_code=True, use_fast=False, token=hf_token)
        except Exception:
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
    if 'ptb' in name:
        try:
            test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
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
            test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
            test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
        except FileNotFoundError:
            # Fallback to HF c4 validation subset if local file is missing
            test_data = load_dataset("allenai/c4", "en", split="validation[:2000]")
            test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
