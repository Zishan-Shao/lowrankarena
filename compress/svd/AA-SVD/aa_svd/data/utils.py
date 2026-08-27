import json
import os
import random
from typing import Any, Optional

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm


def _apply_preprocessing(
    name: str,
    dataset: Any,
    config: Any,
    tokenizer: Optional[Any] = None,
) -> Any:
    """Apply preprocessing to the dataset."""
    if 'test' in dataset:
        dataset['val'] = dataset.pop('test')

    task_type = config.get('type')

    if task_type == 'next_token_prediction':
        def _tokenize(examples):
            key = 'text' if 'text' in examples else 'sentence'
            text = examples[key]
            new_examples = tokenizer(text)
            new_examples['input_ids'] = [
                x + [tokenizer.eos_token_id] for x in new_examples['input_ids']
            ]
            new_examples['len'] = [len(x) for x in new_examples['input_ids']]
            for key in list(new_examples.keys()):
                if key not in ['input_ids', 'len']:
                    new_examples.pop(key)
            return new_examples

        col_names = dataset['train'].column_names
        remove_cols = ['text'] if 'text' in col_names else ['sentence']
        dataset = dataset.map(
            _tokenize, batched=True, remove_columns=remove_cols
        )
        dataset = dataset.filter(lambda x: x['len'] > 2)
        return dataset

    elif task_type == 'compression_default':
        new_dataset = DatasetDict()
        dataset_keys = list(dataset.keys())
        if 'train' in dataset_keys:
            dataset_keys.remove('train')
            dataset_keys = ['train'] + dataset_keys

        nsamples = 2048
        seqlen = 2048
        random.seed(3)
        for dataset_split in dataset_keys:
            try:
                tot_text = "\n\n".join(dataset[dataset_split]['text'])
            except Exception:
                tot_text = "\n\n".join(dataset[dataset_split]['sentence'])
            sequences = []
            lengths = []
            for _ in range(nsamples):
                i = random.randint(0, len(tot_text) - seqlen - 1)
                trainenc = tokenizer(tot_text[i:i + seqlen * 10])
                if len(trainenc.input_ids) < seqlen:
                    continue
                inp = trainenc.input_ids[:seqlen]
                sequences.append(inp)
                lengths.append(len(inp))
            new_dataset[dataset_split] = Dataset.from_dict({
                'input_ids': sequences,
                'len': lengths,
            })
        return new_dataset

    elif task_type == 'compression_v2':
        new_dataset = DatasetDict()
        dataset_keys = list(dataset.keys())
        if 'train' in dataset_keys:
            dataset_keys.remove('train')
            dataset_keys = ['train'] + dataset_keys

        nsamples = 512
        seqlen = 2048
        random.seed(3)
        for dataset_split in dataset_keys:
            sequences = []
            lengths = []
            for _ in range(nsamples):
                i = random.randint(0, len(dataset[dataset_split]['text']))
                try:
                    tot_text = "\n\n".join(
                        dataset[dataset_split]['text'][i:i + 50]
                    )
                except Exception:
                    tot_text = "\n\n".join(
                        dataset[dataset_split]['sentence'][i:i + 50]
                    )
                trainenc = tokenizer(tot_text[:seqlen * 10])
                if len(trainenc.input_ids) < seqlen:
                    continue
                inp = trainenc.input_ids[:seqlen]
                sequences.append(inp)
                lengths.append(len(inp))
            new_dataset[dataset_split] = Dataset.from_dict({
                'input_ids': sequences,
                'len': lengths,
            })
        return new_dataset

    else:
        raise ValueError(f"Unsupported preprocessing task type: {task_type}")


def save_dataset_to_disk(dataset, dir_path, dtype=np.uint16):
    os.makedirs(dir_path, exist_ok=True)
    dtype = np.dtype(dtype)
    with open(os.path.join(dir_path, "dtype.json"), "w", encoding="utf-8") as handle:
        json.dump({"dtype": dtype.name}, handle)
    for split, dset in dataset.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(dir_path, f'{split}.bin')
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = min(1024, len(dset))
        idx = 0
        desc = f'writing {filename}'
        for batch_idx in tqdm(range(total_batches), desc=desc):
            # Keep the Arrow column in its default Python-list format here.
            # ``datasets`` releases predating NumPy 2 request
            # ``np.array(..., copy=False)`` in their NumPy formatter, which
            # raises under NumPy 2 even though a copy is harmless for this
            # one-time cache write.  Converting the list explicitly preserves
            # the token values and works with both NumPy 1.x and 2.x.
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            )
            arr_batch = np.concatenate([
                np.asarray(row, dtype=dtype) for row in batch['input_ids']
            ])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
