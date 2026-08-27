import logging
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SharedCalibrationDataset(torch.utils.data.Dataset):
    """Token-identical calibration windows supplied by LowRankArena."""

    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids.to(dtype=torch.long, device="cpu").contiguous()

    def __len__(self):
        return int(self.input_ids.shape[0])

    def __getitem__(self, index):
        row = self.input_ids[index]
        targets = torch.cat((row[1:], row[-1:]))
        return {"input_ids": row, "targets": targets}


def _load_shared_calibration(config, tokenizer):
    path = os.environ.get("LOWRANKARENA_CALIBRATION_FILE", "").strip()
    if not path:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "lowrankarena_shared_calibration_v1":
        raise ValueError(f"Unsupported shared calibration format in {path}")
    input_ids = payload["input_ids"]
    nsamples = int(config.get("num_samples", input_ids.shape[0]))
    block_size = int(config.get("block_size", input_ids.shape[1]))
    if input_ids.ndim != 2 or input_ids.shape[1] != block_size:
        raise ValueError(
            f"Shared calibration shape {tuple(input_ids.shape)} does not match block_size={block_size}"
        )
    if input_ids.shape[0] < nsamples:
        raise ValueError(f"Shared calibration has {input_ids.shape[0]} samples, requested {nsamples}")
    recorded_vocab = payload.get("tokenizer_vocab_size")
    if recorded_vocab is not None and int(recorded_vocab) != int(tokenizer.vocab_size):
        raise ValueError(
            f"Shared calibration tokenizer vocab {recorded_vocab} != active vocab {tokenizer.vocab_size}"
        )
    logger.info(
        "Using shared calibration %s: dataset=%s samples=%d seqlen=%d sha256=%s",
        path, payload.get("dataset"), nsamples, block_size, payload.get("input_ids_sha256"),
    )
    train = SharedCalibrationDataset(input_ids[:nsamples])
    val_count = min(int(config.get("num_samples_val", 64)), nsamples)
    val = SharedCalibrationDataset(input_ids[:val_count])
    return train, val


def create_datasets(
    config: OmegaConf,
    tokenizer: Optional[Any] = None,
) -> Tuple[Dataset, Dataset]:
    """Create train/val datasets based on configuration."""
    shared = _load_shared_calibration(config, tokenizer)
    if shared is not None:
        return shared
    dataset_type = config.get('type')
    if dataset_type == 'huggingface':
        return _load_huggingface_dataset(config, tokenizer)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _load_huggingface_dataset(
    config: OmegaConf,
    tokenizer: Optional[Any] = None,
) -> Tuple[Dataset, Dataset]:
    """Load a HuggingFace dataset and cache it to disk as binary shards."""
    block_size = config.get('block_size', 512)
    num_samples = config.get('num_samples', None)
    num_samples_val = config.get('num_samples_val', num_samples)
    sampling = config.get('sampling', 'random')
    seed = config.get('seed', 42)

    data_path = os.path.join(
        config.get('data_path'),
        config.get('name'),
        config.get('subset'),
        tokenizer.name_or_path,
        config.get('task_type'),
    )

    if os.path.exists(data_path):
        try:
            logger.info(f"Loading dataset from {data_path}")
            from .iterable_text_dataset import TextCalibrationDataset
            train_dataset = TextCalibrationDataset(
                data_path, block_size=block_size,
                num_samples=num_samples, sampling=sampling,
                split='train', seed=seed,
            )
            val_dataset = TextCalibrationDataset(
                data_path, block_size=block_size,
                num_samples=num_samples_val, sampling=sampling,
                split='val', seed=seed,
            )
            return train_dataset, val_dataset
        except Exception as e:
            logger.warning(
                f"Failed to load dataset from disk: {e}. "
                "Will download from HuggingFace and preprocess."
            )

    name = config.get('name')
    subset = config.get('subset')
    logger.info(f"Loading subset={subset} for dataset={name} from HuggingFace")

    if name == "c4":
        url = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main"
            "/en/c4-train.00000-of-01024.json.gz"
        )
        dataset = load_dataset("json", data_files=url)
        val_url = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main"
            "/en/c4-validation.00000-of-00008.json.gz"
        )
        dataset["val"] = load_dataset("json", data_files=val_url)["train"]
    else:
        dataset = load_dataset(name, subset, num_proc=8)

    if hasattr(config, 'preprocessing'):
        from .utils import _apply_preprocessing
        dataset = _apply_preprocessing(
            name, dataset, config.preprocessing, tokenizer
        )

    from .utils import save_dataset_to_disk
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    logger.info(f"Saving dataset to {data_path}")
    token_dtype = (
        np.uint32
        if int(tokenizer.vocab_size) > np.iinfo(np.uint16).max
        else np.uint16
    )
    save_dataset_to_disk(dataset, data_path, dtype=token_dtype)

    from .iterable_text_dataset import TextCalibrationDataset
    train_dataset = TextCalibrationDataset(
        data_path, block_size=block_size,
        num_samples=num_samples, sampling=sampling,
        split='train', seed=seed,
    )
    val_dataset = TextCalibrationDataset(
        data_path, block_size=block_size,
        num_samples=num_samples_val, sampling=sampling,
        split='val', seed=seed,
    )
    return train_dataset, val_dataset


def get_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader for the given dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
