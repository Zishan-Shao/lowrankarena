import os
import gc
import logging
import random
import string

import hydra
import torch
import wandb
from functools import partial
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from aa_svd.utils import setup_seed
from aa_svd.models import create_model
from aa_svd.data import create_datasets, get_dataloader
from aa_svd.compression import apply_compression
from aa_svd.evaluate import evaluate

logger = logging.getLogger(__name__)


# Register resolver BEFORE @hydra.main
def random_id(length: int = 5):
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choices(chars, k=length))


OmegaConf.register_new_resolver("random_id", random_id, replace=True)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for model compression workflows."""
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    output_dir = HydraConfig.get().runtime.output_dir

    if cfg.wandb.use:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=output_dir,
            id=cfg.wandb.id,
            resume=cfg.wandb.resume,
            settings=wandb.Settings(code_dir="."))

    setup_seed(cfg.seed)

    model_dict = create_model(cfg.model)
    model = model_dict["model"]
    tokenizer = model_dict.get("tokenizer")
    logger.info(f"Created model: {cfg.model.name}")

    # Apply compression if specified
    if (cfg.get("compression") is not None
            and cfg.compression.get("method") is not None):
        if cfg.compression.get("need_calibration_data", False):
            assert tokenizer is not None, (
                "Tokenizer is required for creating calibration datasets"
            )
            # only train/val splits needed for calibration
            calibration_data_train, calibration_data_val = (
                create_datasets(cfg.data, tokenizer)[:2]
            )
            calibration_dataloader_train = get_dataloader(
                calibration_data_train, cfg.compression.get('batch_size', 4))
            calibration_dataloader_val = get_dataloader(
                calibration_data_val, cfg.compression.get('batch_size', 4))
        else:
            calibration_dataloader_train = None
            calibration_dataloader_val = None

        model = apply_compression(
            model, cfg.compression,
            calibration_dataloader_train=calibration_dataloader_train,
            calibration_dataloader_val=calibration_dataloader_val,
            tokenizer=tokenizer
        )

        logger.info(f"Applied compression: {cfg.compression.method}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Garbage collection and CUDA cache cleared")

    # Evaluate model
    if cfg.get("evaluate") is not None:
        if cfg.evaluate.get("compile", False) and hasattr(torch, "compile"):
            logger.info("Compiling model for faster inference")
            model = torch.compile(model)
        evaluate(cfg, model, tokenizer, dataset_name='final', step=None)

    # Save model if specified
    if cfg.get("save") is not None:
        save_path = os.path.join(cfg.save.dir, cfg.save.name)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(save_path)
        else:
            torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
