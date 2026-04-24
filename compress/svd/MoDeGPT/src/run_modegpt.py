import logging
import os

import torch

from src.model_utils import start_memory_usage_worker
from src.calibration import load_calibs

from src.compression.compress_mlp import compress_nystrom

from src.compression.compress_qk import compress_qk
from src.compression.compress_vo import compress_vo

from src.compression_utils import (
    allocate_global_sparsity,
)
from src.eval import (
    compute_perplexity,
)
from src.model_utils import (
    reload_compressed_model,
    save_compressed_model,
)

from src.adapters.model_adapter import ModelAdapter
from src.adapters.CompressionConfig import CompressionConfig
import gc
import optuna

logger = logging.getLogger("MoDeGPT")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_path = os.environ.get("MODEGPT_LOG_PATH", "logs/run_modegpt.log")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    file = logging.FileHandler(log_path)
    file.setFormatter(formatter)
    logger.addHandler(file)

import transformers.modeling_utils as tm
import torch.nn as nn

# 1. Target the correct function from your stack trace
_orig_load_param = tm._load_parameter_into_model


def _debug_load_param(*args, **kwargs):
    try:
        return _orig_load_param(*args, **kwargs)
    except Exception as e:  # Catch RuntimeError
        print("\n" + "!" * 60)
        print("🚨 CRASH DURING PARAMETER LOADING 🚨")

        # In this transformers function, the parameter name is usually the 2nd argument
        param_name = kwargs.get("param_name", kwargs.get("tensor_name", None))
        if not param_name and len(args) > 1:
            param_name = args[1]

        print(f"Failed on global parameter name: {param_name}")
        print(f"Original Error: {e}")
        print("!" * 60 + "\n")
        raise


# Apply the patch
tm._load_parameter_into_model = _debug_load_param


@torch.no_grad()
def main(trial: optuna.Trial = None, config: CompressionConfig | None = None):

    gc.collect()
    torch.cuda.empty_cache()

    adapter = None
    start_memory_usage_worker()

    if not config:
        config = CompressionConfig.from_args()

    print(config.to_dict())

    model, tokenizer = reload_compressed_model(config.model)

    adapter = ModelAdapter.from_model(model=model, tokenizer=tokenizer)
    adapter.config = config
    # adapter.metrics["args"] = vars(config)

    ppl_batch_size = int(os.environ.get("MODEGPT_PPL_BS", "4"))
    logger.info(f"Using perplexity batch size: {ppl_batch_size}")

    baseline_ppl = compute_perplexity(
        model,
        tokenizer,
        bs=ppl_batch_size,
        dataset=adapter.config.dataset,
        adapter=adapter,
    )

    logger.info(f"Baseline ppl: {baseline_ppl}")
    adapter.metrics["baseline-ppl"] = baseline_ppl

    torch.cuda.empty_cache()

    order = config.order
    n_layers = adapter.n_layers
    save_dir = os.path.join(config.output_dir, "model")

    layers_per_step = 48
    rotary_masks = []
    for start_idx in range(0, n_layers, layers_per_step):
        target_layers = [i for i in range(start_idx, min(n_layers, start_idx + layers_per_step))]

        cov_mlp, cov_q, cov_k, cov_x, bi_scores = load_calibs(
            adapter=adapter,
            n_samples=config.calib_size,
            batch_size=config.calibs_batch_size,
            dataset=adapter.config.dataset,
            target_layers=target_layers,
        )

        layer_keep_ratios = allocate_global_sparsity(
            bi_scores,
            compression_ratio=config.compression_ratio,
            smoothing=adapter.config.sparsity_smoothing,
            max_sparsity=adapter.config.max_sparsity,
            adapter=adapter,
        )

        if "mlp" in order:
            compress_nystrom(
                adapter=adapter,
                cov=cov_mlp,
                keep_ratios=layer_keep_ratios,
                target_layers=target_layers,
            )

        if "qk" in order:
            rms = compress_qk(
                adapter=adapter,
                cov=(cov_q, cov_k),
                keep_ratios=layer_keep_ratios,
                target_layers=target_layers,
            )
            rotary_masks.extend(rms)

        if "vo" in order:
            compress_vo(
                adapter=adapter,
                cov=cov_x,
                keep_ratios=layer_keep_ratios,
                target_layers=target_layers,
            )

        del cov_mlp, cov_q, cov_k, cov_x
        cov_mlp, cov_q, cov_k, cov_x = None, None, None, None
        gc.collect()
        torch.cuda.empty_cache()

    adapter.convert_model(saved_layers_dir=adapter.config.temp_storage_dir)
    adapter.patch_config()

    save_compressed_model(
        adapter,
        rotary_masks=rotary_masks,
        save_dir=save_dir,
        source_model_name=adapter.config.model,
    )

    del model
    del tokenizer
    del cov_k
    del cov_q
    del cov_x
    del cov_mlp

    torch.cuda.empty_cache()

    gc.collect()

    model, tokenizer = reload_compressed_model(save_dir)

    adapter.model = model
    adapter.tokenizer = tokenizer

    compressed_ppl = compute_perplexity(
        model,
        tokenizer,
        bs=ppl_batch_size,
        dataset=adapter.config.dataset,
        adapter=adapter,
    )

    adapter.metrics[f"ppl-{adapter.config.dataset}"] = compressed_ppl
    adapter.save_metrics()

    logger.info(f"Compressed (PPL): {compressed_ppl}")

    return compressed_ppl


if __name__ == "__main__":
    main()
