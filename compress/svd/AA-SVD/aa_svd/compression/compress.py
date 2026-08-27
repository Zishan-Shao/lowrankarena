import gc
import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict

import torch
import wandb
from tqdm import tqdm

from .decompose import (
    compress_module_obj1,
    compress_module_obj2,
    compress_module_obj2_evd,
    compress_module_obj3,
    compress_module_obj4,
)

from .utils import map_tensors, get_log_key, rebatch_stream
from .model_adapter import ModelAdapter, LayerAdapter
from .adapters import MODEL_ADAPTER_REGISTRY
from .metrics import MSEMetric, CosineDistanceMetric, NormComparatorMetric, PerplexityMetric
from .compressed_linear import CompressedLinear, QuantizedCompressedLinear

logger = logging.getLogger(__name__)


@torch.no_grad()
def get_layer0_inputs(model_adapter: ModelAdapter, batch: torch.Tensor, device) -> tuple[torch.Tensor, tuple, dict[str, Any]]:
    """
    Returns the inputs to the first layer of the model (after embeddings).

    Also returns the additional args and kwargs that are passed to
    the first layer (such as the attention mask, or caches K/V values).

    This relies on all arguments to subsequent layers being the same.

    NB: this won't work from OPT 350m.
    """

    # Move embeddings to device.
    for module in model_adapter.get_embeddings():
        if isinstance(module, torch.nn.Module):
            for param in module.parameters():
                param.data = param.data.to(device)
        elif isinstance(module, torch.nn.Parameter):
            module.data = module.data.to(device)

    class Catcher(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

    layer0_adapter = model_adapter.get_layers()[0]
    layer0_catcher = Catcher()

    if hasattr(layer0_adapter.layer, 'attention_type'):
        layer0_catcher.attention_type = layer0_adapter.layer.attention_type
    model_adapter.set_raw_layer_at(0, layer0_catcher)

    try:
        batch = map_tensors(batch, device=device)
        # extract targets key if present, otherwise empty dict
        targets = batch.pop('targets', None)
        model_adapter.model(**batch)
    except ValueError:
        pass

    # grab the inputs and caught arguments
    args = layer0_catcher.saved_args
    kwargs = layer0_catcher.saved_kwargs

    # put the caught stuff on cpu
    args = map_tensors(args, device='cpu')
    kwargs = map_tensors(kwargs, device='cpu')
    targets = map_tensors(targets, device='cpu')

    # put the layer back to normal
    model_adapter.set_raw_layer_at(0, layer0_adapter.layer)

    # Move embeddings back to cpu, and clear GPU cache.
    for module in model_adapter.get_embeddings():
        if isinstance(module, torch.nn.Module):
            for param in module.parameters():
                param.data = param.data.to('cpu')
        elif isinstance(module, torch.nn.Parameter):
            module.data = module.data.to('cpu')

    # Run GC and cleanup GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    return args[layer0_adapter.hidden_states_args_position], args, kwargs, targets


@torch.no_grad()
def get_layer_outputs(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]], device: torch.device | str
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    outputs = []

    layer_adapter.layer.to(device)

    for i, (layer_args_batch, layer_kwargs_batch) in tqdm(enumerate(zip(layer_args, layer_kwargs)), desc="Getting layer outputs", total=len(layer_args)):
        layer_args_batch, layer_kwargs_batch = map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=device
        )
        out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        if isinstance(out, tuple):
            out = out[layer_adapter.hidden_states_output_position]
        out = out.cpu()
        outputs.append(out)
        # torch.cuda.empty_cache()

    layer_adapter.layer.to('cpu')
    del layer_args_batch, layer_kwargs_batch, out
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


@torch.no_grad()
def get_head_outputs_with_comparison_metrics(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]],
    original_layer_adapter: LayerAdapter, original_layer_args: list[tuple], device: torch.device | str,
    prefix: str = None, return_outputs: bool = True, targets: list[torch.Tensor] = None
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Returns the output of the layer as well as computes comparison metrics = norm of difference and cosine distance
    """

    assert targets is not None, "Targets must be provided to compute perplexity metrics."

    outputs = []
    outputs_original = []

    # Move layers to device
    layer_adapter.layer.to(device)
    original_layer_adapter.layer.to(device)

    # Final output metrics
    mse_metric = MSEMetric()
    cosine_metric = CosineDistanceMetric()
    comparator_metric = NormComparatorMetric()

    if targets is not None:
        perplexity_metric_orig = PerplexityMetric()
        perplexity_metric = PerplexityMetric()

    # --- Prepare per-linear-layer metrics using hooks ---
    # Map submodule names to modules for both adapters
    def _linear_modules(module: torch.nn.Module):
        return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear) or isinstance(m, CompressedLinear)}

    mods = _linear_modules(layer_adapter.layer)
    mods_orig = _linear_modules(original_layer_adapter.layer)

    # Find common module names
    common_names = [n for n in mods.keys() if n in mods_orig]

    # Prepare metric objects per common linear module
    per_layer_mse = {n: MSEMetric() for n in common_names}
    per_layer_cos = {n: CosineDistanceMetric() for n in common_names}
    per_layer_comparator = {n: NormComparatorMetric() for n in common_names}

    # Buffers to hold outputs captured by hooks for the current forward
    buffer = {"mod": {}, "mod_orig": {}}

    # Register forward hooks
    hooks = []
    for name in common_names:
        mod = mods[name]
        mod_orig = mods_orig[name]

        def make_hook(key):
            def hook(module, input, output):
                # store the raw output tensor (detached)
                buffer["mod"][key] = output.detach()
            return hook

        def make_hook_orig(key):
            def hook(module, input, output):
                buffer["mod_orig"][key] = output.detach()
            return hook

        hooks.append(mod.register_forward_hook(make_hook(name)))
        hooks.append(mod_orig.register_forward_hook(make_hook_orig(name)))

    try:
        layer_kwargs_batch = layer_kwargs[0]  # assume same kwargs for all batches
        rebatch_size = 1 # embedding dimension can be large, use smaller batch size for rebatching
        num_samples = sum(sample[0].shape[0] for sample in layer_args)

        for i, (layer_args_batch, original_layer_args_batch, target_batch) in tqdm(enumerate(zip(rebatch_stream(layer_args, batch_size=rebatch_size), rebatch_stream(original_layer_args, batch_size=rebatch_size), rebatch_stream(targets, batch_size=rebatch_size))), desc="Getting head outputs", total=math.ceil(num_samples/rebatch_size)):
            # move tensors to device
            layer_kwargs_batch = layer_kwargs[0]
            layer_args_batch, layer_kwargs_batch = map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=device
            )

            original_layer_args_batch = map_tensors(
                original_layer_args_batch, device=device
            )

            target_batch = map_tensors(target_batch, device=device)

            # clear buffers
            buffer["mod"].clear()
            buffer["mod_orig"].clear()

            # Forward original first (hooks fill buffer[mod_orig])
            out_original = original_layer_adapter.layer(*original_layer_args_batch, **layer_kwargs_batch)
            if isinstance(out_original, tuple):
                out_original = out_original[original_layer_adapter.hidden_states_output_position]

            # Forward modified (hooks fill buffer[mod])
            out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]

            # Update per-layer metrics for all common linear modules that were captured
            for name in common_names:
                if name in buffer["mod"] and name in buffer["mod_orig"]:
                    y = buffer["mod"][name]
                    y_hat = buffer["mod_orig"][name]
                    per_layer_mse[name].update(y, y_hat)
                    per_layer_cos[name].update(y, y_hat)
                    per_layer_comparator[name].update(y, y_hat)

            # Update final-output metrics
            mse_metric.update(out, out_original)
            cosine_metric.update(out, out_original)
            comparator_metric.update(out, out_original)
            perplexity_metric.update(out, target_batch)
            perplexity_metric_orig.update(out_original, target_batch)

            if return_outputs:
                # Move outputs to cpu and store
                outputs.append(out.cpu())
                outputs_original.append(out_original.cpu())
            else:
                outputs.append(None)
                outputs_original.append(None)

            # free per-batch tensors
            del layer_args_batch, layer_kwargs_batch, original_layer_args_batch, target_batch
            torch.cuda.empty_cache()

    finally:
        # remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        # Move layers back to cpu
        layer_adapter.layer.to('cpu')
        original_layer_adapter.layer.to('cpu')

    # Compute and log final-output metrics
    mse, mse_std = mse_metric.compute()
    cosine, cosine_std = cosine_metric.compute()
    comparator, comparator_std = comparator_metric.compute()
    logger.info(f"Final output MSE: {mse} ± {mse_std}")
    logger.info(f"Final output Cosine Distance: {cosine} ± {cosine_std}")
    logger.info(f"Final output Norm Comparator: {comparator} ± {comparator_std}")

    if targets is not None:
        ppl = perplexity_metric.compute()
        ppl_orig = perplexity_metric_orig.compute()
        logger.info(f"Final output Perplexity: {ppl}")
        logger.info(f"Final output Original Perplexity: {ppl_orig}")

    # Compute and log per-layer metrics
    for name in common_names:
        mse_l, mse_l_std = per_layer_mse[name].compute()
        cos_l, cos_l_std = per_layer_cos[name].compute()
        comparator_l, comparator_l_std = per_layer_comparator[name].compute()
        logger.info(f"Layer {name} -> MSE: {mse_l} ± {mse_l_std}; Cosine: {cos_l} ± {cos_l_std}; Norm Comparator: {comparator_l} ± {comparator_l_std}")

    if wandb.run is not None:
        metrics = {"mse": mse,
                   "mse_std": mse_std,
                   "cosine_distance": cosine,
                   "cosine_distance_std": cosine_std,
                   "norm_comparator": comparator,
                   "norm_comparator_std": comparator_std,
                   "perplexity": ppl if targets is not None else None,
                   "perplexity_original": ppl_orig if targets is not None else None
                   }

        to_log = {}
        for k, v in metrics.items():
            log_key, layer_idx = get_log_key(prefix, k)
            to_log[log_key] = v
        to_log.update({'layer_idx': int(layer_idx)})
        wandb.log(to_log)

        # log per-layer metrics
        for name in common_names:
            layer_name = prefix + name
            mse_l, mse_l_std = per_layer_mse[name].compute()
            cos_l, cos_l_std = per_layer_cos[name].compute()
            comparator_l, comparator_l_std = per_layer_comparator[name].compute()

            metrics = {"mse": mse_l,
                   "mse_std": mse_l_std,
                   "cosine_distance": cos_l,
                   "cosine_distance_std": cos_l_std,
                   "norm_comparator": comparator_l,
                   "norm_comparator_std": comparator_l_std
                   }

            to_log = {}
            for k, v in metrics.items():
                log_key, layer_idx = get_log_key(layer_name, k)
                to_log[log_key] = v
            to_log.update({'layer_idx': int(layer_idx)})
            wandb.log(to_log)

    # cleanup
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, outputs_original


@torch.no_grad()
def get_layer_outputs_with_comparison_metrics(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]],
    original_layer_adapter: LayerAdapter, original_layer_args: list[tuple], device: torch.device | str,
    prefix: str = None, idx_log_delta=None
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Returns the output of the layer as well as computes comparison metrics = norm of difference and cosine distance
    """
    outputs = []
    outputs_original = []

    # Move layers to device
    layer_adapter.layer.to(device)
    original_layer_adapter.layer.to(device)

    # Final output metrics
    mse_metric = MSEMetric()
    cosine_metric = CosineDistanceMetric()
    comparator_metric = NormComparatorMetric()

    # For comparing fine-tuning using CE
    perplexity_metric = PerplexityMetric()

    # --- Prepare per-linear-layer metrics using hooks ---
    # Map submodule names to modules for both adapters
    def _linear_modules(module: torch.nn.Module):
        return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear) or isinstance(m, CompressedLinear)}

    mods = _linear_modules(layer_adapter.layer)
    mods_orig = _linear_modules(original_layer_adapter.layer)

    # Find common module names
    common_names = [n for n in mods.keys() if n in mods_orig]

    # Prepare metric objects per common linear module
    per_layer_mse = {n: MSEMetric() for n in common_names}
    per_layer_cos = {n: CosineDistanceMetric() for n in common_names}
    per_layer_comparator = {n: NormComparatorMetric() for n in common_names}

    # Buffers to hold outputs captured by hooks for the current forward
    buffer = {"mod": {}, "mod_orig": {}}

    # Register forward hooks
    hooks = []
    for name in common_names:
        mod = mods[name]
        mod_orig = mods_orig[name]

        def make_hook(key):
            def hook(module, input, output):
                # store the raw output tensor (detached)
                buffer["mod"][key] = output.detach()
            return hook

        def make_hook_orig(key):
            def hook(module, input, output):
                buffer["mod_orig"][key] = output.detach()
            return hook

        hooks.append(mod.register_forward_hook(make_hook(name)))
        hooks.append(mod_orig.register_forward_hook(make_hook_orig(name)))

    try:
        for i, (layer_args_batch, layer_kwargs_batch, original_layer_args_batch) in tqdm(enumerate(zip(layer_args, layer_kwargs, original_layer_args)), desc="Getting layer outputs", total=len(layer_args)):
            # move tensors to device
            layer_args_batch, layer_kwargs_batch = map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=device
            )

            original_layer_args_batch = map_tensors(
                original_layer_args_batch, device=device
            )

            # clear buffers
            buffer["mod"].clear()
            buffer["mod_orig"].clear()

            # Forward original first (hooks fill buffer[mod_orig])
            out_original = original_layer_adapter.layer(*original_layer_args_batch, **layer_kwargs_batch)
            if isinstance(out_original, tuple):
                out_original = out_original[original_layer_adapter.hidden_states_output_position]

            # Forward modified (hooks fill buffer[mod])
            out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]

            # Update per-layer metrics for all common linear modules that were captured
            for name in common_names:
                if name in buffer["mod"] and name in buffer["mod_orig"]:
                    y = buffer["mod"][name]
                    y_hat = buffer["mod_orig"][name]
                    per_layer_mse[name].update(y, y_hat)
                    per_layer_cos[name].update(y, y_hat)
                    per_layer_comparator[name].update(y, y_hat)

            # Update final-output metrics
            mse_metric.update(out, out_original)
            cosine_metric.update(out, out_original)
            comparator_metric.update(out, out_original)

            perplexity_metric.update(out, out_original)

            # Move outputs to cpu and store
            outputs.append(out.cpu())
            outputs_original.append(out_original.cpu())

            # free per-batch tensors
            del layer_args_batch, layer_kwargs_batch, original_layer_args_batch
            torch.cuda.empty_cache()

    finally:
        # remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        # Move layers back to cpu
        layer_adapter.layer.to('cpu')
        original_layer_adapter.layer.to('cpu')

    # Compute and log final-output metrics
    mse, mse_std = mse_metric.compute()
    cosine, cosine_std = cosine_metric.compute()
    comparator, comparator_std = comparator_metric.compute()
    logger.info(f"Final output MSE: {mse} ± {mse_std}")
    logger.info(f"Final output Cosine Distance: {cosine} ± {cosine_std}")
    logger.info(f"Final output Norm Comparator: {comparator} ± {comparator_std}")

    ce = perplexity_metric.compute(exp=False)
    logger.info(f"Final output ce: {ce}")

    # Compute and log per-layer metrics
    for name in common_names:
        mse_l, mse_l_std = per_layer_mse[name].compute()
        cos_l, cos_l_std = per_layer_cos[name].compute()
        comparator_l, comparator_l_std = per_layer_comparator[name].compute()
        logger.info(f"Layer {name} -> MSE: {mse_l} ± {mse_l_std}; Cosine: {cos_l} ± {cos_l_std}; Norm Comparator: {comparator_l} ± {comparator_l_std}")

    if wandb.run is not None:
        metrics = {"mse": mse,
                   "mse_std": mse_std,
                   "cosine_distance": cosine,
                   "cosine_distance_std": cosine_std,
                   "norm_comparator": comparator,
                   "norm_comparator_std": comparator_std,
                   "ce": ce
                   }

        to_log = {}
        for k, v in metrics.items():
            log_key, layer_idx = get_log_key(prefix, k)
            to_log[log_key] = v
        to_log.update({'layer_idx': int(layer_idx) if idx_log_delta is None else int(layer_idx) + idx_log_delta})
        wandb.log(to_log)

        # log per-layer metrics
        for name in common_names:
            layer_name = prefix + name
            mse_l, mse_l_std = per_layer_mse[name].compute()
            cos_l, cos_l_std = per_layer_cos[name].compute()
            comparator_l, comparator_l_std = per_layer_comparator[name].compute()

            metrics = {
                    "mse": mse_l,
                    "mse_std": mse_l_std,
                    "cosine_distance": cos_l,
                    "cosine_distance_std": cos_l_std,
                    "norm_comparator": comparator_l,
                    "norm_comparator_std": comparator_l_std
                    }

            to_log = {}
            for k, v in metrics.items():
                log_key, layer_idx = get_log_key(layer_name, k)
                to_log[log_key] = v
            to_log.update({'layer_idx': int(layer_idx) if idx_log_delta is None else int(layer_idx) + idx_log_delta})
            wandb.log(to_log)

    # cleanup
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, outputs_original


class GramMatrixContainer:
    """Container to hold gram matrix that can be passed by reference."""
    def __init__(self):
        self.matrix = None
        self.num_samples = 0

    def add(self, x: torch.Tensor, xhat: torch.Tensor = None):
        """Add x.t() @ x to the accumulated gram matrix."""
        if xhat is None:
            # cast to float32 for numerical stability
            x_hp = x.float()
            gram = x_hp.t() @ x_hp
        else:
            x_hp, xhat_hp = x.float(), xhat.float()

            # # compute norm of x_hp and xhat_hp, per sample, and choose the larger one for first position
            # norm_x = x_hp.norm(dim=1)
            # norm_xhat = xhat_hp.norm(dim=1)
            # mask = norm_x <= norm_xhat
            # x_hp = torch.where(mask.unsqueeze(1), xhat_hp, x_hp)

            gram = x_hp.t() @ xhat_hp

        current_num_samples = x.shape[0]
        self.num_samples += current_num_samples

        if self.matrix is None:
            self.matrix = gram/current_num_samples
        else:
            self.matrix = self.matrix*((self.num_samples - current_num_samples)/self.num_samples) + gram/self.num_samples

    def compute_stats(self, key, prefix=None):
        assert self.matrix is not None, "Gram matrix is None, cannot compute stats."
        eigenvalues = torch.linalg.eigvalsh(self.matrix)
        stats = {
            "energy": eigenvalues.sum().item(),
            "max_eigenvalue": eigenvalues.max().item(),
            "min_eigenvalue": eigenvalues.min().item(),
            "condition_number": (eigenvalues.max() / eigenvalues.min()).item()
        }
        logger.info(f"Gram matrix stats for {key}: {stats}")

        if wandb.run is not None:
            to_log = {}
            for k, v in stats.items():
                log_key, layer_idx = get_log_key(key, f"{prefix}_{k}" if prefix is not None else k)
                to_log[log_key] = v
            to_log.update({'layer_idx': int(layer_idx)})
            wandb.log(to_log)


@torch.no_grad()
def collect_gram_matrix_parallel(
                        layer_adapter: LayerAdapter,
                        module: torch.nn.Module,
                        layer_args: list[tuple],
                        layer_kwargs: list[dict[str, Any]],
                        original_layer_adapter: LayerAdapter,
                        original_module: torch.nn.Module,
                        original_layer_args: list[tuple],
                        device: torch.device | str,
                        name: str
                    ):
    layer_adapter.layer.to(device)
    original_layer_adapter.layer.to(device)

    gram_x = GramMatrixContainer()
    gram_xorig = GramMatrixContainer()
    gram_cross = GramMatrixContainer()

    buffer = {"x": None, "x_orig": None}

    # --- Hooks ---
    def hook(module, input):
        buffer["x"] = input[0].view(-1, input[0].shape[-1])
        raise ValueError

    def hook_original(module, input):
        buffer["x_orig"] = input[0].view(-1, input[0].shape[-1])
        raise ValueError

    hook1 = module.register_forward_pre_hook(hook)
    hook2 = original_module.register_forward_pre_hook(hook_original)

    try:
        for (args, kwargs), (args_original) in tqdm(
            zip(zip(layer_args, layer_kwargs),
                original_layer_args),
            total=len(layer_args),
            desc="Computing Dual Gram Matrix"
        ):

            args, kwargs = map_tensors([args, kwargs], device=device)
            args_original = map_tensors(args_original, device=device)

            # --- Forward original (will exit early) ---
            try:
                layer_adapter.layer(*args, **kwargs)
            except ValueError:
                pass

            # --- Forward modified (will exit early) ---
            try:
                original_layer_adapter.layer(*args_original, **kwargs)
            except ValueError:
                pass

            # Now we have both buffers for this batch
            gram_x.add(buffer["x"])
            gram_xorig.add(buffer["x_orig"])
            gram_cross.add(buffer["x_orig"], buffer["x"])

            buffer["x"] = buffer["x_orig"] = None

            del args, kwargs, args_original
            torch.cuda.empty_cache()

    finally:
        hook1.remove()
        hook2.remove()

        for g, prefix in zip((gram_x, gram_xorig, gram_cross), ("x", "x_orig", "cross")):
            if g.matrix is not None:
                # g.compute_stats(key=name, prefix=prefix)
                g.matrix = g.matrix.cpu()

        layer_adapter.layer.to("cpu")
        original_layer_adapter.layer.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    return gram_x.matrix, gram_cross.matrix, gram_xorig.matrix


@torch.no_grad()
def collect_gram_matrix(
    layer_adapter: LayerAdapter,
    module: torch.nn.Module,
    layer_args: list[tuple],
    layer_kwargs: list[dict[str, Any]],
    device: torch.device | str,
    name: str,
    prefix: str = None):
    # apply pre forward hook to module which computes x.t()@x for the input batch and collects the sum, stops the forward iteration and returns it

    # put layer to device
    layer_adapter.layer.to(device)

    # Create container object that will be passed by reference
    gram_container = GramMatrixContainer()

    def hook_fn(module, input):
        x = input[0].view(-1, input[0].shape[-1])
        # Add to gram matrix using the container
        gram_container.add(x)
        # stop forward iteration by raising an exception
        raise ValueError("Hook completed successfully")

    hook = module.register_forward_pre_hook(hook_fn)

    try:
        for i, (layer_args_batch, layer_kwargs_batch) in tqdm(enumerate(zip(layer_args, layer_kwargs)), desc="Computing gram matrix", total=len(layer_args)):
            layer_args_batch, layer_kwargs_batch = map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=device
            )
            try:
                layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
            except ValueError as e:
                # Check if this is our expected exception
                if "Hook completed successfully" in str(e):
                    pass  # This is expected
                else:
                    raise  # Re-raise unexpected errors
            finally:
                # Clean up batch tensors immediately
                del layer_args_batch, layer_kwargs_batch
                torch.cuda.empty_cache()

    finally:
        hook.remove()

        # Move gram_matrix to CPU if it exists
        if gram_container.matrix is not None:
            # compute stats
            # gram_container.compute_stats(key=name, prefix=prefix)
            gram_container.matrix = gram_container.matrix.cpu()

        # put layer to cpu
        layer_adapter.layer.to('cpu')

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

    return gram_container.matrix


def apply_compression_parallel(config: Dict, model: torch.nn.Module, modules_to_replace, calibration_dataloader, test_calibration_dataloader, allocations, device) -> torch.nn.Module:
    save_path = getattr(config, 'save_path', None) # Path to save the compressed model
    dobi_remapping = getattr(config, 'dobi_remapping', False)
    finetune_cfg = getattr(config, 'finetune', defaultdict(lambda: None))
    finetune_layers = finetune_cfg['enabled']

    sub_method = getattr(config, 'sub_method', None)

    adapter_cls = MODEL_ADAPTER_REGISTRY.get(type(model))
    if adapter_cls is None:
        raise ValueError(f"No adapter registered for model type {type(model).__name__}. "
                         "Add it to aa_svd/compression/adapters/__init__.py.")
    model_adapter = adapter_cls(model, modules_to_replace)

    # model_adapter.load(save_path)

    # Get first layer inputs from calibration data
    inps, args, kwargs, targets = [], [], [], []
    for batch in tqdm(calibration_dataloader, desc="Processing layer 0 inputs", total=len(calibration_dataloader)):
        inp_batch, args_batch, kwargs_batch, targets_batch = get_layer0_inputs(model_adapter, batch, device='cuda')
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        targets.append(targets_batch)
    inps_hat = inps

    # test_calibration_dataloader = None # temporarily disable test calibration

    if test_calibration_dataloader is not None:
        # Get first layer inputs from test calibration data
        test_inps, test_args, test_kwargs, test_targets = [], [], [], []
        for batch in tqdm(test_calibration_dataloader, desc="Processing test layer 0 inputs", total=len(test_calibration_dataloader)):
            inp_batch, args_batch, kwargs_batch, targets_batch = get_layer0_inputs(model_adapter, batch, device='cuda')
            test_inps.append(inp_batch)
            test_args.append(args_batch)
            test_kwargs.append(kwargs_batch)
            test_targets.append(targets_batch)
        test_inps_hat = test_inps

    head_adapter = model_adapter.get_last_layer_to_output_adapter()
    layers = model_adapter.get_layers() + [head_adapter]

    # Compress each layer
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Processing layers")):
        logger.info(f"Compressing layer {idx}")
        layer_name_prefix = model_adapter.get_layer_name_prefix(idx)

        layer_adapter_clone = layer_adapter.clone()
        layer_adapter_clone.layer.to(device)

        not_finetuned = model_adapter.load(save_path, layer_adapter=layer_adapter, layer_idx=idx, dobi_remapping=dobi_remapping)

        args_hat = []
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])
            args_hat.append(layer_adapter_clone.get_updated_args(inps_hat[i], args[i]))

        if test_calibration_dataloader is not None:
            test_args_hat = []
            for i, inp in enumerate(test_inps):
                test_args[i] = layer_adapter.get_updated_args(inp, test_args[i])
                test_args_hat.append(layer_adapter_clone.get_updated_args(test_inps_hat[i], test_args[i]))

        for module, module_clone in zip(layer_adapter.get_compression_order(), layer_adapter_clone.get_compression_order()):
            name = layer_name_prefix + layer_adapter.get_module_name(module)
            xhatTxhat, xTxhat, xTx = collect_gram_matrix_parallel(layer_adapter, module, args, kwargs, layer_adapter_clone, module_clone, args_hat, device, name)

            if test_calibration_dataloader is not None:
                test_xhatTxhat, test_xTxhat, test_xTx = collect_gram_matrix_parallel(layer_adapter, module, test_args, test_kwargs, layer_adapter_clone, module_clone, test_args_hat, device, name+'_test')
                logger.info(f"Test Gram matrix shapes: xhatTxhat: {test_xhatTxhat.shape}, xTxhat: {test_xTxhat.shape}, xTx: {test_xTx.shape}")

            # Get the mapping of modules to replace for this compression source module
            for target_module in layer_adapter.get_compression_mapping()[module]:
                target_module_name = layer_name_prefix + layer_adapter.get_module_name(target_module)

                logger.info(f"Compressing module {target_module_name}")

                if sub_method in ['obj1', 'svd']:
                    compressed_module = compress_module_obj1(target_module, ratio=allocations[target_module_name], device=device, dobi_remapping=dobi_remapping)
                elif sub_method in ['obj2', 'svd-llm']:
                    compressed_module = compress_module_obj2(target_module, xTx, ratio=allocations[target_module_name], device=device, dobi_remapping=dobi_remapping)
                elif sub_method == 'svd-llm-v2':
                    compressed_module = compress_module_obj2_evd(target_module, xTx, ratio=allocations[target_module_name], device=device, dobi_remapping=dobi_remapping)
                elif sub_method == 'obj3':
                    compressed_module = compress_module_obj3(target_module, xTx, xTxhat, xhatTxhat, ratio=allocations[target_module_name], device=device, dobi_remapping=dobi_remapping)
                    # compressed_module = compress_module_obj2(target_module, xhatTxhat, ratio=allocations[target_module_name], device=device, dobi_remapping=dobi_remapping)
                elif sub_method == 'obj4':
                    compressed_module = compress_module_obj4(target_module, xTx, xTxhat, xhatTxhat, ratio=allocations[target_module_name], device=device, dobi_remapping=dobi_remapping)

                if save_path is not None:
                    compressed_module.save(os.path.join(save_path, target_module_name.replace('.', '_')))
                compressed_module.log_metrics(target_module.weight.data, name=target_module_name, device=device)
                compressed_module.log_calibration_metrics(target_module.weight.data, xTx, xhatTxhat, xTxhat, name=target_module_name, device=device)

                logger.info(f"Replacing module {target_module_name} with compressed version")
                layer_adapter.replace_module(target_module, compressed_module)

        if idx != len(layers) - 1:
            inps, inps_hat = get_layer_outputs_with_comparison_metrics(layer_adapter, args, kwargs,
                                                                    layer_adapter_clone, args_hat, device='cuda',
                                                                    prefix=layer_name_prefix)

            if test_calibration_dataloader is not None:
                test_inps, test_inps_hat = get_layer_outputs_with_comparison_metrics(layer_adapter, test_args, test_kwargs,
                                                                    layer_adapter_clone, test_args_hat, device='cuda',
                                                                    prefix='_test_' + layer_name_prefix)
            if finetune_layers and not_finetuned:
                logger.info(f"Fine-tuning layer {layer_name_prefix}")

                if finetune_cfg['include'] == 'all':
                    param_names_to_finetune = [name for name, param in layer_adapter.layer.named_parameters()]
                elif isinstance(finetune_cfg['include'], list):
                    param_names_to_finetune = [name for name, param in layer_adapter.layer.named_parameters() if any(cand_name in name for cand_name in finetune_cfg['include'])]
                else:
                    raise ValueError

                for name, param in layer_adapter.layer.named_parameters():
                    if name not in param_names_to_finetune:
                        param.requires_grad = False
                        logger.info(f"Ignoring parameter {name} of shape {param.shape} in layer {layer_name_prefix}")
                    else:
                        logger.info(f"Finetuning parameter {name} of shape {param.shape} in layer {layer_name_prefix}")

                model_adapter.finetune_layer(layer_adapter, args, kwargs, inps_hat,
                                            device='cuda', layer_idx=idx, num_steps=25, lr=1e-4, loss_type='mse')

                # save each module after fine-tuning
                if save_path is not None:
                    for module_name, module in layer_adapter.layer.named_modules():
                        if isinstance(module, CompressedLinear):
                            target_module_name = layer_name_prefix + layer_adapter.get_module_name(module)
                            module.save(os.path.join(save_path, target_module_name.replace('.', '_')))
                            logger.info(f"Saved fine-tuned module {target_module_name} at {os.path.join(save_path, target_module_name.replace('.', '_'))}")
                        elif 'norm' in module_name:
                            target_module_name = layer_name_prefix + module_name
                            os.makedirs(os.path.join(save_path, target_module_name.replace('.', '_')), exist_ok=True)
                            torch.save(module.state_dict(), os.path.join(save_path, target_module_name.replace('.', '_'), 'state_dict.pt'))
                            logger.info(f"Saved fine-tuned module {target_module_name} at {os.path.join(save_path, target_module_name.replace('.', '_'), 'state_dict.pt')}")

                inps, inps_hat = get_layer_outputs_with_comparison_metrics(layer_adapter, args, kwargs,
                                                                        layer_adapter_clone, args_hat, device='cuda',
                                                                        prefix=layer_name_prefix, idx_log_delta=0.5)
                if test_calibration_dataloader is not None:
                    test_inps, test_inps_hat = get_layer_outputs_with_comparison_metrics(layer_adapter, test_args, test_kwargs,
                                                                        layer_adapter_clone, test_args_hat, device='cuda',
                                                                        prefix='_test_' + layer_name_prefix, idx_log_delta=0.5)

            # Release old activation tensors and the layer clone before the next iteration
            # creates a new clone.  Without this, args/args_hat keep the previous layer's
            # inps/inps_hat tensors alive until lines 1647-1656 run at the TOP of the next
            # iteration — after the new clone is already allocated — causing a spike of
            # ~4× activation size (old inps + old inps_hat + new inps + new inps_hat).
            for i, new_inp in enumerate(inps):
                args[i] = layer_adapter.get_updated_args(new_inp, args[i])
            del args_hat
            if test_calibration_dataloader is not None:
                for i, new_inp in enumerate(test_inps):
                    test_args[i] = layer_adapter.get_updated_args(new_inp, test_args[i])
                del test_args_hat
            del layer_adapter_clone
            gc.collect()
        else:
            assert layer_adapter == head_adapter
            del inps, inps_hat
            torch.cuda.empty_cache()
            gc.collect()

            inps, inps_hat = get_head_outputs_with_comparison_metrics(layer_adapter, args, kwargs,
                                                                    layer_adapter_clone, args_hat, device='cuda',
                                                                    prefix=layer_name_prefix, return_outputs=False, targets=targets)

            if test_calibration_dataloader is not None:
                del test_inps, test_inps_hat
                torch.cuda.empty_cache()
                gc.collect()
                test_inps, test_inps_hat = get_head_outputs_with_comparison_metrics(layer_adapter, test_args, test_kwargs,
                                                                    layer_adapter_clone, test_args_hat, device='cuda',
                                                                    prefix='_test_' + layer_name_prefix, return_outputs=False, targets=test_targets)

    model = model_adapter.model

    if dobi_remapping:
        logger.info("Applying Dobi-SVD style remapping + quantization to all compressed modules")
        for name, module in model.named_modules():
            if isinstance(module, QuantizedCompressedLinear):
                logger.info(f"Quantizing module {name}")
                module.apply_quantization()

    return model

