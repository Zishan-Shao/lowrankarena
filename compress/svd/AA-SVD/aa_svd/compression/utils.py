import logging
from collections.abc import Generator
from typing import List, Tuple, TypeVar, Union

import torch

logger = logging.getLogger(__name__)


def rebatch_stream(
    input_args: List[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    batch_size: int,
) -> Generator:
    """Streaming re-batcher that yields fixed-size batches without materializing the full dataset.

    Accepts either a list of tensors or a list of tuples of tensors.
    Yields the same element type as the input (tensor or tuple).
    The last yielded batch may be smaller than batch_size.
    """
    first = input_args[0]
    input_is_tuple = isinstance(first, tuple)

    def wrap(batch):
        return batch if input_is_tuple else (batch,)

    buffers = None
    filled = 0

    for batch in input_args:
        batch_tuple = wrap(batch)
        B = batch_tuple[0].shape[0]

        # Fast path: batch already the right size and nothing buffered
        if B == batch_size and filled == 0:
            yield batch
            continue

        if buffers is None:
            buffers = [[] for _ in batch_tuple]

        offset = 0
        while offset < B:
            take = min(batch_size - filled, B - offset)
            for i, tensor in enumerate(batch_tuple):
                buffers[i].append(tensor[offset: offset + take])
            filled += take
            offset += take

            if filled == batch_size:
                out_tuple = tuple(torch.cat(buf, dim=0) for buf in buffers)
                yield out_tuple if input_is_tuple else out_tuple[0]
                buffers = [[] for _ in batch_tuple]
                filled = 0

    if filled > 0:
        out_tuple = tuple(torch.cat(buf, dim=0) for buf in buffers)
        yield out_tuple if input_is_tuple else out_tuple[0]


def get_submodule(model: torch.nn.Module, target_name: str) -> torch.nn.Module:
    """Return the submodule of *model* at the dot-separated path *target_name*."""
    current_module = model
    for token in target_name.split('.'):
        if hasattr(current_module, token):
            current_module = getattr(current_module, token)
        else:
            raise ValueError(f"Module '{token}' not found in path '{target_name}'")
    return current_module


def replace_module(
    model: torch.nn.Module,
    target_name: str,
    new_module: torch.nn.Module,
) -> None:
    """Replace the submodule at *target_name* in *model* with *new_module*."""
    tokens = target_name.split('.')
    parent_name = '.'.join(tokens[:-1])
    child_name = tokens[-1]
    parent = get_submodule(model, parent_name) if parent_name else model
    setattr(parent, child_name, new_module)


def get_log_key(layer_name: str, metric_name: str) -> Tuple[str, int]:
    """Return a (wandb_key, layer_index) pair for logging a per-layer metric.

    Handles both normal layers (path contains 'layers' or 'blocks') and
    head/classifier modules.  Layer names prefixed with '_test_' are logged
    under a separate 'test_*' namespace.
    """
    is_test = layer_name.startswith('_test_')

    if ('head' in layer_name or 'classifier' in layer_name) and (
        'layer' not in layer_name and 'block' not in layer_name
    ):
        return f'{metric_name}/{layer_name}', 0

    parts = layer_name.split('.')
    try:
        idx_layers = parts.index('layers')
    except ValueError:
        idx_layers = parts.index('blocks')

    layer_idx = int(parts[idx_layers + 1])
    try:
        layer_type = '.'.join(parts[idx_layers + 2:]) or 'layer'
    except IndexError:
        layer_type = 'layer'

    prefix = 'test_' if is_test else ''
    if layer_type == 'layer':
        return f'{prefix}{layer_type}/{metric_name}', layer_idx
    return f'{prefix}{metric_name}/{layer_type}', layer_idx


T = TypeVar('T')


def map_tensors(
    obj: T,
    device: Union[torch.device, str, None] = None,
    dtype: Union[torch.dtype, None] = None,
) -> T:
    """Recursively move all tensors in *obj* to *device* and/or *dtype*.

    Supports tensors, lists, tuples, and dicts; other types are returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    if isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore[return-value]
    return obj


@torch.no_grad()
def safe_cholesky(A: torch.Tensor) -> torch.Tensor:
    """Cholesky decomposition with fp32 upcasting for numerical stability.

    If the matrix is not positive definite, a small diagonal correction derived
    from the minimum eigenvalue is added before retrying.
    """
    original_dtype = A.dtype
    if original_dtype != torch.float64:
        A = A.to(torch.float32)
    try:
        L = torch.linalg.cholesky(A)
    except Exception:
        logger.warning("Matrix not positive definite; adding small diagonal correction for stability")
        eigenvalues = torch.linalg.eigvalsh(A)
        A = A + (-eigenvalues[0] + 1e-3) * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        L = torch.linalg.cholesky(A)
    if original_dtype != torch.float64:
        L = L.to(original_dtype)
    return L


def safe_svd(
    A: torch.Tensor,
    full_matrices: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD with fp32 upcasting for numerical stability.

    Returns (U, S, Vh) matching the convention of torch.linalg.svd.
    """
    original_dtype = A.dtype
    if original_dtype != torch.float64:
        A = A.to(torch.float32)
    try:
        U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    except RuntimeError as e:
        logger.error(f"SVD failed: {e}")
        raise ValueError("SVD decomposition failed") from e
    if original_dtype != torch.float64:
        U = U.to(original_dtype)
        S = S.to(original_dtype)
        Vh = Vh.to(original_dtype)
    return U, S, Vh


def safe_inv(A: torch.Tensor) -> torch.Tensor:
    """Matrix inverse with fp32 upcasting for numerical stability."""
    original_dtype = A.dtype
    if original_dtype != torch.float64:
        A = A.to(torch.float32)
    try:
        A_inv = torch.linalg.inv(A)
    except RuntimeError as e:
        logger.error(f"Matrix inversion failed: {e}")
        raise ValueError("Matrix inversion failed") from e
    if original_dtype != torch.float64:
        A_inv = A_inv.to(original_dtype)
    return A_inv


def safe_eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition of a symmetric/Hermitian matrix with fp32 upcasting.

    Returns (eigenvalues, eigenvectors) matching the convention of torch.linalg.eigh,
    with eigenvalues in ascending order.
    """
    original_dtype = A.dtype
    if original_dtype != torch.float64:
        A = A.to(torch.float32)
    try:
        S2, Vxp = torch.linalg.eigh(A)
    except RuntimeError as e:
        logger.error(f"Eigendecomposition failed: {e}")
        raise ValueError("Eigendecomposition failed") from e
    if original_dtype != torch.float64:
        S2 = S2.to(original_dtype)
        Vxp = Vxp.to(original_dtype)
    return S2, Vxp
