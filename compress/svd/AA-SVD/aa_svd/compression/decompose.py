import logging

import torch

from .utils import safe_cholesky, safe_svd, safe_eigh, safe_inv
from .compressed_linear import CompressedLinear, QuantizedCompressedLinear

logger = logging.getLogger(__name__)


def get_rank_from_ratio(
    target: torch.Tensor, ratio: float, dobi_remapping=False
) -> int:
    """
    Compute the low-rank approximation rank for a weight matrix given a
    parameter ratio.

    Args:
        target: Weight matrix.
        ratio: Target fraction of parameters to retain.
        dobi_remapping: If True, uses max(dim) in the denominator.
    Returns:
        Integer rank.
    """
    if ratio <= 0 or ratio is None:
        raise ValueError(f"Invalid ratio {ratio}. Must be greater than 0.")
    r, c = target.shape[0], target.shape[1]
    if dobi_remapping:
        return int(ratio * r * c / max(r, c))
    return int(ratio * r * c / (r + c))


@torch.no_grad()
def compress_module_obj1(module, ratio, device, dobi_remapping=False):
    """
    Vanilla SVD compression: low-rank approximation of the weight matrix.
    Solves min(norm(W - W')).
    Corresponds to Objective 1 in the paper.
    """
    weight = module.weight.data.clone().to(device)
    rank = (
        min(weight.shape) if ratio >= 1.0
        else get_rank_from_ratio(weight, ratio, dobi_remapping)
    )
    logger.info(
        f"Solving Obj 1: shape={weight.shape}, device={weight.device}, rank={rank}"
    )
    Uws, Sws, Vws = safe_svd(weight, full_matrices=False)
    sq = torch.sqrt(Sws[:rank])
    U = Uws[:, :rank] * sq
    V = Vws[:rank, :].T * sq
    compressed = CompressedLinear.from_linear(
        U=U, S=None, V=V, bias=module.bias
    ).to(device)
    del Uws, Sws, Vws, weight
    torch.cuda.empty_cache()
    return compressed


@torch.no_grad()
def compress_module_obj2_evd(module, xTx, ratio, device, dobi_remapping=False):
    """
    Similar to SVD-LLM V2: whitens weight via EVD/SVD factors of the activation covariance.
    Solves min(norm(WX - W'X)).
    Corresponds to Objective 2 in the paper.
    """

    weight = module.weight.data.clone().to(device)
    original_dtype = weight.dtype
    weight = weight.to(xTx.dtype)
    rank = (
        min(weight.shape) if ratio >= 1.0
        else get_rank_from_ratio(weight, ratio, dobi_remapping)
    )
    xTx = xTx.to(device)
    Ss, Us = safe_eigh(xTx)
    D = weight @ (Us * torch.sqrt(Ss))
    logger.info(
        f"Solving Obj 2 (svd): shape={D.shape}, device={D.device}, rank={rank}"
    )
    Uws, Sws, Vws = safe_svd(D, full_matrices=False)
    U = (Uws[:, :rank] * Sws[:rank]).to(original_dtype)
    V = ((Vws[:rank, :] * (1 / torch.sqrt(Ss))) @ Us.T).T.to(original_dtype)

    cls = (
        QuantizedCompressedLinear if dobi_remapping
        else CompressedLinear
    )
    compressed = cls.from_linear(U=U, S=None, V=V, bias=module.bias).to(device)
    del Uws, Sws, Vws, D, Ss, Us, weight
    torch.cuda.empty_cache()
    return compressed


@torch.no_grad()
def compress_module_obj2(module, xTx, ratio, device, dobi_remapping=False):
    """
    Similar to SVD-LLM: whitens weight via Cholesky of the activation covariance.
    Solves min(norm(WX - W'X)).
    Corresponds to Objective 2 in the paper.
    """
    weight = module.weight.data.clone().to(device)
    original_dtype = weight.dtype
    weight = weight.to(xTx.dtype)
    rank = (
        min(weight.shape) if ratio >= 1.0
        else get_rank_from_ratio(weight, ratio, dobi_remapping)
    )
    xTx = xTx.to(device)
    S = safe_cholesky(xTx)
    S_inv = safe_inv(S)
    weight_s = torch.matmul(weight, S)
    logger.info(
        f"Solving Obj 2 (chol): shape={weight_s.shape}, device={weight_s.device}, "
        f"rank={rank}"
    )
    U, sigma, V = safe_svd(weight_s, full_matrices=False)
    sq = torch.sqrt(sigma[:rank])
    U = (U[:, :rank] * sq).to(original_dtype)
    V = torch.matmul((V[:rank, :].t() * sq).t(), S_inv).t().to(original_dtype)

    cls = (
        QuantizedCompressedLinear if dobi_remapping
        else CompressedLinear
    )
    compressed = cls.from_linear(U=U, S=None, V=V, bias=module.bias).to(device)
    del S, S_inv, weight, U, sigma, V, weight_s
    torch.cuda.empty_cache()
    return compressed


@torch.no_grad()
def compress_module_obj3(
    module, xTx, xTxhat, xhatTxhat, ratio, device,
    dobi_remapping=False,
):
    """
    Similar to Dobi-SVD
    Solves min(norm(WX' - W'X')).
    Corresponds to Objective 3 in the paper.
    """
    return _compress_module_obj34(
        module, xTx, xTxhat, xhatTxhat, ratio, device, alpha=0., dobi_remapping=dobi_remapping)


@torch.no_grad()
def compress_module_obj4(
    module, xTx, xTxhat, xhatTxhat, ratio, device,
    dobi_remapping=False,
):
    """
    AA-SVD
    Solves min(norm(WX - W'X')).
    Corresponds to Objective 4 in the paper.
    """
    return _compress_module_obj34(
        module, xTx, xTxhat, xhatTxhat, ratio, device, alpha=1., dobi_remapping=dobi_remapping)


@torch.no_grad()
def _compress_module_obj34(
    module, xTx, xTxhat, xhatTxhat, ratio, device, alpha=1.0,
    dobi_remapping=False,
):
    weight = module.weight.data.clone().to(device)
    original_dtype = weight.dtype
    weight = weight.to(xhatTxhat.dtype)
    rank = (
        min(weight.shape) if ratio >= 1.0
        else get_rank_from_ratio(weight, ratio, dobi_remapping)
    )
    xhatTxhat = xhatTxhat.to(device)
    xTxhat = xTxhat.to(device)
    xTx = xTx.to(device)

    L = safe_cholesky(xhatTxhat)
    L_inv_T = safe_inv(L.T)

    mixed = alpha * xTxhat + (1 - alpha) * xhatTxhat
    if ratio >= 1.0:
        W_tilde = torch.linalg.solve(xhatTxhat, weight @ mixed, left=False)
    else:
        W_tilde = weight @ mixed @ L_inv_T

    if alpha == 0.0:
        logger.info(
            f"Solving Obj 3: shape={W_tilde.shape}, device={W_tilde.device}, "
            f"rank={rank}"
        )
    elif alpha == 1.0:
        logger.info(
            f"Solving Obj 4: shape={W_tilde.shape}, device={W_tilde.device}, rank={rank}"
        )
    else:
        logger.info(
            f"Mixed Solving Obj 3-4: shape={W_tilde.shape}, device={W_tilde.device}, rank={rank}"
        )
    U, Sigma, Vt = safe_svd(W_tilde, full_matrices=False)
    sq = torch.sqrt(Sigma[:rank])
    U = (U[:, :rank] * sq).to(original_dtype)
    if ratio >= 1.0:
        V = (torch.diag(sq) @ Vt[:rank, :]).t().to(original_dtype)
    else:
        V = (torch.diag(sq) @ Vt[:rank, :] @ L_inv_T.T).t().to(original_dtype)

    cls = (
        QuantizedCompressedLinear if dobi_remapping
        else CompressedLinear
    )
    compressed = cls.from_linear(U=U, S=None, V=V, bias=module.bias).to(device)
    torch.cuda.empty_cache()
    return compressed

