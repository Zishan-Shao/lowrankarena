"""
Utility functions and classes for SVD-LLM compression with scalar correction.

This module contains helper functions, data classes, and utilities that are
shared across the main compression scripts.
"""

import heapq
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.model_utils import (
    LowRankLinear,
    get_layers,
    find_target_modules_in_layer,
    find_layers,
)

# Optional bitsandbytes for remap fake quant
try:
    import bitsandbytes as bnb
except Exception:
    bnb = None

_BNB_CODE_CACHE = {}


def _get_bnb_code(dev: torch.device):
    """Get or create cached bitsandbytes dynamic map code for a device."""
    if bnb is None:
        return None
    if dev.type != "cuda":
        return None
    idx = dev.index if dev.index is not None else 0
    if idx not in _BNB_CODE_CACHE:
        _BNB_CODE_CACHE[idx] = bnb.functional.create_dynamic_map().to(dev)
    return _BNB_CODE_CACHE[idx]


@torch.no_grad()
def _bnb_rowwise_roundtrip_deq(mat_2d: torch.Tensor, code: torch.Tensor, out_dtype: torch.dtype):
    """Quantize and immediately dequantize a 2D matrix row-by-row using bnb dynamic map."""
    assert mat_2d.dim() == 2
    rows = mat_2d.size(0)
    x = mat_2d.contiguous()
    out_rows = []
    for r in range(rows):
        q, (absmax, _) = bnb.functional.quantize(x[r].float(), code=code)
        q_u8 = q.to(torch.uint8)
        deq = bnb.functional.dequantize_no_absmax(q_u8, code=code).to(torch.float32)
        deq = deq * absmax.to(torch.float32)
        out_rows.append(deq)
    return torch.stack(out_rows, dim=0).to(out_dtype)


@torch.no_grad()
def _bnb_colwise_roundtrip_deq(mat_2d: torch.Tensor, code: torch.Tensor, out_dtype: torch.dtype):
    """Quantize and immediately dequantize a 2D matrix column-by-column using bnb dynamic map."""
    assert mat_2d.dim() == 2
    cols = mat_2d.size(1)
    x = mat_2d.contiguous()
    out_cols = []
    for c in range(cols):
        q, (absmax, _) = bnb.functional.quantize(x[:, c].float(), code=code)
        q_u8 = q.to(torch.uint8)
        deq = bnb.functional.dequantize_no_absmax(q_u8, code=code).to(torch.float32)
        deq = deq * absmax.to(torch.float32)
        out_cols.append(deq)
    return torch.stack(out_cols, dim=1).to(out_dtype)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SUBSTRINGS = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
}


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def _is_target_module(module_name: str) -> bool:
    return any(tag in module_name for tag in TARGET_SUBSTRINGS)


def break_even_rank(m: int, n: int) -> int:
    return max(1, (m * n - 1) // (m + n))


def right_solve_Rt(G: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve(R, G.transpose(0, 1)).transpose(0, 1)


def _safe_int(v: float) -> int:
    return int(math.floor(v + 1e-9))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModulePruneState:
    key: str
    layer_idx: int
    module_name: str
    shape: Tuple[int, int]
    sigma: torch.Tensor
    grad_sigma: torch.Tensor
    params: int
    cost: int
    cap_params: int
    max_svs: int
    consumed: int = 0
    dense_consumed: float = 0.0
    version: int = 0
    r_star: int = 0
    current_actual_rank: int = 0
    is_currently_lowrank: bool = False
    keep_mask: Optional[List[bool]] = None
    permutation: Optional[List[int]] = None
    mask_rank_only: bool = False
    total_sigma_sq: float = 0.0
    truncated_sigma_sq: float = 0.0
    beta_mixing: Optional[float] = None
    delta_order: str = "first_order"

    def module_error_ratio(self) -> float:
        if self.total_sigma_sq <= 0:
            return 0.0
        ratio = self.truncated_sigma_sq / self.total_sigma_sq
        ratio = max(0.0, min(1.0, float(ratio)))
        return math.sqrt(ratio)

    def dense_per_sv(self) -> float:
        return float(self.cost)

    def effective_dense_delta_if_drop(self) -> int:
        m, n = self.shape
        r_now = self.current_rank()
        if self.is_currently_lowrank:
            return self.cost
        else:
            if r_now > self.r_star:
                return 0
            elif r_now == self.r_star:
                return (m * n) - self.r_star * (m + n)
            else:
                return self.cost

    def remaining_svs(self) -> int:
        return max(0, self.max_svs - self.consumed)

    def current_rank(self) -> int:
        base_rank = self.sigma.numel()
        return max(0, base_rank - self.consumed)

    def tail_index(self) -> int:
        idx = self.sigma.numel() - 1 - self.consumed
        return idx

    def has_candidate(self) -> bool:
        return self.remaining_svs() > 0 and self.tail_index() >= 0

    def peek_delta(self) -> Optional[float]:
        if not self.has_candidate():
            return None
        idx = self.tail_index()
        sigma = self.sigma[idx].item()
        grad_sigma = self.grad_sigma[idx].item()

        if self.beta_mixing is None:
            if self.delta_order == "second_order":
                return -sigma * grad_sigma + 0.5 * (sigma ** 2) * (grad_sigma ** 2)
            else:
                return -sigma * grad_sigma

        beta = float(self.beta_mixing)
        if grad_sigma == 0.0 and sigma == 0.0:
            return 0.0

        eps = 1e-12
        sig_abs = max(abs(sigma), eps)
        grad_abs = max(abs(grad_sigma), eps)
        sign = 1.0 if grad_sigma >= 0.0 else -1.0
        mixed = (sig_abs ** beta) * (grad_abs ** (1.0 - beta))
        return -sign * mixed

    def peek_grad_sigma(self) -> Optional[float]:
        if not self.has_candidate():
            return None
        idx = self.tail_index()
        return self.grad_sigma[idx].item()

    def accept(self):
        if not self.has_candidate():
            raise RuntimeError("No candidate to accept")
        idx = self.tail_index()

        logical_idx = idx
        if self.permutation is not None and 0 <= idx < len(self.permutation):
            logical_idx = self.permutation[idx]

        if 0 <= idx < self.sigma.numel():
            sigma_val = self.sigma[idx].item()
            self.truncated_sigma_sq += float(sigma_val * sigma_val)

        if self.keep_mask is not None and 0 <= logical_idx < len(self.keep_mask):
            self.keep_mask[logical_idx] = False

        self.consumed += 1
        self.dense_consumed += self.dense_per_sv()
        self.version += 1

    def accept_at(self, idx: int):
        """Accept (drop) a specific SV by index, not just the tail."""
        if self.consumed >= self.max_svs:
            raise RuntimeError("No candidate to accept (cap reached)")
        if self.keep_mask is None:
            raise RuntimeError("keep_mask is None")
        if idx < 0 or idx >= self.sigma.numel():
            raise RuntimeError(f"Bad idx={idx}")

        logical_idx = idx
        if self.permutation is not None and 0 <= idx < len(self.permutation):
            logical_idx = self.permutation[idx]

        # Check if already dropped
        if 0 <= logical_idx < len(self.keep_mask) and not self.keep_mask[logical_idx]:
            raise RuntimeError("SV already dropped")

        sigma_val = self.sigma[idx].item()
        self.truncated_sigma_sq += float(sigma_val * sigma_val)

        if 0 <= logical_idx < len(self.keep_mask):
            self.keep_mask[logical_idx] = False

        self.consumed += 1
        self.dense_consumed += self.dense_per_sv()
        self.version += 1

    def undo_last(self):
        if self.consumed == 0:
            return
        idx = self.tail_index() + 1

        logical_idx = idx
        if self.permutation is not None and 0 <= idx < len(self.permutation):
            logical_idx = self.permutation[idx]

        if 0 <= idx < self.sigma.numel():
            sigma_val = self.sigma[idx].item()
            self.truncated_sigma_sq = max(
                0.0,
                self.truncated_sigma_sq - float(sigma_val * sigma_val),
            )

        if self.keep_mask is not None and 0 <= logical_idx < len(self.keep_mask):
            self.keep_mask[logical_idx] = True

        self.consumed -= 1
        self.dense_consumed = max(0.0, self.dense_consumed - self.dense_per_sv())
        self.version += 1


@dataclass
class StageSelection:
    name: str
    delta: float
    cost: int
    dense_cost: int


# ---------------------------------------------------------------------------
# Priority queue helpers
# ---------------------------------------------------------------------------

class CandidateQueues:
    def __init__(self, use_min_heap: bool = False):
        self.pos = []
        self.neg = []
        self.counter = 0
        self.use_min_heap = use_min_heap

    def push(self, state: ModulePruneState, delta: float):
        priority = abs(delta) if self.use_min_heap else -abs(delta)
        entry = (priority, self.counter, state.key, delta, state.version)
        self.counter += 1
        if delta >= 0:
            heapq.heappush(self.pos, entry)
        else:
            heapq.heappush(self.neg, entry)

    def _pop_from(self, heap):
        while heap:
            priority, _, key, delta, version = heapq.heappop(heap)
            return (priority, key, delta, version)
        return None

    def _peek_from(self, heap):
        if heap:
            priority, _, key, delta, version = heap[0]
            return (priority, key, delta, version)
        return None

    def pop_any(self):
        if not self.pos and not self.neg:
            return None
        if not self.pos:
            return self._pop_from(self.neg)
        if not self.neg:
            return self._pop_from(self.pos)
        p = self._peek_from(self.pos)
        n = self._peek_from(self.neg)
        if abs(p[3]) >= abs(n[3]):
            return self._pop_from(self.pos)
        return self._pop_from(self.neg)

    def pop_prefer_then_any(self, prefer_positive: bool):
        first = self.pos if prefer_positive else self.neg
        other = self.neg if prefer_positive else self.pos
        item = self._pop_from(first)
        if item is not None:
            return item
        item = self._pop_from(other)
        if item is not None:
            return item
        return self.pop_any()

    def empty(self) -> bool:
        return not self.pos and not self.neg


# ---------------------------------------------------------------------------
# Module manipulation utilities
# ---------------------------------------------------------------------------

def get_module_from_key(model_name: str, model, key: str):
    layer_str, name = key.split(".", 1)
    layer_idx = int(layer_str.replace("layer", ""))
    layers = get_layers(model_name, model)
    layer = layers[layer_idx]

    subset = find_target_modules_in_layer(layer)
    if name not in subset:
        subset = find_layers(layer)
        if name not in subset:
            raise KeyError(f"Module {key} not found in model")

    return subset[name], layer_idx, name


def extract_dense_weight(module: nn.Module) -> torch.Tensor:
    if isinstance(module, LowRankLinear):
        with torch.no_grad():
            weight = module.u_proj.weight.data @ module.v_proj.weight.data
        return weight.clone()
    if isinstance(module, nn.Linear):
        return module.weight.data.clone()
    raise TypeError(f"Unsupported module type {type(module)} for extraction")


def get_module_current_rank(module: nn.Module) -> Optional[int]:
    """Return current rank if LowRankLinear, else None (indicating dense)"""
    if isinstance(module, LowRankLinear):
        return module.rank
    return None


def get_module_actual_shape(module: nn.Module) -> Tuple[int, int]:
    """Return the effective weight shape (m, n) regardless of module type"""
    if isinstance(module, LowRankLinear):
        return (module.out_features, module.in_features)
    elif isinstance(module, nn.Linear):
        return module.weight.shape
    raise TypeError(f"Unsupported module type {type(module)}")


def replace_module(model_name: str, model, key: str, new_module: nn.Module):
    layer_str, name = key.split(".", 1)
    layer_idx = int(layer_str.replace("layer", ""))
    layers = get_layers(model_name, model)
    layer = layers[layer_idx]
    parts = name.split(".")
    obj = layer
    for attr in parts[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, parts[-1], new_module)


# ---------------------------------------------------------------------------
# Factorization functions
# ---------------------------------------------------------------------------

def factorize_linear_from_lowrank(
    lr_module: LowRankLinear,
    L: torch.Tensor,
    target_rank: int,
    device,
    use_triangular_solve: bool = True,
    keep_mask: Optional[List[bool]] = None,
    mask_rank_only: bool = False,
    svd_info: Optional[Dict[str, torch.Tensor]] = None,
    remap_fake_bnb: bool = False,
    quantize_8_bit: bool = False,
) -> LowRankLinear:
    """
    Optimized path: when module is already LowRankLinear,
    work directly with U/V factors instead of reconstructing dense weight.
    """
    dtype = lr_module.u_proj.weight.dtype
    current_rank = lr_module.rank

    if target_rank >= current_rank:
        return lr_module

    L = L.to(device, dtype=torch.float32)

    if svd_info is not None and "S" in svd_info:
        S = svd_info["S"].to(device, dtype=torch.float32)
        U = svd_info["U"].to(device, dtype=torch.float32)
        Vh = svd_info["Vh"].to(device, dtype=torch.float32)
    else:
        U_current = lr_module.u_proj.weight.data.to(device, dtype=torch.float32)
        V_current = lr_module.v_proj.weight.data.to(device, dtype=torch.float32)
        V_whitened = V_current @ L
        A = U_current @ V_whitened
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    if keep_mask is not None:
        max_len = min(len(keep_mask), S.shape[0])
        if mask_rank_only:
            keep = sum(1 for i in range(max_len) if keep_mask[i])
            if keep <= 0:
                keep = 1
            S_sel = S[:keep]
            U_sel = U[:, :keep]
            V_sel = Vh[:keep, :]
        else:
            keep_indices = [i for i in range(max_len) if keep_mask[i]]
            if not keep_indices:
                keep_indices = [0]
            idx_tensor = torch.tensor(keep_indices, device=S.device)
            S_sel = S[idx_tensor]
            U_sel = U[:, idx_tensor]
            V_sel = Vh[idx_tensor, :]
            keep = len(keep_indices)
    else:
        keep = max(1, min(target_rank, S.shape[0]))
        S_sel = S[:keep]
        U_sel = U[:, :keep]
        V_sel = Vh[:keep, :]

    eps = 1e-12

    if remap_fake_bnb:
        code = _get_bnb_code(L.device)
        if code is None:
            raise RuntimeError("remap_fake_bnb requested but bitsandbytes code is unavailable")

        m = int(lr_module.out_features)
        n = int(lr_module.in_features)
        k = int(S_sel.numel())
        min_dim = min(m, n)

        # U Sigma and V transpose in the SVD space of W @ L
        U_sigma = U_sel * S_sel.unsqueeze(0)      # [m, k]
        Vt = V_sel.clone()                        # [k, n]

        # Build Vtilde so that W ≈ U_sigma @ Vtilde in the unwhitened space
        if use_triangular_solve:
            Vtilde = torch.linalg.solve(L.transpose(0, 1), Vt.transpose(0, 1)).transpose(0, 1)  # [k, n]
        else:
            try:
                L_inv = torch.linalg.inv(L)
            except RuntimeError:
                L = L + 1e-6 * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
                L_inv = torch.linalg.inv(L)
            Vtilde = Vt @ L_inv  # [k, n]

        V = Vtilde.transpose(0, 1).contiguous()  # [n, k]

        if quantize_8_bit:
            # Uniform fake-quant on both factors. Used when the model is also
            # going to be stored as real int8 via --quantize_8_bit, so the
            # rowwise-all storage layout is what should be simulated here.
            U_sigma[:, :] = _bnb_colwise_roundtrip_deq(
                U_sigma, code=code, out_dtype=torch.float16
            ).to(torch.float32)
            Vt[:, :] = _bnb_rowwise_roundtrip_deq(
                Vt, code=code, out_dtype=torch.float16
            ).to(torch.float32)
        else:
            # Asymmetric fake-quant for vanilla --remap: smaller factor fully
            # quantized; bigger factor's tail beyond min(m,n) kept in fp to
            # mirror the storage layout where the bigger side only quantizes
            # its first min(m,n) rows/columns.
            if m >= n:
                # Vt is the smaller factor: full rowwise quant
                Vt[:, :] = _bnb_rowwise_roundtrip_deq(
                    Vt, code=code, out_dtype=torch.float16
                ).to(torch.float32)
                # U_sigma is the bigger factor: colwise quant, restore rows [min_dim:m] to fp
                U_sigma_orig = U_sigma.clone()
                U_sigma[:, :] = _bnb_colwise_roundtrip_deq(
                    U_sigma, code=code, out_dtype=torch.float16
                ).to(torch.float32)
                U_sigma[min_dim:, :] = U_sigma_orig[min_dim:, :]
                del U_sigma_orig
            else:
                # U_sigma is the smaller factor: full colwise quant
                U_sigma[:, :] = _bnb_colwise_roundtrip_deq(
                    U_sigma, code=code, out_dtype=torch.float16
                ).to(torch.float32)
                # Vt is the bigger factor: rowwise quant, restore columns [min_dim:n] to fp
                Vt_orig = Vt.clone()
                Vt[:, :] = _bnb_rowwise_roundtrip_deq(
                    Vt, code=code, out_dtype=torch.float16
                ).to(torch.float32)
                Vt[:, min_dim:] = Vt_orig[:, min_dim:]
                del Vt_orig

        # rebuild Vtilde after Vt changed
        if use_triangular_solve:
            Vtilde = torch.linalg.solve(L.transpose(0, 1), Vt.transpose(0, 1)).transpose(0, 1)
        else:
            Vtilde = Vt @ L_inv
        V = Vtilde.transpose(0, 1).contiguous()

        # Convert back to stored split factors
        sqrtS = torch.sqrt(torch.clamp(S_sel, min=eps))  # [k]
        inv_sqrtS = 1.0 / sqrtS

        U_trunc = U_sigma * inv_sqrtS.unsqueeze(0)       # [m, k]
        Vh = sqrtS.unsqueeze(1) * Vtilde                 # [k, n]

        u_factor = U_trunc.to(dtype=dtype).cpu()
        v_factor = Vh.to(dtype=dtype).cpu()

    else:
        S_trunc = torch.diag(torch.sqrt(S_sel))
        U_trunc = U_sel
        V_trunc = V_sel

        if use_triangular_solve:
            Vh_raw = S_trunc @ V_trunc
            Vh = torch.linalg.solve(L.transpose(0, 1), Vh_raw.transpose(0, 1)).transpose(0, 1)
        else:
            try:
                L_inv = torch.linalg.inv(L)
            except RuntimeError:
                L = L + 1e-6 * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
                L_inv = torch.linalg.inv(L)
            Vh = (S_trunc @ (V_trunc @ L_inv))

        u_factor = (U_trunc @ S_trunc).to(dtype=dtype).cpu()
        v_factor = Vh.to(dtype=dtype).cpu()

    bias_flag = lr_module.u_proj.bias is not None
    bias_value = lr_module.u_proj.bias.data.clone() if bias_flag else None

    new_lr = LowRankLinear(
        lr_module.in_features,
        lr_module.out_features,
        keep,
        bias=bias_flag
    ).to(dtype=dtype)
    new_lr.u_proj.weight.data.copy_(u_factor)
    new_lr.v_proj.weight.data.copy_(v_factor)
    if bias_flag and bias_value is not None:
        new_lr.u_proj.bias.data.copy_(bias_value)

    return new_lr


def factorize_linear(
    module: nn.Module,
    L: torch.Tensor,
    target_rank: int,
    device,
    use_triangular_solve: bool = True,
    keep_mask: Optional[List[bool]] = None,
    svd_info: Optional[Dict[str, torch.Tensor]] = None,
    mask_rank_only: bool = False,
    remap_fake_bnb: bool = False,
    quantize_8_bit: bool = False,
) -> LowRankLinear:
    """
    Factorize a linear module into low-rank form using whitened SVD.
    """
    if isinstance(module, LowRankLinear):
        return factorize_linear_from_lowrank(
            module, L, target_rank, device,
            use_triangular_solve=use_triangular_solve,
            keep_mask=keep_mask,
            mask_rank_only=mask_rank_only,
            svd_info=svd_info,
            remap_fake_bnb=remap_fake_bnb,
            quantize_8_bit=quantize_8_bit,
        )

    dense_weight = extract_dense_weight(module)
    dtype = dense_weight.dtype
    W = dense_weight.to(device, dtype=torch.float32)
    L = L.to(device, dtype=torch.float32)
    L_inv = None
    if not use_triangular_solve:
        try:
            L_inv = torch.linalg.inv(L)
        except RuntimeError:
            L = L + 1e-6 * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
            L_inv = torch.linalg.inv(L)

    if svd_info is not None and "S" in svd_info:
        S = svd_info["S"].to(device, dtype=torch.float32)
        U = svd_info["U"].to(device, dtype=torch.float32)
        Vh = svd_info["Vh"].to(device, dtype=torch.float32)
    else:
        W_scale = W @ L
        U, S, Vh = torch.linalg.svd(W_scale, full_matrices=False)

    if keep_mask is not None:
        max_len = min(len(keep_mask), S.shape[0])
        if mask_rank_only:
            keep = sum(1 for i in range(max_len) if keep_mask[i])
            if keep <= 0:
                keep = 1
            S_sel = S[:keep]
            U_sel = U[:, :keep]
            V_sel = Vh[:keep, :]
        else:
            keep_indices = [i for i in range(max_len) if keep_mask[i]]
            if not keep_indices:
                keep_indices = [0]
            idx_tensor = torch.tensor(keep_indices, device=S.device)
            S_sel = S[idx_tensor]
            U_sel = U[:, idx_tensor]
            V_sel = Vh[idx_tensor, :]
            keep = len(keep_indices)
    else:
        keep = max(1, min(target_rank, S.shape[0]))
        S_sel = S[:keep]
        U_sel = U[:, :keep]
        V_sel = Vh[:keep, :]

    eps = 1e-12

    if remap_fake_bnb:
        code = _get_bnb_code(W.device)
        if code is None:
            raise RuntimeError("remap_fake_bnb requested but bitsandbytes code is unavailable")

        m = int(W.shape[0])
        n = int(W.shape[1])
        k = int(S_sel.numel())
        min_dim = min(m, n)

        # U Sigma and V transpose in the SVD space of W @ L
        U_sigma = U_sel * S_sel.unsqueeze(0)      # [m, k]
        Vt = V_sel.clone()                        # [k, n]

        # Build Vtilde so that W ≈ U_sigma @ Vtilde in the unwhitened space
        if use_triangular_solve:
            Vtilde = torch.linalg.solve(L.transpose(0, 1), Vt.transpose(0, 1)).transpose(0, 1)  # [k, n]
        else:
            Vtilde = Vt @ L_inv  # [k, n]

        V = Vtilde.transpose(0, 1).contiguous()  # [n, k]

        if quantize_8_bit:
            # Uniform fake-quant on both factors. Used when the model is also
            # going to be stored as real int8 via --quantize_8_bit, so the
            # rowwise-all storage layout is what should be simulated here.
            U_sigma[:, :] = _bnb_colwise_roundtrip_deq(
                U_sigma, code=code, out_dtype=torch.float16
            ).to(torch.float32)
            Vt[:, :] = _bnb_rowwise_roundtrip_deq(
                Vt, code=code, out_dtype=torch.float16
            ).to(torch.float32)
        else:
            # Asymmetric fake-quant for vanilla --remap: smaller factor fully
            # quantized; bigger factor's tail beyond min(m,n) kept in fp to
            # mirror the storage layout where the bigger side only quantizes
            # its first min(m,n) rows/columns.
            if m >= n:
                # Vt is the smaller factor: full rowwise quant
                Vt[:, :] = _bnb_rowwise_roundtrip_deq(
                    Vt, code=code, out_dtype=torch.float16
                ).to(torch.float32)
                # U_sigma is the bigger factor: colwise quant, restore rows [min_dim:m] to fp
                U_sigma_orig = U_sigma.clone()
                U_sigma[:, :] = _bnb_colwise_roundtrip_deq(
                    U_sigma, code=code, out_dtype=torch.float16
                ).to(torch.float32)
                U_sigma[min_dim:, :] = U_sigma_orig[min_dim:, :]
                del U_sigma_orig
            else:
                # U_sigma is the smaller factor: full colwise quant
                U_sigma[:, :] = _bnb_colwise_roundtrip_deq(
                    U_sigma, code=code, out_dtype=torch.float16
                ).to(torch.float32)
                # Vt is the bigger factor: rowwise quant, restore columns [min_dim:n] to fp
                Vt_orig = Vt.clone()
                Vt[:, :] = _bnb_rowwise_roundtrip_deq(
                    Vt, code=code, out_dtype=torch.float16
                ).to(torch.float32)
                Vt[:, min_dim:] = Vt_orig[:, min_dim:]
                del Vt_orig

        # rebuild Vtilde after Vt changed
        if use_triangular_solve:
            Vtilde = torch.linalg.solve(L.transpose(0, 1), Vt.transpose(0, 1)).transpose(0, 1)
        else:
            Vtilde = Vt @ L_inv
        V = Vtilde.transpose(0, 1).contiguous()

        # Convert back to stored split factors
        sqrtS = torch.sqrt(torch.clamp(S_sel, min=eps))  # [k]
        inv_sqrtS = 1.0 / sqrtS

        U_trunc = U_sigma * inv_sqrtS.unsqueeze(0)       # [m, k]
        Vh = sqrtS.unsqueeze(1) * Vtilde                 # [k, n]

        u_factor = U_trunc.to(dtype=dtype).cpu()
        v_factor = Vh.to(dtype=dtype).cpu()

    else:
        S_trunc = torch.diag(torch.sqrt(S_sel))
        U_trunc = U_sel
        V_trunc = V_sel
        if use_triangular_solve:
            Vh_raw = S_trunc @ V_trunc
            Vh = torch.linalg.solve(L.transpose(0, 1), Vh_raw.transpose(0, 1)).transpose(0, 1)
        else:
            Vh = (S_trunc @ (V_trunc @ L_inv))
        u_factor = (U_trunc @ S_trunc).to(dtype=dtype).cpu()
        v_factor = Vh.to(dtype=dtype).cpu()

    if isinstance(module, LowRankLinear):
        bias_flag = module.u_proj.bias is not None
        bias_value = module.u_proj.bias.data.clone() if bias_flag else None
    elif isinstance(module, nn.Linear):
        bias_flag = module.bias is not None
        bias_value = module.bias.data.clone() if bias_flag else None
    else:
        bias_flag = False
        bias_value = None

    low_rank = LowRankLinear(module.in_features, module.out_features, keep, bias=bias_flag).to(dtype=dtype)
    low_rank.u_proj.weight.data.copy_(u_factor)
    low_rank.v_proj.weight.data.copy_(v_factor)
    if bias_flag and bias_value is not None:
        low_rank.u_proj.bias.data.copy_(bias_value)
    return low_rank


# ---------------------------------------------------------------------------
# Parameter counting utilities
# ---------------------------------------------------------------------------

def total_params(states: Dict[str, ModulePruneState]) -> int:
    return sum(state.params for state in states.values())


def compute_dense_params(model_name: str, model) -> int:
    total = 0
    layers = get_layers(model_name, model)
    for i, layer in enumerate(layers):
        for name, module in find_target_modules_in_layer(layer).items():
            if isinstance(module, nn.Linear):
                m, n = module.weight.shape
                total += m * n
                if module.bias is not None:
                    total += module.bias.numel()
            elif isinstance(module, LowRankLinear):
                total += module.u_proj.weight.numel() + module.v_proj.weight.numel()
                if module.u_proj.bias is not None:
                    total += module.u_proj.bias.numel()
    return total


def build_priority_queues(states: Dict[str, ModulePruneState], use_min_heap: bool = False) -> CandidateQueues:
    queues = CandidateQueues(use_min_heap=use_min_heap)
    for state in states.values():
        delta = state.peek_delta()
        if delta is None:
            continue
        queues.push(state, delta)
    return queues


def aggregate_stage_counts(selected: List[StageSelection]) -> Counter:
    counts = Counter()
    for item in selected:
        counts[item.name] += 1
    return counts


# ---------------------------------------------------------------------------
# Densification utilities
# ---------------------------------------------------------------------------

def densify_target_modules(model_name: str, model, verbose: bool = False) -> Dict[str, nn.Module]:
    """
    Replace each target LowRankLinear module with a dense nn.Linear.
    Returns a dict mapping logical module key -> original LowRankLinear.
    """
    replaced: Dict[str, nn.Module] = {}
    layers = get_layers(model_name, model)

    for layer_idx, layer in enumerate(layers):
        for name, mod in layer.named_modules():
            if not isinstance(mod, LowRankLinear):
                continue
            if not _is_target_module(name):
                continue

            key = f"layer{layer_idx}.{name}"
            if key in replaced:
                continue

            W_dense = extract_dense_weight(mod)
            out_features, in_features = W_dense.shape

            ref_weight = mod.u_proj.weight
            ref_device = ref_weight.device
            ref_dtype = ref_weight.dtype

            bias_flag = mod.u_proj.bias is not None
            dense = nn.Linear(in_features, out_features, bias=bias_flag).to(
                device=ref_device, dtype=ref_dtype
            )
            dense.weight.data.copy_(W_dense.to(ref_device, ref_dtype))
            if bias_flag:
                dense.bias.data.copy_(mod.u_proj.bias.data.to(ref_device, ref_dtype))

            replace_module(model_name, model, key, dense)
            replaced[key] = mod

            if verbose:
                print(f"   Densified {key}: LowRankLinear(rank={mod.rank}) -> nn.Linear({out_features}, {in_features})")

    if not verbose:
        print(f"   Densified {len(replaced)} LowRankLinear modules for gradient computation")
    return replaced


def restore_target_modules(model_name: str, model, replaced: Dict[str, nn.Module]):
    """
    Undo densify_target_modules by putting the original LowRankLinear modules back.
    """
    for key, orig_module in replaced.items():
        replace_module(model_name, model, key, orig_module)


# ---------------------------------------------------------------------------
# SVDLLM stage (direct SVD truncation, no greedy planner)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_svdllm_stage(
    args,
    model,
    tokenizer,
    dev,
    stage_idx: int,
    total_dense_baseline: int,
    compute_profiling_matrices_fn,
    build_L_cache_fn,
    ppl_eval_fn=None,
):
    """
    SVDLLM style outer stage.

    For each target module with weight W in R^{m x n} we:
      * compute a keep ratio from global_prune_ratio
      * set r_target = (keep_ratio * m * n) / (m + n)
      * truncate once from full rank to r_target using SVD on W L

    There are no inner stages and no greedy stage plan.

    Args:
        args: Argument namespace with model, global_prune_ratio, verbose, etc.
        model: The model to compress
        tokenizer: Tokenizer for the model
        dev: Device to use
        stage_idx: Current stage index
        total_dense_baseline: Initial dense parameter count
        compute_profiling_matrices_fn: Function to compute profiling matrices
        build_L_cache_fn: Function to build L cache from profiling matrices
        ppl_eval_fn: Optional function for perplexity evaluation
    """
    from tqdm import tqdm

    # Make sure we start from dense modules for this stage
    densified_map = densify_target_modules(args.model, model, verbose=args.verbose)
    if densified_map and not args.verbose:
        print(f"   📦 Densified {len(densified_map)} modules for SVDLLM stage {stage_idx}")
    # Free old LowRankLinear weights
    for old_mod in densified_map.values():
        for p in old_mod.parameters():
            p.data = p.data.cpu()
    del densified_map
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    current_dense = compute_dense_params(args.model, model)
    print(f"\n{'=' * 70}")
    print(
        f"🚉 SVDLLM Stage {stage_idx}/{args.num_stages} | "
        f"current dense params {current_dense:,} | global_prune_ratio={args.global_prune_ratio:.3f}"
    )
    print(f"{'=' * 70}")

    # Parameters we keep per module as a fraction of dense params
    # global_prune_ratio is the fraction we remove, so keep_ratio is what remains
    keep_ratio = max(0.0, 1.0 - float(args.global_prune_ratio))

    # Compute whitening matrices for this dense model
    profiling_mat = compute_profiling_matrices_fn(args, model, tokenizer, dev)
    L_cache, _ = build_L_cache_fn(profiling_mat, dev, dtype=torch.float32)

    layers = get_layers(args.model, model)

    num_compressed = 0
    total_removed_local = 0

    for layer_idx, layer in enumerate(tqdm(layers, desc=f"SVDLLM truncation stage {stage_idx}")):
        subset = find_target_modules_in_layer(layer)
        for name, mod in subset.items():
            if not isinstance(mod, nn.Linear):
                # Should not happen after densify_target_modules,
                # but skip anything that is not a dense linear
                continue

            W = mod.weight
            m, n = W.shape
            full_rank = min(m, n)
            if full_rank <= 1:
                continue

            # Target rank from keep ratio:
            # we want low rank params r_target * (m + n) to be keep_ratio * (m * n)
            r_target_float = keep_ratio * (m * n) / float(m + n)
            r_target = int(max(1, min(full_rank, round(r_target_float))))

            # No compression benefit, skip
            if r_target >= full_rank:
                continue

            # Whitening matrix for this module
            L = L_cache.get(layer_idx, {}).get(name)
            if L is None:
                L = torch.eye(n, device=dev, dtype=torch.float32)
            else:
                L = L.to(dev, dtype=torch.float32)

            key = f"layer{layer_idx}.{name}"

            low_rank = factorize_linear(
                mod,
                L,
                r_target,
                dev,
                use_triangular_solve=args.use_triangular_solve,
            )
            replace_module(args.model, model, key, low_rank.to(dev))

            num_compressed += 1
            dense_before = m * n
            dense_after = r_target * (m + n)
            total_removed_local += max(0, dense_before - dense_after)

    current_dense_after = compute_dense_params(args.model, model)
    print(
        f"   → SVDLLM stage {stage_idx}: compressed {num_compressed} modules, "
        f"module wise removed about {total_removed_local:,} dense params"
    )
    print(
        f"   → Dense params after SVDLLM stage {stage_idx}: "
        f"{current_dense_after:,} "
        f"(removed {total_dense_baseline - current_dense_after:,} from baseline)"
    )

    # Optional evaluation after this stage, before any densification
    if getattr(args, 'eval_ppl_per_inner_stage', False) and ppl_eval_fn is not None:
        print(f"\n   📊 Evaluating perplexity after SVDLLM stage {stage_idx} (low rank)...")
        stage_ppl = ppl_eval_fn(
            model,
            tokenizer,
            datasets=['wikitext2'],
            model_seq_len=args.model_seq_len,
            batch_size=4,
            device=args.DEV,
        )
        print(f"   ✓ Perplexity (after SVDLLM stage {stage_idx}, low rank): {stage_ppl}")

    return model


# ---------------------------------------------------------------------------
# Between-stage correction functions
# ---------------------------------------------------------------------------


def run_gd_correction_after_truncation(
    args,
    model,
    tokenizer,
    dev,
    stage_idx: int,
    inner_idx: int,
):
    """
    One step gradient direction correction.

    Uses up to nsamples_gd_correction samples from args.base_calib_data.
    For kd, aligns cached soft labels using the same global sample pointer
    convention as compute_average_gradients.
    """
    if not getattr(args, "gd_correction_mode", False):
        return

    base_list = getattr(args, "base_calib_data", None)
    if base_list is None:
        print("   GD correction requested but base calibration data is missing, skipping")
        return

    print(f"\n🧭 GD correction after truncation, stage {stage_idx}, inner {inner_idx}, using CE (cross entropy)")

    # Determine how many samples to use
    max_gd_samples = getattr(args, "nsamples_gd_correction", None)
    if max_gd_samples is not None:
        try:
            max_gd_samples = int(max_gd_samples)
        except Exception:
            max_gd_samples = None
        if max_gd_samples is not None and max_gd_samples <= 0:
            max_gd_samples = None

    # Collect parameters of target modules only
    layers = get_layers(args.model, model)
    target_params: List[torch.nn.Parameter] = []
    seen = set()
    for layer in layers:
        subset = find_target_modules_in_layer(layer)
        for _name, mod in subset.items():
            for p in mod.parameters():
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                target_params.append(p)

    if not target_params:
        print("   No target parameters found for GD correction, skipping")
        return

    model = model.to(dev)
    model.train()

    # Disable cache during this gradient pass
    prev_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        prev_use_cache = model.config.use_cache
        model.config.use_cache = False

    # Autocast dtype based on model parameters
    autocast_dtype = (
        torch.bfloat16
        if any(p.dtype == torch.bfloat16 for p in model.parameters())
        else torch.float16
    )

    # Zero grads
    for p in target_params:
        p.grad = None

    global_sample_ptr = 0
    batch_count = 0

    # Set progress bar total to actual samples we will process
    total_samples = len(base_list)
    if max_gd_samples is not None:
        total_samples = min(len(base_list), max_gd_samples)

    for batch in tqdm(base_list, desc="GD correction gradients", total=total_samples):
        if max_gd_samples is not None and global_sample_ptr >= max_gd_samples:
            break

        raw_ids = batch["input_ids"].to(dev)
        raw_am = batch.get("attention_mask", None)
        if raw_am is not None:
            raw_am = raw_am.to(dev)
        else:
            raw_am = torch.ones_like(raw_ids)

        raw_pi = batch.get("position_ids", None)
        if raw_pi is not None:
            raw_pi = raw_pi.to(dev)

        B_all = raw_ids.size(0)

        if max_gd_samples is not None:
            remaining = max_gd_samples - global_sample_ptr
            if remaining <= 0:
                break
            B_use = min(B_all, remaining)
        else:
            B_use = B_all

        if B_use <= 0:
            break

        ids = raw_ids[:B_use]
        am = raw_am[:B_use]
        position_ids = raw_pi[:B_use] if raw_pi is not None else None
        B = ids.size(0)

        inputs = {
            "input_ids": ids,
            "attention_mask": am,
        }
        if position_ids is not None:
            inputs["position_ids"] = position_ids

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            out = model(**inputs)
            logits = out.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )

        loss.backward()

        global_sample_ptr += B
        batch_count += 1

        del out, logits, loss
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    n = max(batch_count, 1)

    # Average grads across batches, then apply one manual step
    with torch.no_grad():
        for p in target_params:
            if p.grad is None:
                continue
            p.grad.data.div_(n)
            p.data.add_(-float(getattr(args, "gd_correction_lr", 1e-4)) * p.grad.data)

    # Clear grads
    for p in target_params:
        p.grad = None

    model.eval()

    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache

    print(f"   ✅ GD correction applied using {batch_count} gradient batches\n")


def run_alpha_blend_between_stages(
    args,
    model,
    dev,
    teacher_weights: Dict[str, torch.Tensor],
    stage_idx: int,
    substituted_keys: Optional[set] = None,
):
    """
    After densification, blend each logical module weight toward teacher:
        W <- W + alpha * (W_T - W)
    This is equivalent to:
        W <- (1 - alpha) * W + alpha * W_T

    Args:
        args: Arguments namespace
        model: The model to modify
        dev: Device
        teacher_weights: Dict mapping module keys to teacher weight tensors
        stage_idx: Current stage index
        substituted_keys: Optional set of keys that have been substituted with teacher weights
                         (these will be skipped). If None, no keys are skipped.
    """
    if not getattr(args, "alpha_blend_correction", False):
        return

    alpha = float(getattr(args, "blend_factor", 0.0))
    if alpha <= 0.0:
        print(f"\n🧪 Stage {stage_idx}: alpha_blend_correction enabled but alpha <= 0, skipping")
        return

    print(f"\n🧪 Stage {stage_idx}: alpha_blend_correction with alpha={alpha}")

    layers = get_layers(args.model, model)

    num_updated = 0
    num_skipped = 0

    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            subset = find_target_modules_in_layer(layer)
            for name, mod in subset.items():
                if not isinstance(mod, nn.Linear):
                    continue

                key = f"layer{layer_idx}.{name}"

                # Respect teacher substitution tracking
                if substituted_keys is not None and key in substituted_keys:
                    num_skipped += 1
                    continue

                w_teacher = teacher_weights.get(key, None)
                if w_teacher is None:
                    num_skipped += 1
                    continue

                W_r = mod.weight.data
                if tuple(w_teacher.shape) != tuple(W_r.shape):
                    num_skipped += 1
                    continue

                # Move teacher to device and dtype for in place update
                wT = w_teacher.to(W_r.device, dtype=W_r.dtype)
                W_r.add_(alpha * (wT - W_r))
                num_updated += 1

    print(f"   ✅ alpha_blend_correction updated {num_updated} modules, skipped {num_skipped}")
