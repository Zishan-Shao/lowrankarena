#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DF-SVD reproduction (whitening + decay-aware rank allocation + feature-preserved weight update)
Target model: meta-llama/Llama-2-7b-hf (Transformers)

Key paper equations implemented:
- Whitening: S = chol(X X^T), decompose W S = U Σ V^T, so W = U Σ V^T S^{-1}
  and define Wu = U Σ, Wv = V^T S^{-1}.
- Decay-aware ranks: fit σ_i ≈ σ0 exp(-λ i), normalize λ, then
  r_trunc = r_old * (1 - λ_norm), r_update = rank_old * (1 - λ_norm),
  r_old = floor(m n r / (m + n)).
- Feature-preserved update: fix Wv and principal components of Wu, only update minor components.

Note: This file focuses on correctness + "can run". It is NOT optimized.
For a paper-aligned calibration flow (Alg.1), use `--whitening_cache_dir` to compute/store whitening factors
from the original (teacher) model once, then reuse them during compression (avoids distribution shift + re-collecting cov).

# smoke test
CUDA_VISIBLE_DEVICES=2 python df_svd.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --output_dir ./llama2_dfsvd_debug \
  --compression_ratio 0.4 \
  --base_update_rank 8 \
  --seq_len 256 \
  --calib_sequences 64 \
  --batch_size 1 \
  --layer_start 0 \
  --layer_end 2 \
  --do_train \
  --train_steps 50

# full run
CUDA_VISIBLE_DEVICES=2 python df_svd.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --output_dir ./llama2_dfsvd_r0.4 \
  --compression_ratio 0.4 \
  --base_update_rank 8 \
  --seq_len 2048 \
  --calib_sequences 256 \
  --cov_max_tokens_total 524288 \
  --batch_size 1 \
  --whitening_cache_dir ./cache/dfsvd_whitening_llama2_seq2048_n256_tok524288 \
  --whitening_cache_dtype float32 \
  --do_train \
  --train_steps 200 \
  --train_lr 5e-4

# example (randomized WS-SVD + cached whitening)
CUDA_VISIBLE_DEVICES=3 conda run -n flashsvd python -u robust/df_svd.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --output_dir ./robust/llama2_dfsvd_r0.4_full_fix_cacheS \
  --compression_ratio 0.4 --base_update_rank 8 \
  --seq_len 2048 --calib_sequences 256 --cov_max_tokens_total 524288 --batch_size 1 \
  --whitening_cache_dir ./cache/dfsvd_whitening_llama2_seq2048_n256_tok524288 \
  --whitening_cache_dtype float16 \
  --whitening_cache_overwrite \
  --ws_svd_method randomized --ws_svd_niter 2 --ws_svd_oversample 8 \
  --factor_dtype float32 --whitening_damping 1e-4 \
  --do_train --train_steps 200 --train_lr 1e-4 \
  --tqdm off


"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator


# LLaMA decoder layer targets (per paper: Q,K,V,O + gate,up,down)
TARGET_SUBMODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def resolve_tqdm_enabled(mode: str) -> bool:
    mode = str(mode).lower().strip()
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto: disable progress bars when stderr is not a TTY (prevents tqdm from spamming logs).
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _log(msg: str, *, tqdm_enabled: bool) -> None:
    if tqdm_enabled:
        try:
            tqdm.write(msg)
            return
        except Exception:
            pass
    print(msg, flush=True)


def _whitening_cache_entry_path(cache_dir: str, module_name: str) -> str:
    # module_name contains '.' which is safe for filenames; keep it for readability.
    return os.path.join(cache_dir, f"{module_name}.pt")


def _whitening_cache_meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "whitening_cache_meta.json")


def _write_whitening_cache_meta(cache_dir: str, meta: Dict[str, Any], *, tqdm_enabled: bool) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _whitening_cache_meta_path(cache_dir)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            # Best-effort mismatch warning (do not fail hard).
            mismatch_keys = []
            for k in ["model_id", "seq_len", "calib_sequences", "cov_max_tokens_total", "cov_sample_tokens_per_batch", "seed", "whitening_damping"]:
                if k in prev and k in meta and prev[k] != meta[k]:
                    mismatch_keys.append(k)
            if mismatch_keys:
                _log(
                    f"[Warn] Whitening cache meta mismatch on keys={mismatch_keys}. "
                    f"Cache may be inconsistent with current args: {path}",
                    tqdm_enabled=tqdm_enabled,
                )
        except Exception as e:
            _log(f"[Warn] Failed to read whitening cache meta: {e}", tqdm_enabled=tqdm_enabled)
        return

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception as e:
        _log(f"[Warn] Failed to write whitening cache meta: {e}", tqdm_enabled=tqdm_enabled)


def load_whitening_S_from_cache(
    cache_dir: str,
    module_name: str,
    *,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    path = _whitening_cache_entry_path(cache_dir, module_name)
    if not os.path.isfile(path):
        return None, None
    entry = torch.load(path, map_location="cpu")
    if torch.is_tensor(entry):
        S_cpu = entry
        n_tokens = 0
    elif isinstance(entry, dict) and "S" in entry:
        S_cpu = entry["S"]
        n_tokens = int(entry.get("n_tokens", 0))
    else:
        raise TypeError(f"Unexpected whitening cache entry type for {path}: {type(entry)}")
    if not torch.is_tensor(S_cpu):
        raise TypeError(f"Whitening cache entry 'S' is not a tensor: {path}")
    S = S_cpu.to(device=device, dtype=torch.float32)
    return S, n_tokens


def save_whitening_S_to_cache(
    cache_dir: str,
    module_name: str,
    *,
    S: torch.Tensor,
    n_tokens: int,
    cache_dtype: torch.dtype,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = _whitening_cache_entry_path(cache_dir, module_name)
    payload = {"S": S.detach().to("cpu", dtype=cache_dtype), "n_tokens": int(n_tokens)}
    torch.save(payload, path)
    return path


def get_or_compute_whitening_S(
    model: nn.Module,
    module_name: str,
    dataloader: DataLoader,
    device: torch.device,
    *,
    cov_max_batches: Optional[int],
    cov_max_tokens_total: Optional[int],
    cov_sample_tokens_per_batch: Optional[int],
    whitening_damping: float,
    whitening_cache_dir: Optional[str],
    whitening_cache_dtype: torch.dtype,
    whitening_cache_overwrite: bool,
) -> Tuple[torch.Tensor, int, float, float, bool]:
    """
    Returns:
      S (on device, float32), cov_tokens, cov_sec, chol_sec, cache_hit
    """
    if whitening_cache_dir is not None and not whitening_cache_overwrite:
        S_cached, n_tokens = load_whitening_S_from_cache(whitening_cache_dir, module_name, device=device)
        if S_cached is not None:
            return S_cached, int(n_tokens or 0), 0.0, 0.0, True

    mod = get_module_by_name(model, module_name)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{module_name} is not nn.Linear")

    t0 = time.perf_counter()
    cov_stats = collect_input_covariance(
        model,
        mod,
        dataloader,
        device,
        max_batches=cov_max_batches,
        max_tokens_total=cov_max_tokens_total,
        sample_tokens_per_batch=cov_sample_tokens_per_batch,
    )
    cov_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    S, _ = cholesky_whitening_from_cov(cov_stats.cov, damping=whitening_damping)
    chol_sec = time.perf_counter() - t0

    if whitening_cache_dir is not None:
        save_whitening_S_to_cache(
            whitening_cache_dir,
            module_name,
            S=S,
            n_tokens=cov_stats.n_tokens,
            cache_dtype=whitening_cache_dtype,
        )

    return S, int(cov_stats.n_tokens), float(cov_sec), float(chol_sec), False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    cur: nn.Module = root
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def set_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _safe_int(x: float, min_v: int = 1, max_v: Optional[int] = None) -> int:
    v = int(math.floor(x))
    v = max(min_v, v)
    if max_v is not None:
        v = min(max_v, v)
    return v


def svd_topk(
    mat: torch.Tensor,
    k: int,
    *,
    method: str = "randomized",
    niter: int = 2,
    oversample: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute top-k SVD of `mat` (m,n) -> (U_k, s_k, Vh_k).

    - method="full": uses torch.linalg.svd (exact, expensive).
    - method="randomized": uses torch.svd_lowrank to approximate top-k.

    Falls back to full SVD if low-rank SVD fails or if k >= min(m,n).
    """
    if mat.ndim != 2:
        raise ValueError(f"svd_topk expects a 2D matrix, got {tuple(mat.shape)}")
    m, n = int(mat.shape[0]), int(mat.shape[1])
    max_rank = min(m, n)
    k = int(max(1, min(int(k), max_rank)))
    method = str(method or "full").strip().lower()

    # Exact path or trivial "top-k == full"
    if method == "full" or k >= max_rank:
        try:
            U, s, Vh = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            U, s, Vh = torch.linalg.svd(mat.cpu(), full_matrices=False)
            U, s, Vh = U.to(mat.device), s.to(mat.device), Vh.to(mat.device)
        return U[:, :k], s[:k], Vh[:k, :]

    # Randomized / low-rank path
    q = min(k + max(0, int(oversample)), max_rank - 1)
    q = max(1, q)
    try:
        U, s, V = torch.svd_lowrank(mat, q=q, niter=max(0, int(niter)))
        # Sort descending by singular value (torch.svd_lowrank is not guaranteed sorted across versions)
        idx = torch.argsort(s, descending=True)
        s = s[idx]
        U = U[:, idx]
        V = V[:, idx]
        return U[:, :k], s[:k], V[:, :k].t()
    except Exception:
        # Fallback to exact SVD (and truncate)
        try:
            U, s, Vh = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            U, s, Vh = torch.linalg.svd(mat.cpu(), full_matrices=False)
            U, s, Vh = U.to(mat.device), s.to(mat.device), Vh.to(mat.device)
        return U[:, :k], s[:k], Vh[:k, :]


# -------------------------
# Huber regression in log-space for exponential spectral decay fitting
# -------------------------
def huber_linefit(x: torch.Tensor, y: torch.Tensor, delta: float = 1.0, iters: int = 20) -> Tuple[float, float]:
    """
    Robust linear regression y ~= a + b x using Huber loss (IRLS).
    Returns (a, b).
    """
    assert x.ndim == 1 and y.ndim == 1 and x.numel() == y.numel()
    x = x.float()
    y = y.float()
    X = torch.stack([torch.ones_like(x), x], dim=1)  # (n,2)

    # OLS init
    theta = torch.linalg.lstsq(X, y).solution  # (2,)
    for _ in range(iters):
        r = y - X @ theta
        w = torch.where(r.abs() <= delta, torch.ones_like(r), delta / (r.abs() + 1e-8))
        Xw = X * w.unsqueeze(1)
        yw = y * w
        theta = torch.linalg.lstsq(Xw, yw).solution
    a, b = theta[0].item(), theta[1].item()
    return a, b


def estimate_lambda_from_sigma(
    sigma: torch.Tensor,
    fit_max_rank: Optional[int] = 256,
    huber_delta: float = 1.0,
    huber_iters: int = 20,
) -> float:
    """
    Fit normalized singular values: sigma_i/sigma_0 ≈ exp(-lambda * i)
    In log-space: log(sigma_i/sigma_0) ≈ -lambda*i
    """
    sigma = sigma.detach().float().clamp_min(1e-12)
    sigma = sigma / sigma[0]
    n = sigma.numel()
    if fit_max_rank is not None:
        n = min(n, fit_max_rank)
        sigma = sigma[:n]
    x = torch.arange(n, device=sigma.device, dtype=torch.float32)
    y = torch.log(sigma)
    _, b = huber_linefit(x, y, delta=huber_delta, iters=huber_iters)
    lam = -b
    return float(lam)


# -------------------------
# Whitening: S = chol(X^T X), and S_inv = S^{-1}
# -------------------------
@dataclass
class CovStats:
    cov: torch.Tensor
    n_tokens: int


def collect_input_covariance(
    model: nn.Module,
    module: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
    max_tokens_total: Optional[int] = 131072,
    sample_tokens_per_batch: Optional[int] = 2048,
    dtype_acc: torch.dtype = torch.float32,
) -> CovStats:
    """
    Collect cov = X^T X where X are the flattened inputs to `module` over calibration data.
    We do NOT normalize by token count (to match S = chol(XX^T) in the paper).
    """
    if not isinstance(module, nn.Linear):
        raise TypeError("collect_input_covariance expects nn.Linear")

    in_features = module.in_features
    cov = torch.zeros((in_features, in_features), device=device, dtype=dtype_acc)
    n_tokens = 0

    def hook(_mod: nn.Module, inp: Tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        nonlocal cov, n_tokens
        x = inp[0].detach()
        x2d = x.reshape(-1, x.shape[-1]).to(dtype_acc)
        if sample_tokens_per_batch is not None and x2d.shape[0] > sample_tokens_per_batch:
            idx = torch.randperm(x2d.shape[0], device=x2d.device)[:sample_tokens_per_batch]
            x2d = x2d[idx]
        # Guard against NaNs/Infs (can happen if previous layers were updated poorly).
        if not torch.isfinite(x2d).all():
            mask = torch.isfinite(x2d).all(dim=1)
            x2d = x2d[mask]
            if x2d.numel() == 0:
                return
        cov = cov + x2d.t() @ x2d
        n_tokens += x2d.shape[0]

    handle = module.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
            if max_tokens_total is not None and n_tokens >= max_tokens_total:
                break

    handle.remove()
    return CovStats(cov=cov, n_tokens=n_tokens)


def cholesky_whitening_from_cov(cov: torch.Tensor, damping: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given cov = X^T X, compute:
      S = chol(cov + eps I)
      S_inv = S^{-1} (lower-triangular inverse)
    """
    assert cov.ndim == 2 and cov.shape[0] == cov.shape[1]
    n = cov.shape[0]
    # Match SVDLLM-style robustness: symmetrize + scale-aware diagonal jitter.
    cov = (cov + cov.t()) * 0.5
    diag_mean = cov.diag().abs().mean()
    if not torch.isfinite(diag_mean):
        diag_mean = torch.tensor(1.0, device=cov.device, dtype=cov.dtype)
    diag_mean_v = float(diag_mean.item())
    if diag_mean_v <= 0:
        diag_mean_v = 1.0
    base_eps = float(max(float(damping), 1e-12)) * diag_mean_v

    eye = torch.eye(n, device=cov.device, dtype=cov.dtype)
    S = None
    # Try progressively larger jitters (this fixes near-PSD and small negative eigenvalues).
    for mul in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        try:
            S = torch.linalg.cholesky(cov + (base_eps * mul) * eye)
            break
        except RuntimeError:
            continue
    if S is None:
        # Last resort: project to SPD by eigenvalue clipping, then Cholesky.
        w, Q = torch.linalg.eigh(cov)
        w = torch.clamp(w, min=base_eps)
        cov_spd = (Q * w) @ Q.t()
        S = torch.linalg.cholesky(cov_spd)

    I = torch.eye(n, device=cov.device, dtype=cov.dtype)
    S_inv = torch.linalg.solve_triangular(S, I, upper=False)
    return S, S_inv


# -------------------------
# DF-SVD factorized linear module:
#   y = (Wp + Bm@Am)^T * (Wv x)
# with Wv, Wp frozen; train Bm,Am (minor components)
# -------------------------
class DFSVDFactorizedLinear(nn.Module):
    """
    Row-major implementation:
      x: (..., in)
      xk = x @ Wv^T           -> (..., k)
      Wu = Wp + Bm@Am         -> (out, k)
      y  = xk @ Wu^T          -> (..., out)
    """

    def __init__(self, in_features: int, out_features: int, rank_k: int, update_rank: int, dtype: torch.dtype):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank_k = int(rank_k)
        self.update_rank = int(update_rank)

        # Store factors in a numerically safe dtype (often float32/bfloat16), and cast outputs
        # back to the input dtype in forward(). This avoids float16 overflow when S^{-1} is ill-conditioned.
        self.register_buffer("Wv", torch.empty((rank_k, in_features), dtype=dtype))
        self.register_buffer("Wp", torch.empty((out_features, rank_k), dtype=dtype))

        if update_rank > 0:
            self.Bm = nn.Parameter(torch.empty((out_features, update_rank), dtype=dtype))
            self.Am = nn.Parameter(torch.empty((update_rank, rank_k), dtype=dtype))
        else:
            self.Bm = None
            self.Am = None

    @torch.no_grad()
    def set_factors(self, Wv: torch.Tensor, Wp: torch.Tensor, Bm: Optional[torch.Tensor], Am: Optional[torch.Tensor]) -> None:
        self.Wv.copy_(Wv)
        self.Wp.copy_(Wp)
        if self.update_rank > 0 and Bm is not None and Am is not None:
            self.Bm.copy_(Bm)  # type: ignore[union-attr]
            self.Am.copy_(Am)  # type: ignore[union-attr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        y2d = self.forward_flat(x2d)
        return y2d.reshape(*orig_shape[:-1], self.out_features)

    def forward_flat(self, x2d: torch.Tensor) -> torch.Tensor:
        # x2d: (tokens, in)
        in_dtype = x2d.dtype
        xk = x2d.to(self.Wv.dtype) @ self.Wv.t()
        Wu = self.Wp
        if self.update_rank > 0 and self.Bm is not None and self.Am is not None:
            Wu = Wu + self.Bm @ self.Am
        y = xk @ Wu.t()
        if y.dtype != in_dtype:
            # Prevent float16 overflow from producing inf/NaN and poisoning later layers.
            try:
                if in_dtype.is_floating_point:
                    maxv = torch.finfo(in_dtype).max
                    y = torch.clamp(y, min=-maxv, max=maxv)
            except Exception:
                pass
            y = y.to(in_dtype)
        return y

    @staticmethod
    def build_from_ws_svd(
        W_dense: torch.Tensor,
        S: torch.Tensor,
        S_inv: torch.Tensor,
        trunc_rank: int,
        update_rank: int,
        *,
        dtype: torch.dtype,
    ) -> "DFSVDFactorizedLinear":
        """
        Whitening + SVD:
          WS = W @ S = U diag(s) V^T
          Wu = U[:, :k] diag(s[:k])         (out,k)
          Wv = V[:k]^T @ S_inv              (k,in)

        Then SVD(Wu) and split into principal/minor:
          Wu = Wp + Wm, freeze Wp, train Wm ≈ Bm@Am
        """
        device = W_dense.device

        W = W_dense.float()
        WS = W @ S  # (out,in)

        # SVD
        try:
            U, s, Vh = torch.linalg.svd(WS, full_matrices=False)
        except RuntimeError:
            # fallback to CPU SVD if GPU fails
            U, s, Vh = torch.linalg.svd(WS.cpu(), full_matrices=False)
            U, s, Vh = U.to(device), s.to(device), Vh.to(device)

        k = int(trunc_rank)
        U_k = U[:, :k]
        s_k = s[:k]
        Vh_k = Vh[:k, :]

        Wu_init = U_k * s_k  # (out,k) = U diag(s)
        Wv = Vh_k @ S_inv    # (k,in)  = V^T S^{-1}

        # split Wu_init into principal + minor via SVD(Wu_init)
        U2, s2, Vh2 = torch.linalg.svd(Wu_init, full_matrices=False)  # U2(out,k) s2(k) Vh2(k,k)

        r = min(int(update_rank), k)
        p = k - r

        if p > 0:
            Wp = (U2[:, :p] * s2[:p]) @ Vh2[:p, :]
        else:
            Wp = torch.zeros((Wu_init.shape[0], Wu_init.shape[1]), device=device, dtype=torch.float32)

        if r > 0:
            Um = U2[:, p:]
            sm = s2[p:].clamp_min(1e-12)
            Vhm = Vh2[p:, :]
            sqrt_sm = torch.sqrt(sm)
            Bm = Um * sqrt_sm
            Am = sqrt_sm.unsqueeze(1) * Vhm
        else:
            Bm = None
            Am = None

        mod = DFSVDFactorizedLinear(
            in_features=W_dense.shape[1],
            out_features=W_dense.shape[0],
            rank_k=k,
            update_rank=r,
            dtype=dtype,
        ).to(device)

        mod.set_factors(
            Wv=Wv.to(dtype),
            Wp=Wp.to(dtype),
            Bm=None if Bm is None else Bm.to(dtype),
            Am=None if Am is None else Am.to(dtype),
        )

        # freeze buffers
        mod.Wv.requires_grad_(False)
        mod.Wp.requires_grad_(False)
        return mod


# -------------------------
# Data: calibration set (Wikitext-2)
# -------------------------
def build_calibration_dataloader(
    tokenizer: Any,
    *,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    num_sequences: int = 256,
    seq_len: int = 512,
    batch_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    raw = load_dataset(dataset_name, dataset_config, split=split)

    # drop empty lines
    raw = raw.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)
    raw = raw.shuffle(seed=seed)
    raw = raw.select(range(min(num_sequences * 2, len(raw))))  # take some extra before chunking

    def tok_fn(examples):
        return tokenizer(examples["text"])
    tok = raw.map(tok_fn, batched=True, remove_columns=raw.column_names)

    def group_texts(examples):
        concatenated: List[int] = []
        for ids in examples["input_ids"]:
            concatenated += ids
        total_len = (len(concatenated) // seq_len) * seq_len
        concatenated = concatenated[:total_len]
        blocks = [concatenated[i:i + seq_len] for i in range(0, total_len, seq_len)]
        return {"input_ids": blocks, "attention_mask": [[1] * seq_len for _ in range(len(blocks))]}

    tok2 = tok.map(group_texts, batched=True)
    tok2 = tok2.select(range(min(num_sequences, len(tok2))))

    return DataLoader(tok2, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)


# -------------------------
# Rank allocation (decay-aware)
# -------------------------
@dataclass
class RankAllocConfig:
    compression_ratio: float
    base_update_rank: int
    fit_max_rank: int = 256
    huber_delta: float = 1.0
    huber_iters: int = 20


def compute_base_rank(out_features: int, in_features: int, compression_ratio: float) -> int:
    # Eq(6): raold = floor(m*n*r/(m+n))
    m = out_features
    n = in_features
    k = math.floor((m * n * compression_ratio) / (m + n))
    return max(1, int(k))


def compute_ranks_for_module(
    out_features: int,
    in_features: int,
    *,
    compression_ratio: float,
    base_update_rank: int,
    lambda_norm: float,
) -> Tuple[int, int]:
    k_base = compute_base_rank(out_features, in_features, compression_ratio)
    k_trunc = _safe_int(k_base * (1.0 - lambda_norm), min_v=1, max_v=min(out_features, in_features))
    r_up = _safe_int(base_update_rank * (1.0 - lambda_norm), min_v=1, max_v=k_trunc)
    return k_trunc, r_up


def collect_all_target_module_names(model: nn.Module, layer_start: int = 0, layer_end: Optional[int] = None) -> List[str]:
    layers = model.model.layers
    if layer_end is None:
        layer_end = len(layers)
    names: List[str] = []
    for i in range(layer_start, layer_end):
        for sub in TARGET_SUBMODULES:
            names.append(f"model.layers.{i}.{sub}")
    return names


def compute_lambdas_two_pass(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    fit_max_rank: int = 256,
    huber_delta: float = 1.0,
    huber_iters: int = 20,
    cov_max_batches: Optional[int] = None,
    cov_max_tokens_total: Optional[int] = 131072,
    cov_sample_tokens_per_batch: Optional[int] = 2048,
    whitening_damping: float = 1e-5,
    ws_svd_method: str = "randomized",
    ws_svd_niter: int = 2,
    ws_svd_oversample: int = 8,
    tqdm_enabled: bool = True,
    log_every: int = 20,
    whitening_cache_dir: Optional[str] = None,
    whitening_cache_dtype: torch.dtype = torch.bfloat16,
    whitening_cache_overwrite: bool = False,
) -> Dict[str, float]:
    """
    Faithful but slow: two-pass normalization for lambda.
    Pass-1: compute lambda for every target matrix (needs whitening + svdvals).
    """
    lambdas: Dict[str, float] = {}
    module_names = collect_all_target_module_names(model, layer_start, layer_end)

    n_total = len(module_names)
    it = (
        tqdm(enumerate(module_names, 1), total=n_total, desc="Lambda global prepass", leave=False)
        if tqdm_enabled
        else enumerate(module_names, 1)
    )
    for idx, name in it:
        if not tqdm_enabled and log_every:
            if idx == 1 or idx % int(log_every) == 0 or idx == n_total:
                _log(f"[Lambda] {idx}/{n_total}: {name}", tqdm_enabled=False)
        mod = get_module_by_name(model, name)
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear")

        S, _cov_tokens, _cov_sec, _chol_sec, _cache_hit = get_or_compute_whitening_S(
            model,
            name,
            dataloader,
            device,
            cov_max_batches=cov_max_batches,
            cov_max_tokens_total=cov_max_tokens_total,
            cov_sample_tokens_per_batch=cov_sample_tokens_per_batch,
            whitening_damping=whitening_damping,
            whitening_cache_dir=whitening_cache_dir,
            whitening_cache_dtype=whitening_cache_dtype,
            whitening_cache_overwrite=whitening_cache_overwrite,
        )

        W = mod.weight.detach().to(device)
        WS = W.float() @ S
        max_rank = min(WS.shape[0], WS.shape[1])
        k_sigma = max_rank if fit_max_rank is None else min(int(fit_max_rank), max_rank)
        _U, sigma, _Vh = svd_topk(
            WS,
            k=k_sigma,
            method=ws_svd_method,
            niter=ws_svd_niter,
            oversample=ws_svd_oversample,
        )

        lam = estimate_lambda_from_sigma(sigma, fit_max_rank=fit_max_rank, huber_delta=huber_delta, huber_iters=huber_iters)
        lambdas[name] = lam

        del S, W, WS, _U, sigma, _Vh
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return lambdas


def populate_whitening_cache(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    cov_max_batches: Optional[int] = None,
    cov_max_tokens_total: Optional[int] = 131072,
    cov_sample_tokens_per_batch: Optional[int] = 2048,
    whitening_damping: float = 1e-5,
    whitening_cache_dir: str,
    whitening_cache_dtype: torch.dtype = torch.bfloat16,
    whitening_cache_overwrite: bool = False,
    tqdm_enabled: bool = True,
    log_every: int = 20,
) -> None:
    """
    Paper-aligned: compute and cache whitening factors S for all target modules on the original model
    BEFORE any compression/training modifies activations.

    This is used when running in one-pass lambda mode but still wanting paper-style SetS caching.
    """
    module_names = collect_all_target_module_names(model, layer_start, layer_end)
    n_total = len(module_names)
    it = (
        tqdm(enumerate(module_names, 1), total=n_total, desc="Whitening cache prepass", leave=False)
        if tqdm_enabled
        else enumerate(module_names, 1)
    )
    for idx, name in it:
        # Skip existing entries unless overwriting.
        if not whitening_cache_overwrite:
            path = _whitening_cache_entry_path(whitening_cache_dir, name)
            if os.path.isfile(path):
                continue

        if not tqdm_enabled and log_every:
            if idx == 1 or idx % int(log_every) == 0 or idx == n_total:
                _log(f"[Whitening] {idx}/{n_total}: {name}", tqdm_enabled=False)

        S, _cov_tokens, _cov_sec, _chol_sec, _cache_hit = get_or_compute_whitening_S(
            model,
            name,
            dataloader,
            device,
            cov_max_batches=cov_max_batches,
            cov_max_tokens_total=cov_max_tokens_total,
            cov_sample_tokens_per_batch=cov_sample_tokens_per_batch,
            whitening_damping=whitening_damping,
            whitening_cache_dir=whitening_cache_dir,
            whitening_cache_dtype=whitening_cache_dtype,
            whitening_cache_overwrite=whitening_cache_overwrite,
        )
        del S
        if device.type == "cuda":
            torch.cuda.empty_cache()


# -------------------------
# Training (feature-preserved weight update) layer-by-layer
# -------------------------
def train_one_layer(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    layer_idx: int,
    dense_weights: Dict[str, torch.Tensor],
    *,
    train_steps: int,
    lr: float,
    max_tokens_per_module_per_batch: int = 2048,
    tqdm_enabled: bool = True,
    train_log_every: int = 50,
) -> None:
    """
    Update only minor components (Bm, Am), freeze Wv and Wp.
    Streaming loss per module: || DFSVD(x) - (x @ W_dense^T) ||^2

    Important trick: run model forward under torch.no_grad() to collect per-module inputs (x),
    then compute y_pred (and the loss graph) outside no_grad via forward_flat(x_detached).
    """
    # collect params in this layer
    train_params: List[nn.Parameter] = []
    for sub in TARGET_SUBMODULES:
        full_name = f"model.layers.{layer_idx}.{sub}"
        mod = get_module_by_name(model, full_name)
        if isinstance(mod, DFSVDFactorizedLinear) and mod.update_rank > 0:
            train_params += [p for p in mod.parameters() if p.requires_grad]

    if not train_params:
        return

    # For DF-SVD, Bm/Am are initialized to match minor components; applying AdamW weight_decay
    # will shrink them towards 0 even when gradients are small. Use weight_decay=0 by default.
    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.0)

    # Defensive: clear NaNs in trainable params before training.
    with torch.no_grad():
        for p in train_params:
            if torch.isnan(p).any():
                p[torch.isnan(p)] = 0  # type: ignore[index]
            if torch.isinf(p).any():
                p[torch.isinf(p)] = 0  # type: ignore[index]

    x_cache: Dict[str, torch.Tensor] = {}

    def make_pre_hook(full_name: str):
        def hook(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
            x = inp[0].detach()
            x2d = x.reshape(-1, x.shape[-1])
            if x2d.shape[0] > max_tokens_per_module_per_batch:
                idx = torch.randperm(x2d.shape[0], device=x2d.device)[:max_tokens_per_module_per_batch]
                x2d = x2d[idx]
            if not torch.isfinite(x2d).all():
                mask = torch.isfinite(x2d).all(dim=1)
                x2d = x2d[mask]
                if x2d.numel() == 0:
                    return
            x_cache[full_name] = x2d
        return hook

    handles = []
    for sub in TARGET_SUBMODULES:
        full_name = f"model.layers.{layer_idx}.{sub}"
        mod = get_module_by_name(model, full_name)
        if isinstance(mod, DFSVDFactorizedLinear) and full_name in dense_weights:
            handles.append(mod.register_forward_pre_hook(make_pre_hook(full_name)))

    model.train()
    data_iter = iter(dataloader)

    pbar = None
    if tqdm_enabled:
        pbar = tqdm(range(train_steps), desc=f"Train layer {layer_idx}", leave=False)
        step_iter = pbar
    else:
        step_iter = range(train_steps)

    last_loss: Optional[float] = None
    for step in step_iter:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        opt.zero_grad(set_to_none=True)
        x_cache.clear()

        with torch.no_grad():
            _ = model(**batch)

        if not x_cache:
            # Usually means activations became non-finite and got filtered out, or hooks didn't fire.
            if pbar is not None:
                pbar.set_postfix({"loss": "no_inputs"})
            continue

        loss = torch.zeros((), device=device)
        for full_name, x2d in x_cache.items():
            W_dense = dense_weights[full_name]  # (out,in) on device
            mod = get_module_by_name(model, full_name)
            # Compute in the factor dtype to avoid float16 overflow in factorized weights.
            x_in = x2d
            try:
                if hasattr(mod, "Wv"):
                    x_in = x2d.to(getattr(mod, "Wv").dtype)  # type: ignore[attr-defined]
            except Exception:
                x_in = x2d
            # Compute target in fp32 for stability.
            y_tgt = x_in.float() @ W_dense.float().t()
            y_pred = mod.forward_flat(x_in)  # type: ignore[attr-defined]
            y_pred_f = y_pred.float()
            # Filter non-finite rows (can happen with extreme factors).
            if y_tgt.dim() == 2 and y_pred_f.dim() == 2:
                m = torch.isfinite(y_tgt).all(dim=1) & torch.isfinite(y_pred_f).all(dim=1)
                if m.any():
                    loss = loss + F.mse_loss(y_pred_f[m], y_tgt[m])
                else:
                    continue
            else:
                if torch.isfinite(y_tgt).all() and torch.isfinite(y_pred_f).all():
                    loss = loss + F.mse_loss(y_pred_f, y_tgt)
                else:
                    continue

        # Keep loss scale roughly constant regardless of how many hooks fired.
        loss = loss / float(max(len(x_cache), 1))

        if not loss.requires_grad:
            # This can happen if all rows were filtered as non-finite and no loss term contributed.
            if pbar is not None:
                pbar.set_postfix({"loss": "no_grad"})
            continue
        if not torch.isfinite(loss):
            if pbar is not None:
                pbar.set_postfix({"loss": "nonfinite"})
            # Skip update to avoid poisoning subsequent layers.
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
        opt.step()
        last_loss = float(loss.detach().cpu().item())
        if pbar is not None:
            pbar.set_postfix({"loss": last_loss})
        else:
            if train_log_every and (step == 0 or (step + 1) % int(train_log_every) == 0 or (step + 1) == int(train_steps)):
                _log(
                    f"[Train] layer={layer_idx} step={step + 1}/{train_steps} loss={last_loss:.6g}",
                    tqdm_enabled=False,
                )

    for h in handles:
        h.remove()


# -------------------------
# Compression (per layer, per module)
# -------------------------
def compress_layer_modules(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    layer_idx: int,
    lambdas: Optional[Dict[str, float]],
    lambda_minmax: Optional[Tuple[float, float]],
    rank_cfg: RankAllocConfig,
    *,
    cov_max_batches: Optional[int] = None,
    cov_max_tokens_total: Optional[int] = 131072,
    cov_sample_tokens_per_batch: Optional[int] = 2048,
    whitening_damping: float = 1e-5,
    running_lambda_state: Optional[Dict[str, float]] = None,  # for one-pass running min/max
    ws_svd_method: str = "randomized",
    ws_svd_niter: int = 2,
    ws_svd_oversample: int = 8,
    factor_dtype: torch.dtype = torch.float32,
    whitening_cache_dir: Optional[str] = None,
    whitening_cache_dtype: torch.dtype = torch.bfloat16,
    whitening_cache_overwrite: bool = False,
    timing_modules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compress 7 matrices in one layer.
    Returns original dense weights (on device) for that layer (used during train_one_layer).
    """
    dense_weights: Dict[str, torch.Tensor] = {}

    for sub in TARGET_SUBMODULES:
        mod_t0 = time.perf_counter()
        name = f"model.layers.{layer_idx}.{sub}"
        mod = get_module_by_name(model, name)
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear")

        W_dense = mod.weight.detach().to(device)
        dense_weights[name] = W_dense

        # Whitening (paper Alg.1): cache S computed from the original (teacher) activations.
        # If `whitening_cache_dir` is set, we load S from cache (or compute+save when missing).
        S, cov_tokens, cov_sec, chol_sec, cache_hit = get_or_compute_whitening_S(
            model,
            name,
            dataloader,
            device,
            cov_max_batches=cov_max_batches,
            cov_max_tokens_total=cov_max_tokens_total,
            cov_sample_tokens_per_batch=cov_sample_tokens_per_batch,
            whitening_damping=whitening_damping,
            whitening_cache_dir=whitening_cache_dir,
            whitening_cache_dtype=whitening_cache_dtype,
            whitening_cache_overwrite=whitening_cache_overwrite,
        )

        # Compute WS SVD once (truncated) and reuse for lambda + factorization.
        WS = W_dense.float() @ S
        max_rank = min(WS.shape[0], WS.shape[1])
        k_base = compute_base_rank(W_dense.shape[0], W_dense.shape[1], rank_cfg.compression_ratio)
        k_need = k_base
        if rank_cfg.fit_max_rank is not None:
            k_need = max(k_need, min(int(rank_cfg.fit_max_rank), max_rank))
        k_need = min(max_rank, max(1, int(k_need)))

        t0 = time.perf_counter()
        U_need, s_need, Vh_need = svd_topk(
            WS,
            k=k_need,
            method=ws_svd_method,
            niter=ws_svd_niter,
            oversample=ws_svd_oversample,
        )
        svd_sec = time.perf_counter() - t0

        # lambda_norm
        if lambdas is not None and lambda_minmax is not None:
            lam = lambdas[name]
            lam_min, lam_max = lambda_minmax
        else:
            lam = estimate_lambda_from_sigma(
                s_need,
                fit_max_rank=rank_cfg.fit_max_rank,
                huber_delta=rank_cfg.huber_delta,
                huber_iters=rank_cfg.huber_iters,
            )

            if running_lambda_state is None:
                running_lambda_state = {"min": lam, "max": lam}
            else:
                running_lambda_state["min"] = min(running_lambda_state["min"], lam)
                running_lambda_state["max"] = max(running_lambda_state["max"], lam)

            lam_min, lam_max = running_lambda_state["min"], running_lambda_state["max"]

        lam_norm = 0.0 if lam_max == lam_min else (lam - lam_min) / (lam_max - lam_min)

        # ranks
        k_trunc, r_up = compute_ranks_for_module(
            out_features=W_dense.shape[0],
            in_features=W_dense.shape[1],
            compression_ratio=rank_cfg.compression_ratio,
            base_update_rank=rank_cfg.base_update_rank,
            lambda_norm=lam_norm,
        )

        # Build factorized module from the precomputed top-k SVD.
        k_trunc = min(int(k_trunc), int(U_need.shape[1]))
        k_trunc = max(1, k_trunc)
        r_up = min(int(r_up), k_trunc)

        U_k = U_need[:, :k_trunc]
        s_k = s_need[:k_trunc]
        Vh_k = Vh_need[:k_trunc, :]

        # DF-SVD (NOT SVD-LLM) factorization:
        # WS = U Σ V^T  =>  W = U Σ V^T S^{-1}
        # Keep ALL singular values in Wu (UΣ), and fix:
        #   Wv = V^T S^{-1}
        # so that (Wv X) becomes isotropic (paper: Hessian = 2I).
        Wu_init = U_k * s_k  # (out,k) = U diag(s)
        # Avoid forming S^{-1}; solve S^T * (Wv^T) = V^T for Wv.
        Wv = torch.linalg.solve_triangular(S.t(), Vh_k.t(), upper=True).t()  # (k,in) = V^T S^{-1}

        # split Wu_init into principal + minor via SVD(Wu_init)
        try:
            U2, s2, Vh2 = torch.linalg.svd(Wu_init, full_matrices=False)
        except RuntimeError:
            U2, s2, Vh2 = torch.linalg.svd(Wu_init.cpu(), full_matrices=False)
            U2, s2, Vh2 = U2.to(device), s2.to(device), Vh2.to(device)

        r = min(int(r_up), int(k_trunc))
        p = int(k_trunc) - r

        if p > 0:
            Wp = (U2[:, :p] * s2[:p]) @ Vh2[:p, :]
        else:
            Wp = torch.zeros((Wu_init.shape[0], Wu_init.shape[1]), device=device, dtype=torch.float32)

        if r > 0:
            Um = U2[:, p:]
            sm = s2[p:].clamp_min(1e-12)
            Vhm = Vh2[p:, :]
            sqrt_sm = torch.sqrt(sm)
            Bm = Um * sqrt_sm
            Am = sqrt_sm.unsqueeze(1) * Vhm
        else:
            Bm = None
            Am = None

        new_mod = DFSVDFactorizedLinear(
            in_features=W_dense.shape[1],
            out_features=W_dense.shape[0],
            rank_k=int(k_trunc),
            update_rank=int(r),
            dtype=factor_dtype,
        ).to(device)
        new_mod.set_factors(
            Wv=Wv.to(factor_dtype),
            Wp=Wp.to(factor_dtype),
            Bm=None if Bm is None else Bm.to(factor_dtype),
            Am=None if Am is None else Am.to(factor_dtype),
        )
        new_mod.Wv.requires_grad_(False)
        new_mod.Wp.requires_grad_(False)
        set_module_by_name(model, name, new_mod)

        if timing_modules is not None:
            total_sec = time.perf_counter() - mod_t0
            other_sec = total_sec - (cov_sec + chol_sec + svd_sec)
            timing_modules.append(
                {
                    "layer": int(layer_idx),
                    "name": name,
                    "sub": sub,
                    "in_features": int(mod.in_features),
                    "out_features": int(mod.out_features),
                    "cov_tokens": int(cov_tokens),
                    "cov_sec": float(cov_sec),
                    "chol_sec": float(chol_sec),
                    "svd_sec": float(svd_sec),
                    "other_sec": float(other_sec),
                    "total_sec": float(total_sec),
                    "whitening_cache_hit": bool(cache_hit),
                    "ws_svd_method": ws_svd_method,
                    "k_need": int(k_need),
                    "k_trunc": int(k_trunc),
                    "r_up": int(r_up),
                    "lambda": float(lam),
                    "lambda_norm": float(lam_norm),
                }
            )

        del S, WS, U_need, s_need, Vh_need, U_k, s_k, Vh_k, Wu_init, Wv, U2, s2, Vh2, Wp, Bm, Am
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return dense_weights


# -------------------------
# Saving / Loading
# -------------------------
def extract_dfsvd_manifest(model: nn.Module, layer_start: int = 0, layer_end: Optional[int] = None) -> Dict[str, Any]:
    layers = model.model.layers
    if layer_end is None:
        layer_end = len(layers)
    items = []
    for i in range(layer_start, layer_end):
        for sub in TARGET_SUBMODULES:
            name = f"model.layers.{i}.{sub}"
            mod = get_module_by_name(model, name)
            if isinstance(mod, DFSVDFactorizedLinear):
                items.append(
                    {
                        "name": name,
                        "in_features": mod.in_features,
                        "out_features": mod.out_features,
                        "rank_k": mod.rank_k,
                        "update_rank": mod.update_rank,
                    }
                )
    return {"dfsvd_version": 1, "items": items}


def save_dfsvd_checkpoint(model: nn.Module, tokenizer: Any, out_dir: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dfsvd_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), os.path.join(out_dir, "dfsvd_state.pt"))
    tokenizer.save_pretrained(out_dir)

def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _timing_write(out_dir: str, filename: str, timing: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    return path


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--compression_ratio", type=float, default=0.4)
    ap.add_argument("--base_update_rank", type=int, default=8)

    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--calib_sequences", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument(
        "--factor_dtype",
        type=str,
        default="float32",
        help="Dtype for DF-SVD factor matrices (Wv/Wp/Bm/Am). Use float32/bfloat16 to avoid float16 overflow from S^{-1}.",
    )
    ap.add_argument("--seed", type=int, default=42)

    # layer range (for debug / partial run)
    ap.add_argument("--layer_start", type=int, default=0)
    ap.add_argument("--layer_end", type=int, default=None)

    # lambda normalization mode (paper uses global min/max over all compressible weights)
    ap.add_argument(
        "--lambda_one_pass",
        action="store_true",
        help="Use running min/max for lambda normalization (faster but can collapse ranks). Default is global min/max (paper-aligned).",
    )
    ap.add_argument(
        "--lambda_two_pass",
        action="store_true",
        help="(Deprecated) Kept for backwards compatibility. Global min/max is now the default; pass --lambda_one_pass to opt into running normalization.",
    )
    ap.add_argument("--fit_max_rank", type=int, default=256)
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--huber_iters", type=int, default=20)

    # covariance controls
    ap.add_argument("--cov_max_batches", type=int, default=None)
    ap.add_argument("--cov_max_tokens_total", type=int, default=524288)
    ap.add_argument("--cov_sample_tokens_per_batch", type=int, default=2048)
    ap.add_argument("--whitening_damping", type=float, default=1e-5)
    ap.add_argument(
        "--whitening_cache_dir",
        type=str,
        default=None,
        help="Cache per-module Cholesky factor S (teacher/original activations) to align with paper Alg.1 and avoid re-collecting cov during compression.",
    )
    ap.add_argument(
        "--whitening_cache_dtype",
        type=str,
        default="float32",
        help="Dtype for cached whitening factor S on disk (float16/bfloat16/float32).",
    )
    ap.add_argument(
        "--whitening_cache_overwrite",
        action="store_true",
        help="Overwrite existing whitening cache entries.",
    )

    # WS SVD (truncated) controls
    ap.add_argument(
        "--ws_svd_method",
        type=str,
        default="randomized",
        choices=["randomized", "full"],
        help="SVD method for WS=W@S. 'randomized' uses torch.svd_lowrank to approximate top-k; 'full' uses torch.linalg.svd.",
    )
    ap.add_argument("--ws_svd_niter", type=int, default=2, help="Power iterations for randomized SVD (torch.svd_lowrank).")
    ap.add_argument("--ws_svd_oversample", type=int, default=8, help="Oversampling for randomized SVD (torch.svd_lowrank).")

    # training
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--train_steps", type=int, default=200)
    ap.add_argument("--train_lr", type=float, default=5e-4)
    ap.add_argument("--train_max_tokens_per_module_per_batch", type=int, default=2048)
    ap.add_argument(
        "--train_log_every",
        type=int,
        default=50,
        help="When tqdm is disabled, print a training loss line every N steps (0 disables).",
    )

    # logging / progress bars
    ap.add_argument(
        "--tqdm",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Progress bars. 'auto' disables tqdm when stderr is not a TTY (prevents log spam).",
    )
    ap.add_argument(
        "--lambda_log_every",
        type=int,
        default=20,
        help="When tqdm is disabled, print a lambda-prepass progress line every N matrices (0 disables).",
    )

    # timing
    ap.add_argument(
        "--timing_file",
        type=str,
        default="dfsvd_timing.json",
        help="Write timing breakdown JSON to <output_dir>/<timing_file>.",
    )

    args = ap.parse_args()
    set_seed(args.seed)
    tqdm_enabled = resolve_tqdm_enabled(args.tqdm)
    if args.tqdm == "auto" and not tqdm_enabled:
        _log("[Info] tqdm disabled (stderr is not a TTY). Pass --tqdm on to force progress bars.", tqdm_enabled=False)

    run_start = time.perf_counter()
    timing: Dict[str, Any] = {
        "started_at": _now_iso(),
        "args": vars(args),
        "stages": [],
        "layers": [],
        "modules": [],
    }
    timing["args"]["tqdm_enabled"] = bool(tqdm_enabled)

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    factor_dtype = parse_dtype(args.factor_dtype)
    whitening_cache_dtype = parse_dtype(args.whitening_cache_dtype)
    whitening_cache_dir = args.whitening_cache_dir

    if whitening_cache_dir:
        _write_whitening_cache_meta(
            whitening_cache_dir,
            {
                "created_at": _now_iso(),
                "model_id": args.model_id,
                "seq_len": int(args.seq_len),
                "calib_sequences": int(args.calib_sequences),
                "batch_size": int(args.batch_size),
                "cov_max_tokens_total": args.cov_max_tokens_total,
                "cov_sample_tokens_per_batch": args.cov_sample_tokens_per_batch,
                "seed": int(args.seed),
                "whitening_damping": float(args.whitening_damping),
                "whitening_cache_dtype": args.whitening_cache_dtype,
            },
            tqdm_enabled=tqdm_enabled,
        )

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    timing["stages"].append({"name": "load_tokenizer", "sec": time.perf_counter() - t0})

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    model.config.use_cache = False  # avoid kv-cache during calibration/training
    timing["stages"].append({"name": "load_model", "sec": time.perf_counter() - t0})

    t0 = time.perf_counter()
    dataloader = build_calibration_dataloader(
        tokenizer,
        num_sequences=args.calib_sequences,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    timing["stages"].append({"name": "build_calib_dataloader", "sec": time.perf_counter() - t0})

    rank_cfg = RankAllocConfig(
        compression_ratio=args.compression_ratio,
        base_update_rank=args.base_update_rank,
        fit_max_rank=args.fit_max_rank,
        huber_delta=args.huber_delta,
        huber_iters=args.huber_iters,
    )

    # Global min/max across all target weights (paper Algorithm 2).
    # We get this by a prepass that computes lambda for every target matrix.
    # Note: this is expensive but avoids pathological rank collapse from running min/max.
    use_global_lambda_norm = not bool(args.lambda_one_pass)
    if args.lambda_one_pass and args.lambda_two_pass:
        raise ValueError("Conflicting flags: --lambda_one_pass and --lambda_two_pass")

    lambdas = None
    lambda_minmax = None
    if use_global_lambda_norm:
        t0 = time.perf_counter()
        lambdas = compute_lambdas_two_pass(
            model,
            dataloader,
            device,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            fit_max_rank=args.fit_max_rank,
            huber_delta=args.huber_delta,
            huber_iters=args.huber_iters,
            cov_max_batches=args.cov_max_batches,
            cov_max_tokens_total=args.cov_max_tokens_total,
            cov_sample_tokens_per_batch=args.cov_sample_tokens_per_batch,
            whitening_damping=args.whitening_damping,
            ws_svd_method=args.ws_svd_method,
            ws_svd_niter=args.ws_svd_niter,
            ws_svd_oversample=args.ws_svd_oversample,
            tqdm_enabled=tqdm_enabled,
            log_every=args.lambda_log_every,
            whitening_cache_dir=whitening_cache_dir,
            whitening_cache_dtype=whitening_cache_dtype,
            whitening_cache_overwrite=bool(args.whitening_cache_overwrite),
        )
        vals = list(lambdas.values())
        lambda_minmax = (min(vals), max(vals))
        timing["stages"].append({"name": "lambda_global_prepass", "sec": time.perf_counter() - t0, "min": lambda_minmax[0], "max": lambda_minmax[1]})

    # Paper Alg.1: optionally build whitening cache on the teacher/original model before compression.
    if (not use_global_lambda_norm) and whitening_cache_dir:
        t0 = time.perf_counter()
        populate_whitening_cache(
            model,
            dataloader,
            device,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            cov_max_batches=args.cov_max_batches,
            cov_max_tokens_total=args.cov_max_tokens_total,
            cov_sample_tokens_per_batch=args.cov_sample_tokens_per_batch,
            whitening_damping=args.whitening_damping,
            whitening_cache_dir=whitening_cache_dir,
            whitening_cache_dtype=whitening_cache_dtype,
            whitening_cache_overwrite=bool(args.whitening_cache_overwrite),
            tqdm_enabled=tqdm_enabled,
            log_every=args.lambda_log_every,
        )
        timing["stages"].append({"name": "whitening_cache_prepass", "sec": time.perf_counter() - t0})

    layers = model.model.layers
    layer_end = args.layer_end if args.layer_end is not None else len(layers)

    # One-pass mode needs a persistent min/max across modules to compute lambda_norm.
    running_lambda_state = None if use_global_lambda_norm else {"min": float("inf"), "max": float("-inf")}

    timing_path: Optional[str] = None
    try:
        for i in range(args.layer_start, layer_end):
            layer_rec: Dict[str, Any] = {"layer": i, "compress_sec": None, "train_sec": None, "total_sec": None}
            layer_t0 = time.perf_counter()

            t0 = time.perf_counter()
            dense_w = compress_layer_modules(
                model,
                dataloader,
                device,
                i,
                lambdas=lambdas,
                lambda_minmax=lambda_minmax,
                rank_cfg=rank_cfg,
                cov_max_batches=args.cov_max_batches,
                cov_max_tokens_total=args.cov_max_tokens_total,
                cov_sample_tokens_per_batch=args.cov_sample_tokens_per_batch,
                whitening_damping=args.whitening_damping,
                running_lambda_state=running_lambda_state,
                ws_svd_method=args.ws_svd_method,
                ws_svd_niter=args.ws_svd_niter,
                ws_svd_oversample=args.ws_svd_oversample,
                factor_dtype=factor_dtype,
                whitening_cache_dir=whitening_cache_dir,
                whitening_cache_dtype=whitening_cache_dtype,
                whitening_cache_overwrite=False,
                timing_modules=timing["modules"],
            )
            layer_rec["compress_sec"] = time.perf_counter() - t0

            if args.do_train:
                t0 = time.perf_counter()
                train_one_layer(
                    model,
                    dataloader,
                    device,
                    i,
                    dense_weights=dense_w,
                    train_steps=args.train_steps,
                    lr=args.train_lr,
                    max_tokens_per_module_per_batch=args.train_max_tokens_per_module_per_batch,
                    tqdm_enabled=tqdm_enabled,
                    train_log_every=args.train_log_every,
                )
                layer_rec["train_sec"] = time.perf_counter() - t0

            del dense_w
            if device.type == "cuda":
                torch.cuda.empty_cache()

            layer_rec["total_sec"] = time.perf_counter() - layer_t0
            timing["layers"].append(layer_rec)
            if not tqdm_enabled:
                _log(
                    f"[Layer] {i}: compress_sec={layer_rec['compress_sec']:.2f} train_sec={(layer_rec.get('train_sec') or 0.0):.2f} total_sec={layer_rec['total_sec']:.2f}",
                    tqdm_enabled=False,
                )

        t0 = time.perf_counter()
        manifest = extract_dfsvd_manifest(model, args.layer_start, layer_end)
        save_dfsvd_checkpoint(model, tokenizer, args.output_dir, manifest)
        timing["stages"].append({"name": "save_checkpoint", "sec": time.perf_counter() - t0})

        _log(f"[OK] Saved DF-SVD checkpoint to: {args.output_dir}", tqdm_enabled=tqdm_enabled)
    finally:
        timing["ended_at"] = _now_iso()
        timing["total_sec"] = time.perf_counter() - run_start
        try:
            timing_path = _timing_write(args.output_dir, args.timing_file, timing)
        except Exception as e:
            _log(f"[Warn] Failed to write timing file: {e}", tqdm_enabled=tqdm_enabled)

    if timing_path:
        total = float(timing.get("total_sec", 0.0))
        _log(f"[Time] total={total:.2f}s timing_json={timing_path}", tqdm_enabled=tqdm_enabled)


if __name__ == "__main__":
    main()
