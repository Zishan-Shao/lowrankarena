#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAES-SVD reproduction (calibration + whitening + ACES beta selection + layer-wise decomposition).

This script follows the SAES-SVD calibration formulation:
  - streaming H_l = (2/N) sum x x^T
  - streaming Delta_l = (2/N) sum (x_f - x) x^T
  - whitening by right-multiplying (H_l + lambda I)^(-1/2) via Cholesky
  - factorization from G_l(beta)=W_l (H_l + beta Delta_l) (H_l + lambda I)^(-1/2)
  - optional ACES-style fixed-subspace beta search with guardrails.

IMPORTANT IMPLEMENTATION NOTE:
  When using Cholesky H+lambda I = L L^T, a consistent way to obtain the
  desired right-side inverse (H+lambda I)^{-1} is to use both L^{-T} and L^{-1}.
  This implementation computes:
    G = W (H + beta Delta) L^{-T}
    then B = ... V^T L^{-1}
  so that the reconstructed weight becomes:
    W_hat = (U S V^T) L^{-1} = W (H+beta Delta) L^{-T} L^{-1} = W (H+beta Delta) (H+lambda I)^{-1}

Usage example (Llama-2-7b-hf, from the LowRankArena repository root):
CUDA_VISIBLE_DEVICES=3 python -u compress/svd/SAES-SVD-reproduction/saes_svd.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --output_dir /path/to/llama2_saes_r0.4 \
  --compression_ratio 0.4 \
  --seq_len 2048 \
  --calib_sequences 128 \
  --batch_size 1 \
  --max_tokens_total 262144 \
  --beta_mode aces \
  --device cuda \
  --teacher_device cuda \
  --dtype bfloat16 \
  --teacher_dtype bfloat16 \
  --factor_dtype bfloat16

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_MIX_CALIB_IMPORT_ERR: Optional[str] = None
try:
    from utils.data_utils import get_mixed_calib_train_data
except Exception as e:
    get_mixed_calib_train_data = None
    _MIX_CALIB_IMPORT_ERR = repr(e)
    try:
        import importlib.util as _ilu

        _data_utils_path = os.path.join(_REPO_ROOT, "utils", "data_utils.py")
        if os.path.isfile(_data_utils_path):
            _spec = _ilu.spec_from_file_location("local_utils_data_utils", _data_utils_path)
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore
                _fn = getattr(_mod, "get_mixed_calib_train_data", None)
                if callable(_fn):
                    get_mixed_calib_train_data = _fn
                    _MIX_CALIB_IMPORT_ERR = None
    except Exception as e2:
        _MIX_CALIB_IMPORT_ERR = f"{_MIX_CALIB_IMPORT_ERR}; fallback_error={repr(e2)}"


TARGET_SUBMODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# Within one Transformer block, o_proj depends on q/k/v outputs and down_proj depends
# on gate/up outputs. For stricter "upstream already compressed" semantics, compress in 2 phases.
PHASE1_SUBMODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
]
PHASE2_SUBMODULES = [
    "self_attn.o_proj",
    "mlp.down_proj",
]


def resolve_tqdm_enabled(mode: str) -> bool:
    mode = str(mode).lower().strip()
    if mode == "on":
        return True
    if mode == "off":
        return False
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


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _timing_write(out_dir: str, filename: str, timing: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_dtype(s: str) -> torch.dtype:
    s = str(s).lower().strip()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def parse_device(s: str, fallback: Optional[torch.device] = None) -> torch.device:
    ss = str(s).lower().strip()
    if ss == "auto":
        if fallback is not None:
            return fallback
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(ss)


def _ensure_tokenizer(tokenizer_obj: Any, model_hint: str, hf_token: Optional[str] = None) -> Any:
    try:
        if tokenizer_obj is not None and not isinstance(tokenizer_obj, bool) and callable(tokenizer_obj):
            return tokenizer_obj
    except Exception:
        pass

    hints: List[str] = []
    env_hint = os.getenv("SVDLLM_TOKENIZER_MODEL")
    if env_hint and str(env_hint).strip():
        hints.append(str(env_hint).strip())
    if model_hint and str(model_hint).strip():
        h = str(model_hint).strip()
        if h not in hints:
            hints.append(h)

    def _load_auto_tokenizer(hint: str, use_fast: bool) -> Any:
        kwargs: Dict[str, Any] = {"trust_remote_code": True, "use_fast": bool(use_fast)}
        if hf_token is not None:
            kwargs["token"] = hf_token
        try:
            return AutoTokenizer.from_pretrained(hint, **kwargs)
        except TypeError:
            if "token" in kwargs:
                kwargs.pop("token", None)
                kwargs["use_auth_token"] = hf_token
                return AutoTokenizer.from_pretrained(hint, **kwargs)
            raise

    for hint in hints:
        for use_fast in (True, False):
            try:
                tok = _load_auto_tokenizer(hint, use_fast=use_fast)
                if tok is not None and not isinstance(tok, bool) and callable(tok):
                    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                        tok.pad_token = tok.eos_token
                    return tok
            except Exception:
                continue

        try:
            from transformers import LlamaTokenizer, LlamaTokenizerFast

            for cls in (LlamaTokenizerFast, LlamaTokenizer):
                try:
                    kwargs: Dict[str, Any] = {}
                    if hf_token is not None:
                        kwargs["token"] = hf_token
                    tok = cls.from_pretrained(hint, **kwargs)
                except TypeError:
                    kwargs = {}
                    if hf_token is not None:
                        kwargs["use_auth_token"] = hf_token
                    tok = cls.from_pretrained(hint, **kwargs)
                if tok is not None and not isinstance(tok, bool) and callable(tok):
                    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                        tok.pad_token = tok.eos_token
                    return tok
        except Exception:
            pass

    raise TypeError(
        "Tokenizer is not callable and could not be reconstructed. "
        "Try setting --tokenizer_model or env SVDLLM_TOKENIZER_MODEL."
    )


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


def compute_base_rank(out_features: int, in_features: int, compression_ratio: float) -> int:
    m = int(out_features)
    n = int(in_features)
    k = int(math.floor((m * n * float(compression_ratio)) / float(m + n)))
    return max(1, min(k, min(m, n)))


def build_calibration_dataloader(
    tokenizer: Any,
    *,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    num_sequences: int = 128,
    seq_len: int = 2048,
    batch_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    raw = load_dataset(dataset_name, dataset_config, split=split)
    raw = raw.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)
    raw = raw.shuffle(seed=seed)
    raw = raw.select(range(min(num_sequences * 2, len(raw))))

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


def _normalize_calib_sample(
    sample: Dict[str, Any],
    *,
    seq_len: int,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    ids = sample.get("input_ids", None)
    if ids is None:
        raise KeyError("Mixed calibration sample missing 'input_ids'")
    attn = sample.get("attention_mask", None)

    ids_t = ids if torch.is_tensor(ids) else torch.tensor(ids, dtype=torch.long)
    if ids_t.dim() == 2 and ids_t.shape[0] == 1:
        ids_t = ids_t[0]
    elif ids_t.dim() > 1:
        ids_t = ids_t.reshape(-1)
    ids_t = ids_t.to(dtype=torch.long)

    if attn is None:
        attn_t = torch.ones_like(ids_t, dtype=torch.long)
    else:
        attn_t = attn if torch.is_tensor(attn) else torch.tensor(attn, dtype=torch.long)
        if attn_t.dim() == 2 and attn_t.shape[0] == 1:
            attn_t = attn_t[0]
        elif attn_t.dim() > 1:
            attn_t = attn_t.reshape(-1)
        attn_t = attn_t.to(dtype=torch.long)

    if int(ids_t.numel()) > int(seq_len):
        ids_t = ids_t[: int(seq_len)]
        attn_t = attn_t[: int(seq_len)]
    elif int(ids_t.numel()) < int(seq_len):
        pad_len = int(seq_len) - int(ids_t.numel())
        ids_t = torch.cat([ids_t, torch.full((pad_len,), int(pad_token_id), dtype=torch.long)], dim=0)
        attn_t = torch.cat([attn_t, torch.zeros((pad_len,), dtype=torch.long)], dim=0)

    return {"input_ids": ids_t.contiguous(), "attention_mask": attn_t.contiguous()}


def _mixed_calib_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_mixed_calibration_dataloader(
    tokenizer: Any,
    *,
    num_sequences: int,
    seq_len: int,
    batch_size: int,
    seed: int,
    bucket_props: str,
    bucket_lm_datasets: str,
    bucket_ling_datasets: str,
    c4_stream: bool,
    per_bucket: bool,
    dump_bucket_debug: bool,
) -> DataLoader:
    """
    Build a mixed calibration loader for SAES stats collection:
      - LM bucket (e.g., wikitext2/ptb/c4_stream)
      - Linguistic bucket (mapped to data_utils INST bucket, e.g., cola/sst2/blimp-like)
    """
    if get_mixed_calib_train_data is None:
        detail = _MIX_CALIB_IMPORT_ERR or "unknown import error"
        raise RuntimeError(
            "utils.data_utils.get_mixed_calib_train_data is unavailable in this environment. "
            f"import_detail={detail}"
        )

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    mixed_samples = get_mixed_calib_train_data(
        tokenizer=tokenizer,
        nsamples=int(num_sequences),
        seqlen=int(seq_len),
        seed=int(seed),
        bucket_props=str(bucket_props),
        bucket_lm_datasets=str(bucket_lm_datasets),
        bucket_inst_datasets=str(bucket_ling_datasets),
        bucket_math_datasets="",
        c4_stream=bool(c4_stream),
        per_bucket=bool(per_bucket),
        dump_bucket_debug=bool(dump_bucket_debug),
    )
    if not mixed_samples:
        raise RuntimeError("Mixed calibration dataloader is empty. Check your bucket datasets and cache.")

    normalized = [
        _normalize_calib_sample(x, seq_len=int(seq_len), pad_token_id=int(pad_token_id))
        for x in mixed_samples
    ]
    return DataLoader(normalized, batch_size=int(batch_size), shuffle=False, collate_fn=_mixed_calib_collate)


def _flatten_to_d_by_n(x: torch.Tensor) -> torch.Tensor:
    """
    Convert activations to [d_in, N] float32 for SAES statistics accumulation.
    Accepts [B,T,d], [T,d], [B,d], or arbitrary [...,d].
    """
    if x is None:
        raise ValueError("x is None")
    if x.dim() == 3:
        x2 = x.reshape(-1, x.shape[-1])
    elif x.dim() == 2:
        x2 = x
    else:
        x2 = x.reshape(-1, x.shape[-1])
    return x2.transpose(0, 1).contiguous().to(dtype=torch.float32)


@dataclass
class SAESStats:
    """
    Streaming SAES stats for one linear layer input dimension d_in:
      H = (2/N) * sum xx^T
      Delta = (2/N) * sum (x_f - x) x^T
    """
    d_in: int
    device: torch.device = torch.device("cpu")
    n_tokens: int = 0
    H: Optional[torch.Tensor] = None
    Delta: Optional[torch.Tensor] = None

    def _lazy_init(self) -> None:
        if self.H is None:
            self.H = torch.zeros((self.d_in, self.d_in), device=self.device, dtype=torch.float32)
        if self.Delta is None:
            self.Delta = torch.zeros((self.d_in, self.d_in), device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, x: torch.Tensor, x_fp: torch.Tensor) -> None:
        X = _flatten_to_d_by_n(x)
        Xf = _flatten_to_d_by_n(x_fp)
        if X.shape != Xf.shape:
            raise ValueError(f"Shape mismatch: X {tuple(X.shape)} vs Xf {tuple(Xf.shape)}")
        if int(X.shape[0]) != int(self.d_in):
            raise ValueError(f"d_in mismatch: expected {self.d_in}, got {int(X.shape[0])}")

        X = X.to(self.device, non_blocking=True)
        Xf = Xf.to(self.device, non_blocking=True)
        self._lazy_init()

        m = int(X.shape[1])
        t = int(self.n_tokens)
        n_new = t + m
        if n_new <= 0:
            return

        gamma = 0.0 if t == 0 else (float(t) / float(n_new))
        if gamma != 1.0:
            self.H.mul_(gamma)      # type: ignore[union-attr]
            self.Delta.mul_(gamma)  # type: ignore[union-attr]

        self.n_tokens = n_new

        s = math.sqrt(2.0 / float(self.n_tokens))
        Xs = X * s
        Xfs = Xf * s
        self.H.add_(Xs @ Xs.transpose(0, 1))                 # type: ignore[union-attr]
        self.Delta.add_((Xfs - Xs) @ Xs.transpose(0, 1))     # type: ignore[union-attr]

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        self._lazy_init()
        return self.H, self.Delta, int(self.n_tokens)  # type: ignore[return-value]


def _safe_cholesky(H: torch.Tensor, damping: float, max_tries: int = 8) -> Tuple[torch.Tensor, float]:
    """
    Returns lower-triangular chol L with H + lam I = L L^T.
    Retry with progressively larger lam if Cholesky fails.

    `damping` is interpreted as an absolute ridge lambda (paper-style), not a
    relative multiplier of H's diagonal statistics.
    """
    if H.dim() != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got {tuple(H.shape)}")
    d = int(H.shape[0])
    eye = torch.eye(d, device=H.device, dtype=H.dtype)
    Hs = 0.5 * (H + H.transpose(0, 1))
    lam = max(float(damping), 1e-12)
    for _ in range(max(1, int(max_tries))):
        try:
            return torch.linalg.cholesky(Hs + lam * eye), lam
        except RuntimeError:
            lam *= 10.0

    # last try with eig projection fallback
    w, q = torch.linalg.eigh(Hs)
    w = torch.clamp(w, min=lam)
    H_spd = (q * w) @ q.transpose(0, 1)
    return torch.linalg.cholesky(H_spd), lam


def right_whiten(M: torch.Tensor, chol_lower: torch.Tensor) -> torch.Tensor:
    """
    Compute M @ inv(L) using triangular solve (no explicit inverse),
    where L is lower-triangular (Cholesky factor).
    """
    rt = chol_lower.transpose(-1, -2)  # L^T
    x_t = torch.linalg.solve_triangular(rt, M.transpose(-1, -2), upper=True)
    return x_t.transpose(-1, -2).contiguous()


def right_whiten_inv_lt(M: torch.Tensor, chol_lower: torch.Tensor) -> torch.Tensor:
    """
    Compute M @ inv(L^T) using triangular solve (no explicit inverse),
    where L is lower-triangular (Cholesky factor).
    Equivalent: (M @ L^{-T})^T = L^{-1} M^T.
    """
    x_t = torch.linalg.solve_triangular(chol_lower, M.transpose(-1, -2), upper=False)
    return x_t.transpose(-1, -2).contiguous()


def _truncated_svd(
    mat: torch.Tensor,
    rank: int,
    *,
    svd_lowrank: bool,
    oversample: int,
    n_iter: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns truncated SVD of mat:
      U_r [m,r], S_r [r], Vh_r [r,n].
    """
    if mat.dim() != 2:
        raise ValueError(f"mat must be 2D, got {tuple(mat.shape)}")
    m, n = int(mat.shape[0]), int(mat.shape[1])
    r = min(max(1, int(rank)), min(m, n))

    if svd_lowrank and r < min(m, n):
        q = min(max(r + max(0, int(oversample)), r + 1), min(m, n))
        try:
            U, S, V = torch.svd_lowrank(mat, q=q, niter=max(0, int(n_iter)))
            idx = torch.argsort(S, descending=True)
            S = S[idx]
            U = U[:, idx]
            V = V[:, idx]
            return U[:, :r], S[:r], V[:, :r].transpose(0, 1)
        except Exception:
            pass

    try:
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    except RuntimeError:
        U, S, Vh = torch.linalg.svd(mat.cpu(), full_matrices=False)
        U = U.to(mat.device)
        S = S.to(mat.device)
        Vh = Vh.to(mat.device)
    return U[:, :r], S[:r], Vh[:r, :]


@torch.no_grad()
def saes_whitened_factorize(
    W: torch.Tensor,
    H: torch.Tensor,
    Delta: Optional[torch.Tensor],
    rank: int,
    beta: float,
    damping: float = 1e-5,
    max_tries: int = 8,
    compute_device: Optional[torch.device] = None,
    compute_dtype: torch.dtype = torch.float32,
    svd_lowrank: bool = True,
    oversample: int = 32,
    n_iter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    SAES factorization (fixed implementation):
      Let H_lam = H + lambda I = L L^T (Cholesky, L lower-triangular).
      Define:
        G(beta) = W (H + beta Delta) L^{-T}
      Then take truncated SVD:
        G_r = U_r S_r V_r^T
      And construct factors:
        A = U_r S_r^(1/2)
        B = S_r^(1/2) V_r^T L^{-1}
      So the reconstructed weight becomes:
        A B = W (H + beta Delta) L^{-T} L^{-1} = W (H + beta Delta) (H + lambda I)^{-1}
    """
    if W.dim() != 2:
        raise ValueError(f"W must be 2D, got {tuple(W.shape)}")
    out_dim, in_dim = int(W.shape[0]), int(W.shape[1])
    if H.shape != (in_dim, in_dim):
        raise ValueError(f"H shape must be {(in_dim, in_dim)}, got {tuple(H.shape)}")
    if Delta is None:
        Delta = torch.zeros_like(H)
    if Delta.shape != (in_dim, in_dim):
        raise ValueError(f"Delta shape must be {(in_dim, in_dim)}, got {tuple(Delta.shape)}")
    if not (0.0 <= float(beta) < 1.0):
        raise ValueError(f"beta must be in [0,1), got {beta}")
    r = min(max(1, int(rank)), min(out_dim, in_dim))
    dev = compute_device if compute_device is not None else W.device

    Wc = W.to(device=dev, dtype=compute_dtype)
    Hc = H.to(device=dev, dtype=compute_dtype)
    Dc = Delta.to(device=dev, dtype=compute_dtype)

    chol, used_lam = _safe_cholesky(Hc, damping=damping, max_tries=max_tries)
    Hbeta = Hc + float(beta) * Dc

    # IMPORTANT: Use L^{-T} here (not L^{-1}), so final AB includes L^{-T}L^{-1} = (H+lambda I)^{-1}.
    G = right_whiten_inv_lt(Wc @ Hbeta, chol)

    U_r, S_r, Vh_r = _truncated_svd(
        G,
        r,
        svd_lowrank=svd_lowrank,
        oversample=oversample,
        n_iter=n_iter,
    )

    sqrtS = torch.sqrt(torch.clamp(S_r, min=0.0))
    A = U_r * sqrtS.unsqueeze(0)

    # Unwhiten V^T with L^{-1} (so AB = ... L^{-1})
    VhL = right_whiten(Vh_r, chol)
    B = sqrtS.unsqueeze(1) * VhL

    return A, B, used_lam


def _project_residual(M: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute P_perp M Q_perp with:
      P_perp = I - U U^T, Q_perp = I - V V^T.
    """
    ut_m = U.transpose(0, 1) @ M
    m_v = M @ V
    ut_m_v = ut_m @ V
    return M - U @ ut_m - m_v @ V.transpose(0, 1) + U @ ut_m_v @ V.transpose(0, 1)


def _objective_ratio(beta: float, a: float, b: float, c: float, A: float, B: float, C: float) -> float:
    num = a * beta * beta + b * beta + c
    den = A * beta * beta + B * beta + C
    if den <= 1e-30:
        return float("inf")
    return float(num / den)


def _objective_energy(beta: float, a: float, b: float, c: float) -> float:
    return float(a * beta * beta + b * beta + c)


def _quadratic_roots(p2: float, p1: float, p0: float) -> List[float]:
    eps = 1e-20
    if abs(p2) <= eps:
        if abs(p1) <= eps:
            return []
        return [float(-p0 / p1)]
    disc = p1 * p1 - 4.0 * p2 * p0
    if disc < 0:
        return []
    sqrt_disc = math.sqrt(max(0.0, disc))
    return [float((-p1 + sqrt_disc) / (2.0 * p2)), float((-p1 - sqrt_disc) / (2.0 * p2))]


def _apply_beta_guardrails(
    beta: float,
    *,
    beta_min: float,
    beta_max: float,
    beta_cap: float,
    beta_shrink: float,
) -> float:
    b_low = max(0.0, float(beta_min))
    b_high = min(float(beta_max), 1.0 - 1e-7)
    if b_high < b_low:
        b_high = b_low

    b = min(max(float(beta), b_low), b_high)          # clip
    b = min(b, float(beta_cap))                       # cap
    b = b * float(beta_shrink)                        # shrink
    b = min(max(b, b_low), min(b_high, float(beta_cap), 1.0 - 1e-7))
    return float(b)


@torch.no_grad()
def select_beta_fixed_subspace_aces(
    W: torch.Tensor,
    H: torch.Tensor,
    Delta: torch.Tensor,
    rank: int,
    *,
    beta_base: float,
    beta_min: float,
    beta_max: float,
    beta_cap: float,
    beta_shrink: float,
    objective: str,
    damping: float,
    max_tries: int,
    compute_device: Optional[torch.device],
    compute_dtype: torch.dtype,
    svd_lowrank: bool,
    oversample: int,
    n_iter: int,
) -> float:
    """
    ACES-style fixed-subspace beta search:
      1) build G0 = W H L^{-T}, Gd = W Delta L^{-T}
      2) freeze top-r subspace of G0 (U,V)
      3) minimize rational quadratic approximation of tail ratio:
           f(beta) = (a beta^2 + b beta + c) / (A beta^2 + B beta + C)
         stationary points satisfy:
           (aB - bA) beta^2 + 2(aC - cA) beta + (bC - cB) = 0
      4) apply guardrails (clip/cap/shrink) to each candidate before scoring.

    objective:
      - "ratio": minimize tail/total ratio surrogate (paper-recommended).
      - "energy": minimize tail energy surrogate (more conservative).
    """
    if Delta is None or float(torch.norm(Delta.float()).item()) == 0.0:
        return _apply_beta_guardrails(
            beta_base,
            beta_min=beta_min,
            beta_max=beta_max,
            beta_cap=beta_cap,
            beta_shrink=beta_shrink,
        )

    dev = compute_device if compute_device is not None else W.device
    Wc = W.to(device=dev, dtype=compute_dtype)
    Hc = H.to(device=dev, dtype=compute_dtype)
    Dc = Delta.to(device=dev, dtype=compute_dtype)

    chol, _ = _safe_cholesky(Hc, damping=damping, max_tries=max_tries)

    # IMPORTANT: Use L^{-T} here (not L^{-1}) to match saes_whitened_factorize().
    G0 = right_whiten_inv_lt(Wc @ Hc, chol)
    Gd = right_whiten_inv_lt(Wc @ Dc, chol)

    if not torch.isfinite(Gd).all():
        return _apply_beta_guardrails(
            beta_base,
            beta_min=beta_min,
            beta_max=beta_max,
            beta_cap=beta_cap,
            beta_shrink=beta_shrink,
        )
    if float(torch.norm(Gd).item()) <= 1e-12:
        return _apply_beta_guardrails(
            beta_base,
            beta_min=beta_min,
            beta_max=beta_max,
            beta_cap=beta_cap,
            beta_shrink=beta_shrink,
        )

    r = min(max(1, int(rank)), min(int(G0.shape[0]), int(G0.shape[1])))
    U_r, _S_r, Vh_r = _truncated_svd(
        G0,
        r,
        svd_lowrank=svd_lowrank,
        oversample=oversample,
        n_iter=n_iter,
    )
    V_r = Vh_r.transpose(0, 1)

    # Numerator coefficients from projected tail energies.
    R0 = _project_residual(G0, U_r, V_r)
    R1 = _project_residual(Gd, U_r, V_r)

    a = float(torch.sum(R1 * R1).item())
    b = float((2.0 * torch.sum(R0 * R1)).item())
    c = float(torch.sum(R0 * R0).item())

    # Denominator coefficients from total energies.
    A = float(torch.sum(Gd * Gd).item())
    B = float((2.0 * torch.sum(G0 * Gd)).item())
    C = float(torch.sum(G0 * G0).item())

    p2 = a * B - b * A
    p1 = 2.0 * (a * C - c * A)
    p0 = b * C - c * B

    objective_mode = str(objective).strip().lower()
    candidates_raw: List[float] = [float(beta_base), float(beta_min), float(beta_max), float(beta_cap)]
    if objective_mode == "ratio":
        candidates_raw.extend(_quadratic_roots(p2, p1, p0))
    elif objective_mode == "energy":
        if abs(a) > 1e-20:
            candidates_raw.append(float(-b / (2.0 * a)))
    else:
        raise ValueError(f"Unknown ACES objective: {objective}")

    candidates = [
        _apply_beta_guardrails(
            cand,
            beta_min=beta_min,
            beta_max=beta_max,
            beta_cap=beta_cap,
            beta_shrink=beta_shrink,
        )
        for cand in candidates_raw
        if math.isfinite(cand)
    ]
    if not candidates:
        candidates = [
            _apply_beta_guardrails(
                beta_base,
                beta_min=beta_min,
                beta_max=beta_max,
                beta_cap=beta_cap,
                beta_shrink=beta_shrink,
            )
        ]

    best_beta = candidates[0]
    if objective_mode == "energy":
        best_obj = _objective_energy(best_beta, a, b, c)
    else:
        best_obj = _objective_ratio(best_beta, a, b, c, A, B, C)

    for cand in candidates[1:]:
        if not math.isfinite(cand):
            continue
        beta = float(cand)
        if objective_mode == "energy":
            obj = _objective_energy(beta, a, b, c)
        else:
            obj = _objective_ratio(beta, a, b, c, A, B, C)
        if obj < best_obj:
            best_obj = obj
            best_beta = beta

    return float(best_beta)


class SAESFactorizedLinear(nn.Module):
    """
    Frozen factorized linear:
      y = (x @ B^T) @ A^T + bias
    where A:[out,r], B:[r,in].
    """

    def __init__(self, in_features: int, out_features: int, rank: int, dtype: torch.dtype):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        # Use zeros instead of empty to avoid catastrophic random uninitialized values
        # in case of partial/mismatched checkpoint loads downstream.
        self.register_buffer("A", torch.zeros((self.out_features, self.rank), dtype=dtype))
        self.register_buffer("B", torch.zeros((self.rank, self.in_features), dtype=dtype))
        self.register_buffer("bias", torch.zeros((0,), dtype=dtype))

    @torch.no_grad()
    def set_factors(self, A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        if A.shape != (self.out_features, self.rank):
            raise ValueError(f"A shape mismatch: expected {(self.out_features, self.rank)}, got {tuple(A.shape)}")
        if B.shape != (self.rank, self.in_features):
            raise ValueError(f"B shape mismatch: expected {(self.rank, self.in_features)}, got {tuple(B.shape)}")
        if not torch.isfinite(A).all():
            raise ValueError("A contains NaN/Inf")
        if not torch.isfinite(B).all():
            raise ValueError("B contains NaN/Inf")

        self.A.copy_(A)
        self.B.copy_(B)
        if bias is None:
            self.bias = torch.zeros((0,), device=self.A.device, dtype=self.A.dtype)
        else:
            bb = bias.detach().to(device=self.A.device, dtype=self.A.dtype)
            if bb.dim() != 1 or bb.numel() != self.out_features:
                raise ValueError(f"bias shape mismatch: expected ({self.out_features},), got {tuple(bb.shape)}")
            self.bias = bb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        y2d = self.forward_flat(x2d)
        return y2d.reshape(*x.shape[:-1], self.out_features)

    def forward_flat(self, x2d: torch.Tensor) -> torch.Tensor:
        in_dtype = x2d.dtype
        z = x2d.to(self.B.dtype) @ self.B.transpose(0, 1)
        y = z @ self.A.transpose(0, 1)
        if self.bias.numel() > 0:
            y = y + self.bias
        if y.dtype != in_dtype:
            try:
                if in_dtype.is_floating_point:
                    maxv = torch.finfo(in_dtype).max
                    y = torch.clamp(y, min=-maxv, max=maxv)
            except Exception:
                pass
            y = y.to(in_dtype)
        return y


def extract_saes_manifest(model: nn.Module, layer_start: int, layer_end: int) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for i in range(layer_start, layer_end):
        for sub in TARGET_SUBMODULES:
            name = f"model.layers.{i}.{sub}"
            mod = get_module_by_name(model, name)
            if isinstance(mod, SAESFactorizedLinear):
                items.append(
                    {
                        "name": name,
                        "in_features": int(mod.in_features),
                        "out_features": int(mod.out_features),
                        "rank": int(mod.rank),
                    }
                )
    return {"saes_version": 1, "items": items}


def save_saes_checkpoint(model: nn.Module, tokenizer: Any, out_dir: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "saes_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), os.path.join(out_dir, "saes_state.pt"))
    tokenizer.save_pretrained(out_dir)


def _pair_flatten_and_sample(
    x: torch.Tensor,
    x_fp: torch.Tensor,
    sample_tokens_per_batch: Optional[int],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    x2d = x.detach().reshape(-1, x.shape[-1])
    xf2d = x_fp.detach().reshape(-1, x_fp.shape[-1])
    n = min(int(x2d.shape[0]), int(xf2d.shape[0]))
    if n <= 0:
        return None
    x2d = x2d[:n]
    xf2d = xf2d[:n]

    if sample_tokens_per_batch is not None and n > int(sample_tokens_per_batch):
        idx = torch.randperm(n, device=x2d.device)[: int(sample_tokens_per_batch)]
        x2d = x2d.index_select(0, idx)
        xf2d = xf2d.index_select(0, idx)

    if not torch.isfinite(x2d).all() or not torch.isfinite(xf2d).all():
        m = torch.isfinite(x2d).all(dim=1) & torch.isfinite(xf2d).all(dim=1)
        if not bool(m.any()):
            return None
        x2d = x2d[m]
        xf2d = xf2d[m]
    return x2d, xf2d


@torch.no_grad()
def collect_layer_saes_stats(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataloader: DataLoader,
    *,
    layer_idx: int,
    target_submodules: List[str],
    student_device: torch.device,
    teacher_device: torch.device,
    stats_device: torch.device,
    max_batches: Optional[int],
    max_tokens_total: Optional[int],
    sample_tokens_per_batch: Optional[int],
    tqdm_enabled: bool,
) -> Dict[str, SAESStats]:
    """
    Collect SAES calibration stats for selected linears in one layer.
    Student path provides X, teacher path provides X_f.
    """
    if len(target_submodules) == 0:
        raise ValueError("target_submodules must be non-empty")

    stats: Dict[str, SAESStats] = {}
    student_cache: Dict[str, torch.Tensor] = {}
    teacher_cache: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(cache: Dict[str, torch.Tensor], key: str):
        def hook(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
            if len(inp) == 0:
                return
            cache[key] = inp[0].detach()

        return hook

    for sub in target_submodules:
        name = f"model.layers.{layer_idx}.{sub}"
        s_mod = get_module_by_name(student_model, name)
        t_mod = get_module_by_name(teacher_model, name)
        if not hasattr(s_mod, "in_features"):
            raise TypeError(f"{name} in student model has no in_features")
        if not hasattr(t_mod, "in_features"):
            raise TypeError(f"{name} in teacher model has no in_features")
        d_in = int(getattr(s_mod, "in_features"))
        stats[sub] = SAESStats(d_in=d_in, device=stats_device)
        handles.append(s_mod.register_forward_pre_hook(make_hook(student_cache, sub)))
        handles.append(t_mod.register_forward_pre_hook(make_hook(teacher_cache, sub)))

    student_model.eval()
    teacher_model.eval()
    it = tqdm(enumerate(dataloader, 1), desc=f"Collect layer {layer_idx}", leave=False) if tqdm_enabled else enumerate(dataloader, 1)
    for bi, batch in it:
        if max_batches is not None and bi > int(max_batches):
            break

        student_cache.clear()
        teacher_cache.clear()

        batch_student = {k: v.to(student_device) for k, v in batch.items()}
        if teacher_device == student_device:
            batch_teacher = batch_student
        else:
            batch_teacher = {k: v.to(teacher_device) for k, v in batch.items()}

        _ = student_model(**batch_student)
        _ = teacher_model(**batch_teacher)

        for sub in target_submodules:
            if sub not in student_cache:
                raise RuntimeError(f"Missing student activation for layer={layer_idx}, sub={sub}")
            if sub not in teacher_cache:
                raise RuntimeError(f"Missing teacher activation for layer={layer_idx}, sub={sub}")

            pair = _pair_flatten_and_sample(
                student_cache[sub],
                teacher_cache[sub],
                sample_tokens_per_batch=sample_tokens_per_batch,
            )
            if pair is None:
                continue
            x_cur, x_fp = pair
            stats[sub].update(x_cur, x_fp)

        ref_tokens = stats[target_submodules[0]].n_tokens
        if max_tokens_total is not None and ref_tokens >= int(max_tokens_total):
            break

    for h in handles:
        h.remove()

    return stats


@torch.no_grad()
def compress_one_layer(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataloader: DataLoader,
    *,
    layer_idx: int,
    compression_ratio: float,
    align_alpha: float,
    beta_fixed: Optional[float],
    beta_mode: str,
    beta_min: float,
    beta_max: float,
    beta_cap: float,
    beta_shrink: float,
    aces_objective: str,
    student_device: torch.device,
    teacher_device: torch.device,
    stats_device: torch.device,
    compute_device: torch.device,
    compute_dtype: torch.dtype,
    factor_dtype: torch.dtype,
    max_batches: Optional[int],
    max_tokens_total: Optional[int],
    sample_tokens_per_batch: Optional[int],
    whitening_damping: float,
    cholesky_max_tries: int,
    svd_lowrank: bool,
    svd_oversample: int,
    svd_niter: int,
    tqdm_enabled: bool,
) -> List[Dict[str, Any]]:
    module_records: List[Dict[str, Any]] = []
    beta_from_alpha = float(align_alpha) / (1.0 + float(align_alpha))
    if beta_fixed is not None:
        beta_from_alpha = float(beta_fixed)

    phase_submodule_groups = [PHASE1_SUBMODULES, PHASE2_SUBMODULES]
    for phase_submodules in phase_submodule_groups:
        stats_by_sub = collect_layer_saes_stats(
            student_model,
            teacher_model,
            dataloader,
            layer_idx=layer_idx,
            target_submodules=phase_submodules,
            student_device=student_device,
            teacher_device=teacher_device,
            stats_device=stats_device,
            max_batches=max_batches,
            max_tokens_total=max_tokens_total,
            sample_tokens_per_batch=sample_tokens_per_batch,
            tqdm_enabled=tqdm_enabled,
        )

        for sub in phase_submodules:
            name = f"model.layers.{layer_idx}.{sub}"
            mod = get_module_by_name(student_model, name)
            if not isinstance(mod, nn.Linear):
                raise TypeError(f"{name} is not nn.Linear before compression")
            W = mod.weight.detach()
            out_features, in_features = int(W.shape[0]), int(W.shape[1])
            rank = compute_base_rank(out_features, in_features, compression_ratio=compression_ratio)
            H, Delta, n_tokens = stats_by_sub[sub].get()

            if beta_mode == "aces":
                beta = select_beta_fixed_subspace_aces(
                    W,
                    H,
                    Delta,
                    rank,
                    beta_base=beta_from_alpha,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    beta_cap=beta_cap,
                    beta_shrink=beta_shrink,
                    objective=aces_objective,
                    damping=whitening_damping,
                    max_tries=cholesky_max_tries,
                    compute_device=compute_device,
                    compute_dtype=compute_dtype,
                    svd_lowrank=svd_lowrank,
                    oversample=svd_oversample,
                    n_iter=svd_niter,
                )
            else:
                beta = _apply_beta_guardrails(
                    beta_from_alpha,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    beta_cap=beta_cap,
                    beta_shrink=beta_shrink,
                )

            A, B, used_damping = saes_whitened_factorize(
                W=W,
                H=H,
                Delta=Delta,
                rank=rank,
                beta=beta,
                damping=whitening_damping,
                max_tries=cholesky_max_tries,
                compute_device=compute_device,
                compute_dtype=compute_dtype,
                svd_lowrank=svd_lowrank,
                oversample=svd_oversample,
                n_iter=svd_niter,
            )

            new_mod = SAESFactorizedLinear(
                in_features=in_features,
                out_features=out_features,
                rank=int(A.shape[1]),
                dtype=factor_dtype,
            ).to(student_device)
            new_mod.set_factors(
                A=A.to(device=student_device, dtype=factor_dtype),
                B=B.to(device=student_device, dtype=factor_dtype),
                bias=None if mod.bias is None else mod.bias.detach(),
            )
            set_module_by_name(student_model, name, new_mod)

            module_records.append(
                {
                    "layer": int(layer_idx),
                    "name": name,
                    "sub": sub,
                    "phase": int(1 if sub in PHASE1_SUBMODULES else 2),
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": int(A.shape[1]),
                    "n_tokens": int(n_tokens),
                    "beta": float(beta),
                    "used_damping": float(used_damping),
                    "svd_lowrank": bool(svd_lowrank),
                    "aces_objective": str(aces_objective),
                }
            )

            del H, Delta, W, A, B, new_mod
            if student_device.type == "cuda":
                torch.cuda.empty_cache()

    return module_records


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    ap.add_argument(
        "--tokenizer_model",
        type=str,
        default=None,
        help="Optional tokenizer model id/path. Defaults to --model_id.",
    )
    ap.add_argument("--hf_token", type=str, default=None, help="HF token for gated/private repos.")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--compression_ratio", type=float, default=0.4)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--calib_sequences", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dataset_name", type=str, default="wikitext")
    ap.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    ap.add_argument("--dataset_split", type=str, default="train")
    ap.add_argument(
        "--calib_mix",
        action="store_true",
        help="Enable mixed calibration set (LM + linguistic buckets) for SAES stats collection.",
    )
    ap.add_argument(
        "--calib_mix_bucket_props",
        type=str,
        default="LM:0.7,INST:0.3,MATH:0.0",
        help="Bucket ratios for mixed calibration. Example: LM:0.7,INST:0.3,MATH:0.0",
    )
    ap.add_argument(
        "--calib_mix_lm_datasets",
        type=str,
        default="wikitext2,ptb,c4_stream",
        help="Comma-separated LM datasets for mixed calibration.",
    )
    ap.add_argument(
        "--calib_mix_ling_datasets",
        type=str,
        default="cola,sst2",
        help="Comma-separated linguistic datasets for mixed calibration (mapped to INST bucket).",
    )
    ap.add_argument(
        "--calib_mix_c4_stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether mixed calibration should use streaming C4 for c4/c4_stream entries.",
    )
    ap.add_argument(
        "--calib_mix_per_bucket",
        action="store_true",
        help="Use per-bucket full budgets instead of a shared total budget in mixed calibration.",
    )
    ap.add_argument(
        "--calib_mix_dump_debug",
        action="store_true",
        help="Print per-dataset mixed calibration sample counts.",
    )

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--teacher_device", type=str, default="auto")
    ap.add_argument("--stats_device", type=str, default="cpu")
    ap.add_argument("--compute_device", type=str, default="auto")

    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--teacher_dtype", type=str, default="float16")
    ap.add_argument("--factor_dtype", type=str, default="float32")
    ap.add_argument("--compute_dtype", type=str, default="float32")

    ap.add_argument("--layer_start", type=int, default=0)
    ap.add_argument("--layer_end", type=int, default=None)

    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--max_tokens_total", type=int, default=262144)
    ap.add_argument(
        "--sample_tokens_per_batch",
        type=int,
        default=None,
        help="Token subsampling per batch for H/Delta statistics. None means use all tokens (paper-aligned).",
    )

    ap.add_argument("--whitening_damping", type=float, default=1e-5)
    ap.add_argument("--cholesky_max_tries", type=int, default=8)

    ap.add_argument("--beta_mode", type=str, choices=["fixed", "aces"], default="aces")
    ap.add_argument("--align_alpha", type=float, default=1.0)
    ap.add_argument("--beta_fixed", type=float, default=None)
    ap.add_argument("--beta_min", type=float, default=0.2)
    ap.add_argument("--beta_max", type=float, default=0.428571)
    ap.add_argument("--beta_cap", type=float, default=0.428571)
    ap.add_argument("--beta_shrink", type=float, default=1.0)
    ap.add_argument(
        "--aces_objective",
        type=str,
        choices=["ratio", "energy"],
        default="ratio",
        help="ACES objective. 'ratio' is paper-recommended; 'energy' is more conservative.",
    )

    ap.add_argument(
        "--svd_method",
        type=str,
        choices=["randomized", "full"],
        default="randomized",
        help="Truncated SVD backend for SAES factorization.",
    )
    ap.add_argument("--svd_oversample", type=int, default=32)
    ap.add_argument("--svd_niter", type=int, default=2)

    ap.add_argument(
        "--tqdm",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Progress bars. 'auto' disables tqdm in non-TTY logs.",
    )
    ap.add_argument("--timing_file", type=str, default="saes_timing.json")

    args = ap.parse_args()
    set_seed(args.seed)
    if args.align_alpha < 0:
        raise ValueError("--align_alpha must be >= 0")
    if args.beta_fixed is not None and not (0.0 <= float(args.beta_fixed) < 1.0):
        raise ValueError("--beta_fixed must be in [0, 1)")
    if args.beta_min < 0.0 or args.beta_min >= 1.0:
        raise ValueError("--beta_min must be in [0, 1)")
    if args.beta_max < args.beta_min or args.beta_max >= 1.0:
        raise ValueError("--beta_max must satisfy beta_min <= beta_max < 1")
    if args.beta_cap <= 0.0 or args.beta_cap >= 1.0:
        raise ValueError("--beta_cap must be in (0, 1)")
    if args.beta_cap < args.beta_min:
        raise ValueError("--beta_cap must be >= beta_min for a valid guarded interval")
    if args.beta_shrink <= 0.0 or args.beta_shrink > 1.0:
        raise ValueError("--beta_shrink must be in (0, 1]")

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

    student_device = parse_device(args.device)
    teacher_device = parse_device(args.teacher_device, fallback=student_device)
    stats_device = parse_device(args.stats_device)
    compute_device = parse_device(args.compute_device, fallback=student_device)
    model_dtype = parse_dtype(args.dtype)
    teacher_dtype = parse_dtype(args.teacher_dtype)
    factor_dtype = parse_dtype(args.factor_dtype)
    compute_dtype = parse_dtype(args.compute_dtype)

    t0 = time.perf_counter()
    tokenizer_hint = args.model_id if args.tokenizer_model is None else str(args.tokenizer_model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_hint,
            use_fast=True,
            trust_remote_code=True,
            token=args.hf_token,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_hint,
            use_fast=True,
            trust_remote_code=True,
            use_auth_token=args.hf_token,
        )
    except Exception:
        tokenizer = None
    tokenizer = _ensure_tokenizer(tokenizer, tokenizer_hint, hf_token=args.hf_token)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    timing["stages"].append({"name": "load_tokenizer", "sec": time.perf_counter() - t0})

    t0 = time.perf_counter()
    student_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=model_dtype).to(student_device)
    student_model.config.use_cache = False
    student_model.eval()
    timing["stages"].append({"name": "load_student_model", "sec": time.perf_counter() - t0})

    t0 = time.perf_counter()
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=teacher_dtype).to(teacher_device)
    teacher_model.config.use_cache = False
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    timing["stages"].append({"name": "load_teacher_model", "sec": time.perf_counter() - t0})

    t0 = time.perf_counter()
    if args.calib_mix:
        dataloader = build_mixed_calibration_dataloader(
            tokenizer,
            num_sequences=args.calib_sequences,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
            bucket_props=args.calib_mix_bucket_props,
            bucket_lm_datasets=args.calib_mix_lm_datasets,
            bucket_ling_datasets=args.calib_mix_ling_datasets,
            c4_stream=bool(args.calib_mix_c4_stream),
            per_bucket=bool(args.calib_mix_per_bucket),
            dump_bucket_debug=bool(args.calib_mix_dump_debug),
        )
        timing["stages"].append(
            {
                "name": "build_calib_dataloader_mix",
                "sec": time.perf_counter() - t0,
                "bucket_props": str(args.calib_mix_bucket_props),
                "lm_datasets": str(args.calib_mix_lm_datasets),
                "ling_datasets": str(args.calib_mix_ling_datasets),
            }
        )
    else:
        dataloader = build_calibration_dataloader(
            tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.dataset_split,
            num_sequences=args.calib_sequences,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        timing["stages"].append(
            {
                "name": "build_calib_dataloader_single",
                "sec": time.perf_counter() - t0,
                "dataset_name": str(args.dataset_name),
                "dataset_config": str(args.dataset_config),
                "dataset_split": str(args.dataset_split),
            }
        )

    layers = student_model.model.layers
    layer_end = args.layer_end if args.layer_end is not None else len(layers)
    layer_end = min(layer_end, len(layers))
    if args.layer_start < 0 or args.layer_start >= layer_end:
        raise ValueError(f"Invalid layer range: start={args.layer_start}, end={layer_end}")

    svd_lowrank = bool(args.svd_method == "randomized")
    timing_path: Optional[str] = None
    try:
        for layer_idx in range(args.layer_start, layer_end):
            layer_t0 = time.perf_counter()
            module_records = compress_one_layer(
                student_model,
                teacher_model,
                dataloader,
                layer_idx=layer_idx,
                compression_ratio=args.compression_ratio,
                align_alpha=args.align_alpha,
                beta_fixed=args.beta_fixed,
                beta_mode=args.beta_mode,
                beta_min=args.beta_min,
                beta_max=args.beta_max,
                beta_cap=args.beta_cap,
                beta_shrink=args.beta_shrink,
                aces_objective=args.aces_objective,
                student_device=student_device,
                teacher_device=teacher_device,
                stats_device=stats_device,
                compute_device=compute_device,
                compute_dtype=compute_dtype,
                factor_dtype=factor_dtype,
                max_batches=args.max_batches,
                max_tokens_total=args.max_tokens_total,
                sample_tokens_per_batch=args.sample_tokens_per_batch,
                whitening_damping=args.whitening_damping,
                cholesky_max_tries=args.cholesky_max_tries,
                svd_lowrank=svd_lowrank,
                svd_oversample=args.svd_oversample,
                svd_niter=args.svd_niter,
                tqdm_enabled=tqdm_enabled,
            )
            layer_sec = time.perf_counter() - layer_t0
            timing["layers"].append({"layer": int(layer_idx), "sec": float(layer_sec), "modules": len(module_records)})
            timing["modules"].extend(module_records)
            if not tqdm_enabled:
                _log(f"[Layer] {layer_idx}: {layer_sec:.2f}s", tqdm_enabled=False)

            if student_device.type == "cuda":
                torch.cuda.empty_cache()

        t0 = time.perf_counter()
        manifest = extract_saes_manifest(student_model, args.layer_start, layer_end)
        save_saes_checkpoint(student_model, tokenizer, args.output_dir, manifest)
        timing["stages"].append({"name": "save_checkpoint", "sec": time.perf_counter() - t0})
        _log(f"[OK] Saved SAES-SVD checkpoint to: {args.output_dir}", tqdm_enabled=tqdm_enabled)
    finally:
        timing["ended_at"] = _now_iso()
        timing["total_sec"] = time.perf_counter() - run_start
        try:
            timing_path = _timing_write(args.output_dir, args.timing_file, timing)
        except Exception as e:
            _log(f"[Warn] Failed to write timing file: {e}", tqdm_enabled=tqdm_enabled)

    if timing_path:
        _log(f"[Time] total={float(timing['total_sec']):.2f}s timing_json={timing_path}", tqdm_enabled=tqdm_enabled)


if __name__ == "__main__":
    main()
