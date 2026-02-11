# profile_svdllm_v1.py — SVD-LLM v1 for ModernBERT (Whitening-SVD)
# -------------------------------------------------------------------------------
# Adapted from RoBERTaWhiten for ModernBERT
# Key changes:
#   - AutoModelForSequenceClassification (with trust_remote_code=True)
#   - model.model.layers (not model.bert.encoder.layer)
#   - Fused Wqkv [3*dm, dm] (verified: split Q[:768], K[768:1536], V[1536:])
#   - Fused Wi [2*d_ff, dm] (verified: split gate[:1152], input[1152:])
#   - Pre-norm architecture (LayerNorm before attention/FFN, not after)
#   - GeGLU activation (GELU(gate) * input instead of GELU(x))
#   - No bias in Wqkv and Wi (verified: bias=None)
#   - RoPE positional encoding (handled by model's rotary_emb, not in weights)
# -------------------------------------------------------------------------------

import os
import sys
import time
import itertools
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from evaluate import load as load_metric

from flash_attn_triton import flash_attn_triton

# ─── locate repo & model ─────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

task_name = "sst2"
MODEL_DIR = "mrm8488/ModernBERT-base-ft-sst2"  # Fine-tuned on SST-2

# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------
def _safe_cholesky(C: torch.Tensor, max_tries: int = 5, base_eps: float = 1e-6):
    """
    Robust Cholesky: if C is near-singular (few calibration samples), add jitter.
    Returns lower-triangular S with C + eps*I = S S^T.
    """
    D = C.shape[-1]
    eps = base_eps * float(C.diag().mean().item() + 1.0)
    I = torch.eye(D, dtype=C.dtype, device=C.device)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(C + eps * I)
        except RuntimeError:
            eps *= 10.0
    # last resort: add bigger ridge
    return torch.linalg.cholesky(C + (1e-2 * float(C.diag().mean().item() + 1.0)) * I)

def rank_from_ratio(m: int, n: int, ratio: float, min_rank: int = 1) -> int:
    r = int(m * n * ratio / (m + n))
    r = max(min_rank, min(r, min(m, n)))
    return r

def _data_aware_low_rank(W_in_out: torch.Tensor, rank: int, cov_in: torch.Tensor):
    """
    SVD-LLM v1 whitening-SVD factorization.

    We keep your convention: W_in_out is [d_in, d_out] (often you pass .t()).
    cov_in is input covariance C = E[x x^T] in the SAME input space (d_in x d_in).

    Steps (v1):
      L = chol(C)  (C = L L^T)
      W_scale = L^T W
      SVD(W_scale) = U S V^T (truncate)
      Unwhiten:  W ≈ (L^{-T} U_k sqrtS) (sqrtS V_k^T)
    Return U:[d_in,k], V:[k,d_out] so that W ≈ U @ V.
    """
    d_in, d_out = W_in_out.shape
    if rank <= 0:
        raise ValueError("rank must be positive")

    Wf = W_in_out.float()
    Cf = cov_in.float()

    # C = L L^T
    L = _safe_cholesky(Cf)  # lower

    # whiten in input space
    W_scale = L.t().contiguous() @ Wf  # [d_in, d_out]

    # SVD and truncate
    U, s, Vh = torch.linalg.svd(W_scale, full_matrices=False)  # U:[d_in,*], Vh:[*,d_out]
    k = min(rank, s.numel())
    U_k = U[:, :k]          # [d_in,k]
    s_k = s[:k]             # [k]
    Vh_k = Vh[:k, :]        # [k,d_out]

    # unwhiten: solve L^T X = U_k  => X = L^{-T} U_k
    X = torch.linalg.solve_triangular(L.t(), U_k, upper=True)  # [d_in,k]

    sqrt_s = torch.sqrt(torch.clamp(s_k, min=0))
    U_data = X * sqrt_s.unsqueeze(0)          # [d_in,k]
    V_data = sqrt_s.unsqueeze(1) * Vh_k       # [k,d_out]

    return U_data.to(W_in_out.dtype), V_data.to(W_in_out.dtype)


def _data_aware_per_head(Wt_dm_dh: torch.Tensor, rank: int, cov_in: torch.Tensor, num_heads: int):
    """
    DRONE-style factorization per-head for attention Q/K/V.
    Input:
      - Wt_dm_dh: weight^T with shape [d_model, d_head * H] but we reshape per-head
                  (we pass per-head as [d_model, dh] same as the original code did)
      - rank: target rank per head
      - cov_in: input covariance for this linear's input space (d_model x d_model)
      - num_heads: H
    Returns:
      stacked U:[H, d_model, rank], V:[H, rank, dh]
    """
    d_model = Wt_dm_dh.shape[0]
    dh = Wt_dm_dh.shape[1] // num_heads
    Wt3 = Wt_dm_dh.view(d_model, num_heads, dh)

    Us, Vs = [], []
    for h in range(num_heads):
        Wh = Wt3[:, h, :]  # [d_model, dh]
        Uh, Vh = _data_aware_low_rank(Wh, rank, cov_in)
        Us.append(Uh)                 # [d_model, rank]
        Vs.append(Vh)                 # [rank, dh]
    U = torch.stack(Us, dim=0)        # [H, d_model, rank]
    V = torch.stack(Vs, dim=0)        # [H, rank, dh]
    return U, V

# -----------------------------------------------------------------------------
# 1) LayerShim
# -----------------------------------------------------------------------------
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        raw_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 4:
            raw_mask = (attention_mask[:, 0, 0, :] == 0)
        return (self.block(hidden_states, raw_mask),)

# -----------------------------------------------------------------------------
# 2) Data-aware SVDBlock for ModernBERT (DRONE)
# -----------------------------------------------------------------------------
class ModernBertSVDBlock(nn.Module):
    def __init__(
        self,
        hf_layer,
        rank_attn: int,
        rank_ff: int,
        cov_attn_in: torch.Tensor,
        cov_attn_out: torch.Tensor,
        cov_ffn_in: torch.Tensor,
        cov_ffn_out: torch.Tensor,
        rank_wo: int = 768,
        num_heads: int = 12,
    ):
        super().__init__()
        d_model = hf_layer.attn.Wqkv.in_features  # 768
        H = num_heads
        dh = d_model // H
        d_ff = hf_layer.mlp.Wi.out_features // 2  # Wi is [2*d_ff, dm], so out_features=2*d_ff

        # 1) Extract Q/K/V from fused Wqkv (VERIFIED: [3*dm, dm] = [2304, 768])
        Wqkv_full = hf_layer.attn.Wqkv.weight.data  # [3*dm, dm] = [2304, 768]
        # VERIFIED split positions: Q[:768], K[768:1536], V[1536:]
        Wq = Wqkv_full[:d_model, :]      # [dm, dm]
        Wk = Wqkv_full[d_model:2*d_model, :]  # [dm, dm]
        Wv = Wqkv_full[2*d_model:, :]    # [dm, dm]

        # No bias in Wqkv (verified: bias=None), so no bq/bk/bv

        # 2) DRONE factorization per head on Q/K/V using cov_attn_in (dm x dm)
        WqT = Wq.t()  # [dm, dm]
        WkT = Wk.t()
        WvT = Wv.t()
        Uq, Vq = _data_aware_per_head(WqT, rank_attn, cov_attn_in, H)
        Uk, Vk = _data_aware_per_head(WkT, rank_attn, cov_attn_in, H)
        Uv, Vv = _data_aware_per_head(WvT, rank_attn, cov_attn_in, H)

        # 3) Extract gate and input from fused Wi (VERIFIED: [2*d_ff, dm] = [2304, 768])
        Wi_full = hf_layer.mlp.Wi.weight.data  # [2*d_ff, dm]
        # VERIFIED split: gate[:1152], input[1152:]
        W_gate = Wi_full[:d_ff, :]    # [d_ff, dm]
        W_input = Wi_full[d_ff:, :]   # [d_ff, dm]
        # No bias in Wi (verified: bias=None)

        # FFN factorization (data-aware)
        # For gate and input, we use cov_ffn_in (dm x dm)
        U_gate, V_gate = _data_aware_low_rank(W_gate.t(), rank_ff, cov_ffn_in)
        U_input, V_input = _data_aware_low_rank(W_input.t(), rank_ff, cov_ffn_in)

        # For FFN down projection Wo: [dm, d_ff] (VERIFIED)
        WoT = hf_layer.mlp.Wo.weight.data.t()  # [d_ff, dm]
        bo_ffn = hf_layer.mlp.Wo.bias.data if hf_layer.mlp.Wo.bias is not None else torch.zeros(d_model, device=WoT.device)
        U2, V2 = _data_aware_low_rank(WoT, rank_ff, cov_ffn_out)  # input cov: d_ff x d_ff

        # 4) Attention output projection Wo (data-aware)
        Wo_attn_full = hf_layer.attn.Wo.weight.data  # [dm, dm] (out,in)
        bo_attn = hf_layer.attn.Wo.bias.data if hf_layer.attn.Wo.bias is not None else torch.zeros(d_model, device=Wo_attn_full.device)
        Uo, Vo = _data_aware_low_rank(Wo_attn_full.t(), rank_wo, cov_attn_out)  # input cov: dm x dm

        # Store as Parameters (NO unsqueeze for Pq/Pk/Pv - ModernBERT uses [H, dm, R] layout)
        self.Pq = nn.Parameter(Uq)  # [H, dm, R]
        self.Vq = nn.Parameter(Vq)  # [H, R, dh]
        self.Pk = nn.Parameter(Uk)
        self.Vk = nn.Parameter(Vk)
        self.Pv = nn.Parameter(Uv)
        self.Vv = nn.Parameter(Vv)

        self.Uo = nn.Parameter(Uo)
        self.Vo = nn.Parameter(Vo)
        self.bo_attn = nn.Parameter(bo_attn)

        self.U_gate = nn.Parameter(U_gate)
        self.V_gate = nn.Parameter(V_gate)
        self.U_input = nn.Parameter(U_input)
        self.V_input = nn.Parameter(V_input)
        self.U2 = nn.Parameter(U2)
        self.V2 = nn.Parameter(V2)
        self.bo_ffn = nn.Parameter(bo_ffn)

        # LayerNorm (pre-norm)
        self.ln1 = hf_layer.attn_norm
        self.ln2 = hf_layer.mlp_norm

        # Store RoPE for later use
        self.rotary_emb = hf_layer.attn.rotary_emb if hasattr(hf_layer.attn, 'rotary_emb') else None

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        # x: [B, H, M, dh]
        # Split last dim in half, negate second half, swap
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, q, k, position_ids):
        """
        Apply RoPE to Q and K using ModernBERT's native rotary_emb.

        Args:
            q: [B, H, M, dh] query tensor
            k: [B, H, M, dh] key tensor
            position_ids: [B, M] position indices

        Returns:
            q_rotated, k_rotated: [B, H, M, dh]
        """
        if self.rotary_emb is None:
            return q, k

        # Get cos and sin from ModernBERT's RoPE
        # rotary_emb expects [B, H, M, dh] and returns cos, sin of shape [B, M, dh]
        cos, sin = self.rotary_emb(q, position_ids)

        # Reshape cos/sin to broadcast: [B, 1, M, dh]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Apply RoPE: q_rotated = (q * cos) + (rotate_half(q) * sin)
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H = self.Pq.shape[0]
        dh = dm // H

        # Pre-norm for attention
        x_normed = self.ln1(x)

        # Project into low-rank Q/K/V
        def project(x, P, V):
            # x:[B,M,dm], P:[H,dm,R], V:[H,R,dh]
            # NO indexing P[0] - ModernBERT uses [H, dm, R] directly
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V)

        Q = project(x_normed, self.Pq, self.Vq).contiguous()
        K = project(x_normed, self.Pk, self.Vk).contiguous()
        V = project(x_normed, self.Pv, self.Vv).contiguous()

        # Construct position_ids: [B, M]
        position_ids = torch.arange(M, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        # Apply RoPE to Q and K using ModernBERT's native rotary_emb
        Q, K = self.apply_rotary(Q, K, position_ids)

        # Attention mask
        if mask is not None:
            mask4d = mask.view(B, 1, 1, M).expand(B, H, 1, M).to(torch.bool)
        else:
            mask4d = torch.ones(B, H, 1, M, device=x.device, dtype=torch.bool)

        # Flash-attn returns [B, H, M, dh] float32
        attn = flash_attn_triton(Q, K, V, mask4d, BLOCK_M=32)

        del Q, K, V
        torch.cuda.empty_cache()

        # Back to [B,M,dm]
        attn = attn.transpose(1, 2).reshape(B, M, dm)
        # Attention output projection (low-rank)
        attn_out = (attn @ self.Uo) @ self.Vo + self.bo_attn
        # Residual connection
        x1 = x + attn_out

        # Pre-norm for FFN
        x1_normed = self.ln2(x1)

        # GeGLU FFN (low-rank)
        # gate = x @ U_gate @ V_gate
        # input = x @ U_input @ V_input
        # hidden = GELU(gate) * input
        gate = (x1_normed @ self.U_gate) @ self.V_gate
        inp = (x1_normed @ self.U_input) @ self.V_input
        mlp_hidden = F.gelu(gate) * inp

        # Down projection (low-rank)
        mlp_out = (mlp_hidden @ self.U2) @ self.V2 + self.bo_ffn
        # Residual connection
        out = x1 + mlp_out
        return out

# -----------------------------------------------------------------------------
# 3) Calibration: collect per-layer input covariances (one-shot)
# -----------------------------------------------------------------------------
@torch.no_grad()
def calibrate_covariances(model,
                          loader: DataLoader,
                          device: str,
                          max_batches: int = 4) -> Dict[str, List[torch.Tensor]]:
    """
    Collects (online) covariance estimates for ModernBERT layers:
      - cov_attn_in: input to attention (after input_layernorm)  -> dm x dm
      - cov_attn_out: output of attention (before residual, input to Wo) -> dm x dm
      - cov_ffn_in: input to FFN (after post_attn_layernorm) -> dm x dm
      - cov_ffn_out: output of GeGLU (before down projection Wo) -> d_ff x d_ff
    Returns dict with lists over layers.
    """
    model.eval()
    enc = model.model.layers  # ModernBERT encoder access
    num_layers = len(enc)
    dm = model.config.hidden_size
    d_ff = model.config.intermediate_size

    # Allocate accumulators on CUDA for speed; finalize to CPU later
    cov_attn_in  = [torch.zeros(dm, dm,  dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_attn_in    = [0 for _ in range(num_layers)]

    cov_attn_out = [torch.zeros(dm, dm,  dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_attn_out   = [0 for _ in range(num_layers)]

    cov_ffn_in   = [torch.zeros(dm, dm,  dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_ffn_in     = [0 for _ in range(num_layers)]

    cov_ffn_out  = [torch.zeros(d_ff, d_ff, dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_ffn_out    = [0 for _ in range(num_layers)]

    handles = []

    def _upd(cov_mat, n_store, idx, x):
        # x:[B,M,D] -> [N,D]
        if x is None:
            return
        x = x.detach()
        BMD = x.shape[0] * x.shape[1]
        X2d = x.reshape(BMD, x.shape[-1]).to(device=device, dtype=torch.float32)
        cov_mat[idx] += X2d.t() @ X2d
        n_store[idx] += BMD

    # Register hooks for ModernBERT (pre-norm architecture)
    for i, layer in enumerate(enc):
        # 1) cov_attn_in: hook AFTER attn_norm (pre-norm)
        def ln1_post_hook(mod, inp, out, idx=i):
            _upd(cov_attn_in, n_attn_in, idx, out)
        handles.append(layer.attn_norm.register_forward_hook(ln1_post_hook))

        # 2) cov_attn_out: hook BEFORE attention output projection Wo
        def attn_wo_pre_hook(mod, inp, idx=i):
            _upd(cov_attn_out, n_attn_out, idx, inp[0])
        handles.append(layer.attn.Wo.register_forward_pre_hook(attn_wo_pre_hook))

        # 3) cov_ffn_in: hook AFTER mlp_norm (pre-norm)
        def ln2_post_hook(mod, inp, out, idx=i):
            _upd(cov_ffn_in, n_ffn_in, idx, out)
        handles.append(layer.mlp_norm.register_forward_hook(ln2_post_hook))

        # 4) cov_ffn_out: hook BEFORE FFN down projection Wo (after GeGLU activation)
        def ffn_wo_pre_hook(mod, inp, idx=i):
            _upd(cov_ffn_out, n_ffn_out, idx, inp[0])
        handles.append(layer.mlp.Wo.register_forward_pre_hook(ffn_wo_pre_hook))

    # Run a few batches to collect stats
    seen = 0
    for batch in loader:
        if seen >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        seen += 1

    # Remove hooks
    for h in handles:
        h.remove()

    # Finalize covariance: normalize and add small ridge for stability; move to CPU
    def _finalize(cov_list, n_list):
        out = []
        for C, n in zip(cov_list, n_list):
            if n == 0:
                # fallback to identity if no samples (shouldn't happen)
                D = C.shape[0]
                Cn = torch.eye(D, dtype=torch.float32, device=C.device)
            else:
                Cn = C / float(n)
                # light ridge
                ridge = 1e-6 * float(Cn.diag().mean().item() + 1.0)
                Cn = Cn + ridge * torch.eye(Cn.shape[0], dtype=Cn.dtype, device=Cn.device)
            out.append(Cn.cpu())
        return out

    return {
        "cov_attn_in":  _finalize(cov_attn_in,  n_attn_in),
        "cov_attn_out": _finalize(cov_attn_out, n_attn_out),
        "cov_ffn_in":   _finalize(cov_ffn_in,   n_ffn_in),
        "cov_ffn_out":  _finalize(cov_ffn_out,  n_ffn_out),
    }

# -----------------------------------------------------------------------------
# 4) Benchmark helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def acc_peak_time(mdl, loader, device, task_name: str):
    mdl.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if task_name == "stsb":
        metric = load_metric("pearsonr")
        metric_key = "pearsonr"
    else:
        metric = load_metric("accuracy")
        metric_key = "accuracy"
    total, steps = 0.0, 0
    start = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = mdl(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"]).logits
        if task_name == "stsb":
            preds = logits.squeeze(-1)
        else:
            preds = torch.argmax(logits, -1)
        total += metric.compute(predictions=preds.cpu(), references=batch["labels"].cpu())[metric_key]
        steps += 1
    torch.cuda.synchronize()
    ms_per_batch = (time.perf_counter() - start) * 1000.0 / max(steps, 1)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return total / max(steps, 1), peak, ms_per_batch

# -----------------------------------------------------------------------------
# 5) Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    RATIO = 0.5

    BATCH_SIZE = 32
    SEQ_LEN    = 128 * 2
    device     = "cuda"


    # ─── GLUE load & tokenize ────────────────────────────────────────────────
    if task_name == "mnli":
        val_split = "validation_matched"
    else:
        val_split = "validation"
    raw = load_dataset("glue", task_name, split=val_split)
    tokz = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    single_sent_tasks = {"cola", "sst2"}
    pair_sent_tasks   = {"qqp", "mnli", "qnli", "stsb", "rte", "mrpc"}
    field_map = {
        "qqp":  ("question1", "question2"),
        "mnli": ("premise",   "hypothesis"),
        "qnli": ("question",  "sentence"),
        "stsb": ("sentence1", "sentence2"),
        "rte":  ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
    }

    def tokenize_fn(batch):
        if task_name in single_sent_tasks:
            return tokz(batch["sentence"], padding="max_length", truncation=True, max_length=SEQ_LEN)
        else:
            f1, f2 = field_map[task_name]
            return tokz(batch[f1], batch[f2], padding="max_length", truncation=True, max_length=SEQ_LEN)

    remove_cols = [c for c in raw.column_names if c != "label"]
    ds = raw.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    ds.set_format("torch")
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels":         torch.tensor([x["label"]         for x in b]),
        },
    )


    # ─── Build & move model ──────────────────────────────────────────────────
    if task_name == "mnli":
        num_labels, problem_type = 3, None
    elif task_name == "stsb":
        num_labels, problem_type = 1, "regression"
    else:
        num_labels, problem_type = 2, None

    cfg = AutoConfig.from_pretrained(MODEL_DIR, num_labels=num_labels, problem_type=problem_type, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=cfg, trust_remote_code=True)
    model = model.to(device).eval()

    # ---- SVD-LLM v1 official-style rank by ratio ----
    RATIO = 0.5  # try 0.5 / 0.3 / 0.2 later

    dm  = cfg.hidden_size              # 768
    dff = cfg.intermediate_size        # 1152 (ModernBERT-base)
    H   = cfg.num_attention_heads      # 12
    dh  = dm // H
    RANK_ATTN  = rank_from_ratio(dm, dh, RATIO)  # per-head rank for Q/K/V
    RANK_FF    = rank_from_ratio(dm, dff, RATIO)  # rank for FFN Wi (gate/input) and Wo
    RANK_WO    = rank_from_ratio(dm, dm, RATIO)   # rank for attention output projection Wo
    print(f"BATCH_SIZE: {BATCH_SIZE}  RANK_ATTN: {RANK_ATTN}  RANK_FF: {RANK_FF}  RANK_WO: {RANK_WO}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ─── Calibration pass: collect covariances ───────────────────────────────
    print("Calibrating input covariances (Whitening for ModernBERT)…")
    covs = calibrate_covariances(model, loader, device, max_batches=4)

    # ─── Replace each encoder layer with data-aware low-rank block ───────────
    for i, layer in enumerate(model.model.layers):  # ModernBERT layer access
        blk = ModernBertSVDBlock(
            hf_layer=layer,
            rank_attn=RANK_ATTN,
            rank_ff=RANK_FF,
            cov_attn_in=covs["cov_attn_in"][i].to(device),
            cov_attn_out=covs["cov_attn_out"][i].to(device),
            cov_ffn_in=covs["cov_ffn_in"][i].to(device),
            cov_ffn_out=covs["cov_ffn_out"][i].to(device),
            rank_wo=RANK_WO,
            num_heads=H,
        )
        model.model.layers[i] = LayerShim(blk).to(device).eval().float()  # ModernBERT layer replacement

    # ─── Memory accounting (persistent params only) ──────────────────────────
    def summarize_dense_vs_lowrank(model):
        dense_bytes, lowrank_bytes = 0, 0
        for name, p in model.named_parameters():
            size = p.numel() * p.element_size()
            if ".block." in name or (
                name.startswith("model.layers")  # ModernBERT namespace
                and any(part in name for part in ("Pq","Vq","Pk","Vk","Pv","Vv","U_gate","V_gate","U_input","V_input","U2","V2","Uo","Vo"))
            ):
                lowrank_bytes += size
            else:
                dense_bytes += size
        print(f"{'Type':<12}{'MiB':>8}")
        print("----------------------")
        print(f"{'Dense':<12}{dense_bytes/1024**2:8.1f}")
        print(f"{'Low-rank':<12}{lowrank_bytes/1024**2:8.1f}")
        print("----------------------")
        print(f"{'TOTAL':<12}{(dense_bytes+lowrank_bytes)/1024**2:8.1f}")
        return (dense_bytes+lowrank_bytes)

    baseline_bytes = summarize_dense_vs_lowrank(model)
    with_act = torch.cuda.max_memory_allocated() / 1024**2
    print(f"low-rank model storage with GPU redundancy: {with_act:.1f} MiB")

    print(f"Persistent low-rank model storage (ModernBERT Whitening): {baseline_bytes/1024**2:6.1f} MiB")

    # ─── Evaluate ────────────────────────────────────────────────────────────
    metric_name = "pearson" if task_name == "stsb" else "acc"
    acc, peak_lr, t = acc_peak_time(model, loader, device, task_name)
    print(f"ModernBERT Whitening v1 | {metric_name}={acc:.4f} | peak ={peak_lr:6.1f} MiB | {t:6.1f} ms/b")
