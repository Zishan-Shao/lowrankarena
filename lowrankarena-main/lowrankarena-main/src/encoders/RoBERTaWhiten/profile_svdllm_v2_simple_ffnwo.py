# profile_svdllm_v2_simple_ffnwo.py — SVD-LLM v2 for RoBERTa (Whitening-SVD + Local Update)
# -------------------------------------------------------------------------------
# Adapted from BERTWhiting for RoBERTa
# Key changes: BertForSequenceClassification -> RobertaForSequenceClassification
#              model.bert.encoder -> model.roberta.encoder
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
from transformers import RobertaForSequenceClassification, AutoTokenizer, AutoConfig
from evaluate import load as load_metric

from flash_attn_triton import flash_attn_triton

# ─── locate repo & model ─────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

task_name = "sst2"
MODEL_DIR = "textattack/roberta-base-SST-2"  # RoBERTa model for SST-2

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
# 2) Data-aware SVDBlock (DRONE)
# -----------------------------------------------------------------------------
class SVDBlock(nn.Module):
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
    ):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features

        # 1) grab weights (transpose to [d_in, d_out] as in original code)
        WqT = hf_layer.attention.self.query.weight.data.t()   # [dm, dm] but we do per-head slicing
        WkT = hf_layer.attention.self.key.weight.data.t()
        WvT = hf_layer.attention.self.value.weight.data.t()
        bq  = hf_layer.attention.self.query.bias.data.view(1, H, 1, dh)
        bk  = hf_layer.attention.self.key.bias.data.view(1, H, 1, dh)
        bv  = hf_layer.attention.self.value.bias.data.view(1, H, 1, dh)

        # 2) DRONE factorization per head on Q/K/V using cov_attn_in (dm x dm)
        Uq, Vq = _data_aware_per_head(WqT, rank_attn, cov_attn_in, H)
        Uk, Vk = _data_aware_per_head(WkT, rank_attn, cov_attn_in, H)
        Uv, Vv = _data_aware_per_head(WvT, rank_attn, cov_attn_in, H)

        # 3) FFN factorization (data-aware)
        Wi   = hf_layer.intermediate.dense.weight.data.t()     # [dm, d_ff]
        bi   = hf_layer.intermediate.dense.bias.data
        WoT  = hf_layer.output.dense.weight.data.t()           # [d_ff, dm]
        bo2  = hf_layer.output.dense.bias.data

        U1, V1 = _data_aware_low_rank(Wi,  rank_ff, cov_ffn_in)   # input cov: dm x dm
        U2, V2 = _data_aware_low_rank(WoT, rank_ff, cov_ffn_out)  # input cov: d_ff x d_ff

        # 4) Attention output projection W_o (data-aware)
        Wo_full = hf_layer.attention.output.dense.weight.data    # [dm, dm] (out,in)
        bo_attn = hf_layer.attention.output.dense.bias.data
        # We followed the original code convention to pass .t() so shape is [dm, dm]
        Uo, Vo = _data_aware_low_rank(Wo_full.t(), rank_wo, cov_attn_out)  # input cov: dm x dm

        # stash everything as Parameters
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)

        self.U1, self.V1, self.b1 = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2 = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        self.ln1, self.ln2 = hf_layer.attention.output.LayerNorm, hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        _, H, _, R = self.Pq.shape
        dh = dm // H

        # project into low-rank Q/K/V
        def project(x, P, V, b):
            # x:[B,M,dm], P:[H,dm,R], V:[H,R,dh], b:[1,H,1,dh]
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b

        Q = project(x, self.Pq[0], self.Vq[0], self.bq).contiguous()
        K = project(x, self.Pk[0], self.Vk[0], self.bk).contiguous()
        V = project(x, self.Pv[0], self.Vv[0], self.bv).contiguous()

        # Attention mask
        if mask is not None:
            mask4d = mask.view(B, 1, 1, M).expand(B, H, 1, M).to(torch.bool)
        else:
            mask4d = torch.ones(B, H, 1, M, device=x.device, dtype=torch.bool)

        # Flash-attn returns [B, H, M, dh] float32
        attn = flash_attn_triton(Q, K, V, mask4d, BLOCK_M=32)

        del Q, K, V
        torch.cuda.empty_cache()

        # back to [B,M,dm]
        attn = attn.transpose(1, 2).reshape(B, M, dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # FFN: (dm -> d_ff -> dm)
        mid  = x1 @ self.U1
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y    = (midA @ self.U2) @ self.V2 + self.b2
        out  = self.ln2(x1 + y)
        return out

# -----------------------------------------------------------------------------
# 3) Calibration: collect per-layer input covariances (one-shot)
# -----------------------------------------------------------------------------
@torch.no_grad()
def calibrate_covariances(model: RobertaForSequenceClassification,
                          loader: DataLoader,
                          device: str,
                          max_batches: int = 4) -> Dict[str, List[torch.Tensor]]:
    """
    Collects (online) covariance estimates for inputs of:
      - attention.self.query (shared for Q/K/V)  -> dm x dm
      - attention.output.dense                   -> dm x dm
      - intermediate.dense                       -> dm x dm
      - output.dense (FFN out, post-GELU input)  -> d_ff x d_ff
    Returns dict with lists over layers.
    """
    model.eval()
    enc = model.roberta.encoder  # RoBERTa encoder access
    num_layers = len(enc.layer)
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

    # Register hooks
    for i, layer in enumerate(enc.layer):
        # Inputs to Q/K/V (they share the same input): hook query pre-forward
        def q_pre_hook(mod, inp, idx=i):
            _upd(cov_attn_in, n_attn_in, idx, inp[0])
        handles.append(layer.attention.self.query.register_forward_pre_hook(q_pre_hook))

        # Inputs to attention.output.dense (post attention, before add&norm)
        def attn_out_pre_hook(mod, inp, idx=i):
            _upd(cov_attn_out, n_attn_out, idx, inp[0])
        handles.append(layer.attention.output.dense.register_forward_pre_hook(attn_out_pre_hook))

        # Inputs to intermediate.dense (after LN1)
        def ffn_in_pre_hook(mod, inp, idx=i):
            _upd(cov_ffn_in, n_ffn_in, idx, inp[0])
        handles.append(layer.intermediate.dense.register_forward_pre_hook(ffn_in_pre_hook))

        # Inputs to FFN output.dense (post-GELU)
        def ffn_out_pre_hook(mod, inp, idx=i):
            _upd(cov_ffn_out, n_ffn_out, idx, inp[0])
        handles.append(layer.output.dense.register_forward_pre_hook(ffn_out_pre_hook))

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
# 4.6) SVD-LLM v2 (paper-style): simple local update (FFN-V + WO-V, online)
# -----------------------------------------------------------------------------
@torch.no_grad()
def svdllm_v2_simple_local_update_ffn_wo_V(
    student,
    teacher,
    loader,
    device: str,
    max_batches: int = 4,          # ✅ Optimal: tested 4/8/16, best at 4
    max_rows_per_hook: int = 4096,  # ✅ Optimal: 4096 vs 8192, similar acc but faster
    ridge: float = 1e-4,           # ✅ Optimal: 1e-4 better than 3e-4
):
    """
    Paper-style SVD-LLM v2: *simple local update* (no layerwise propagation).

    Update scope:
      - FFN intermediate/output: update V1 and V2 (fix U1/U2)
      - Attention output projection WO (attention.output.dense): update Vo (fix Uo)
    Parameterization in this codebase:
        Y ≈ (X @ U) @ V + b
    For each linear, we update V by ridge least squares using teacher IO pairs:
        Z = X @ U
        Solve (sum Z^T Z + ridge I) V = sum Z^T (Y - b_teacher)

    We accumulate the normal-equation statistics online to keep memory low.
    """
    student.eval()
    teacher.eval()

    # Track memory usage during local update
    mem_start = torch.cuda.memory_allocated() / 1024**2
    print(f"  📊 Local update starting - GPU memory: {mem_start:.1f} MiB")

    def _to_2d(x):
        if x is None:
            return None
        if x.dim() == 3:
            return x.reshape(-1, x.shape[-1])
        if x.dim() == 2:
            return x
        return x.view(-1, x.shape[-1])

    def _subsample_rows(X, Y, max_rows):
        if X.shape[0] <= max_rows:
            return X, Y
        idx = torch.randperm(X.shape[0], device=X.device)[:max_rows]
        return X[idx], Y[idx]

    num_layers = len(student.roberta.encoder.layer)  # RoBERTa layer access

    for i in range(num_layers):
        shim   = student.roberta.encoder.layer[i]  # RoBERTa layer access
        blk    = shim.block
        t_layer = teacher.roberta.encoder.layer[i]  # RoBERTa layer access

        # ---- fixed U (student) ----
        U1 = blk.U1.detach().to(device=device, dtype=torch.float32)   # [dm, r1]
        U2 = blk.U2.detach().to(device=device, dtype=torch.float32)   # [dff, r2]
        Uo = blk.Uo.detach().to(device=device, dtype=torch.float32)   # [dm, ro]

        r1 = U1.shape[1]
        r2 = U2.shape[1]
        ro = Uo.shape[1]

        dff = t_layer.intermediate.dense.out_features
        dm  = t_layer.output.dense.out_features  # == hidden_size

        # ---- teacher bias targets ----
        b1_t = t_layer.intermediate.dense.bias.detach().to(device=device, dtype=torch.float32).view(1, -1)  # [1,dff]
        b2_t = t_layer.output.dense.bias.detach().to(device=device, dtype=torch.float32).view(1, -1)        # [1,dm]
        bo_t = t_layer.attention.output.dense.bias.detach().to(device=device, dtype=torch.float32).view(1, -1)  # [1,dm]

        # ---- online accumulators ----
        A1 = torch.zeros(r1, r1, device=device, dtype=torch.float32)
        B1 = torch.zeros(r1, dff, device=device, dtype=torch.float32)

        A2 = torch.zeros(r2, r2, device=device, dtype=torch.float32)
        B2 = torch.zeros(r2, dm,  device=device, dtype=torch.float32)

        Ao = torch.zeros(ro, ro, device=device, dtype=torch.float32)
        Bo = torch.zeros(ro, dm, device=device, dtype=torch.float32)

        def hook_ffn1(mod, inp, out):
            nonlocal A1, B1
            X = _to_2d(inp[0]).detach().to(device=device, dtype=torch.float32)
            Y = _to_2d(out).detach().to(device=device, dtype=torch.float32)
            X, Y = _subsample_rows(X, Y, max_rows_per_hook)
            Z  = X @ U1
            Yc = Y - b1_t
            A1 += Z.t() @ Z
            B1 += Z.t() @ Yc

        def hook_ffn2(mod, inp, out):
            nonlocal A2, B2
            X = _to_2d(inp[0]).detach().to(device=device, dtype=torch.float32)
            Y = _to_2d(out).detach().to(device=device, dtype=torch.float32)
            X, Y = _subsample_rows(X, Y, max_rows_per_hook)
            Z  = X @ U2
            Yc = Y - b2_t
            A2 += Z.t() @ Z
            B2 += Z.t() @ Yc

        def hook_wo(mod, inp, out):
            nonlocal Ao, Bo
            X = _to_2d(inp[0]).detach().to(device=device, dtype=torch.float32)
            Y = _to_2d(out).detach().to(device=device, dtype=torch.float32)
            X, Y = _subsample_rows(X, Y, max_rows_per_hook)
            Z  = X @ Uo
            Yc = Y - bo_t
            Ao += Z.t() @ Z
            Bo += Z.t() @ Yc

        h1 = t_layer.intermediate.dense.register_forward_hook(hook_ffn1)
        h2 = t_layer.output.dense.register_forward_hook(hook_ffn2)
        ho = t_layer.attention.output.dense.register_forward_hook(hook_wo)

        seen = 0
        for batch in loader:
            if seen >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            seen += 1

        h1.remove()
        h2.remove()
        ho.remove()

        # ---- solve V ----
        I1 = torch.eye(r1, device=device, dtype=torch.float32)
        I2 = torch.eye(r2, device=device, dtype=torch.float32)
        Io = torch.eye(ro, device=device, dtype=torch.float32)

        V1_new = torch.linalg.solve(A1 + ridge * I1, B1)  # [r1, dff]
        V2_new = torch.linalg.solve(A2 + ridge * I2, B2)  # [r2, dm]
        Vo_new = torch.linalg.solve(Ao + ridge * Io, Bo)  # [ro, dm]

        blk.V1.data.copy_(V1_new.to(dtype=blk.V1.dtype))
        blk.V2.data.copy_(V2_new.to(dtype=blk.V2.dtype))
        blk.Vo.data.copy_(Vo_new.to(dtype=blk.Vo.dtype))

        # Aggressive cleanup to minimize memory footprint during iteration
        del A1, B1, A2, B2, Ao, Bo
        del V1_new, V2_new, Vo_new
        del U1, U2, Uo
        del b1_t, b2_t, bo_t
        del I1, I2, Io  # Also clean up identity matrices
        torch.cuda.empty_cache()

        print(f"[v2-simple] updated FFN V + WO V (online) at layer {i}")

    # Track memory usage at end of local update
    mem_end = torch.cuda.memory_allocated() / 1024**2
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    mem_change = mem_end - mem_start

    print(f"\n  📊 Local update completed:")
    print(f"    • Start memory: {mem_start:.1f} MiB")
    print(f"    • End memory: {mem_end:.1f} MiB")
    print(f"    • Peak memory during update: {mem_peak:.1f} MiB")
    print(f"    • Net change: {mem_change:+.1f} MiB")
    print(f"  ⚠️  Note: Final peak will include both local update and inference phases")

    return student

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
    tokz = AutoTokenizer.from_pretrained(MODEL_DIR)

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

    cfg = AutoConfig.from_pretrained(MODEL_DIR, num_labels=num_labels, problem_type=problem_type)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, config=cfg)
    model = model.to(device).eval()
    # Keep a dense teacher for v2 local update (must be BEFORE replacing layers)
    teacher = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, config=cfg).to(device).eval()

    # ---- SVD-LLM v1 official-style rank by ratio ----
    RATIO = 0.5  # try 0.5 / 0.3 / 0.2 later

    dm  = cfg.hidden_size              # 768
    dff = cfg.intermediate_size        # 3072
    H   = cfg.num_attention_heads      # 12
    dh  = dm // H
    RANK_ATTN  = rank_from_ratio(dm, dh, RATIO)  # per-head rank for Q/K/V
    RANK_FF    = rank_from_ratio(dm, dff, RATIO)  # rank for FFN Wi and Wo
    RANK_WO    = rank_from_ratio(dm, dm, RATIO)   # rank for attention output projection Wo
    print(f"BATCH_SIZE: {BATCH_SIZE}  RANK_ATTN: {RANK_ATTN}  RANK_FF: {RANK_FF}  RANK_WO: {RANK_WO}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ─── Calibration pass: collect covariances ───────────────────────────────
    print("Calibrating input covariances (Whitening for RoBERTa)…")
    covs = calibrate_covariances(model, loader, device, max_batches=4)

    # ─── Replace each encoder layer with data-aware low-rank block ───────────
    for i, layer in enumerate(model.roberta.encoder.layer):  # RoBERTa layer access
        blk = SVDBlock(
            hf_layer=layer,
            rank_attn=RANK_ATTN,
            rank_ff=RANK_FF,
            cov_attn_in=covs["cov_attn_in"][i].to(device),
            cov_attn_out=covs["cov_attn_out"][i].to(device),
            cov_ffn_in=covs["cov_ffn_in"][i].to(device),
            cov_ffn_out=covs["cov_ffn_out"][i].to(device),
            rank_wo=RANK_WO,
        )
        model.roberta.encoder.layer[i] = LayerShim(blk).to(device).eval().float()  # RoBERTa layer replacement

    print("Running SVD-LLM v2 SIMPLE local update (FFN V + WO V, paper-style)...")
    model = svdllm_v2_simple_local_update_ffn_wo_V(
        student=model,
        teacher=teacher,
        loader=loader,
        device=device,
        max_batches=4,
        max_rows_per_hook=4096,
        ridge=1e-4,
    )

    # ─── Explicitly release teacher model to free GPU memory for fair inference peak ───
    print("\n🗑️  Releasing teacher model to free memory...")

    # Record memory before release
    mem_before = torch.cuda.memory_allocated() / 1024**2
    print(f"  • Memory before teacher release: {mem_before:.1f} MiB")

    # Move teacher to CPU first (safer than direct deletion)
    try:
        teacher.to("cpu")
        print(f"  • Teacher moved to CPU ✅")
    except Exception as e:
        print(f"  • Warning: Could not move teacher to CPU: {e}")

    # Delete teacher reference
    del teacher
    print(f"  • Teacher reference deleted ✅")

    # Aggressive GPU memory cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Record memory after release
    mem_after = torch.cuda.memory_allocated() / 1024**2
    mem_freed = mem_before - mem_after
    print(f"  • Memory after teacher release: {mem_after:.1f} MiB")
    print(f"  • Memory freed: {mem_freed:.1f} MiB ✅")
    print()


    # =============================================================

    # ─── Memory accounting (persistent params only) ──────────────────────────
    def summarize_dense_vs_lowrank(model):
        dense_bytes, lowrank_bytes = 0, 0
        for name, p in model.named_parameters():
            size = p.numel() * p.element_size()
            if ".block." in name or (
                name.startswith("roberta.encoder.layer")  # RoBERTa namespace
                and any(part in name for part in ("Pq","Vq","Pk","Vk","Pv","Vv","U1","V1","U2","V2","Uo","Vo"))
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

    print(f"Persistent low-rank model storage (RoBERTa Whitening v2): {baseline_bytes/1024**2:6.1f} MiB")

    # ─── Evaluate ────────────────────────────────────────────────────────────
    # Capture peak before evaluate (includes local update peak)
    peak_before_eval = torch.cuda.max_memory_allocated() / 1024**2

    metric_name = "pearson" if task_name == "stsb" else "acc"
    acc, peak_lr, t = acc_peak_time(model, loader, device, task_name)

    # Use the true peak (max of all phases)
    true_peak = max(peak_before_eval, peak_lr)
    print(f"RoBERTa Whitening v2 | {metric_name}={acc:.4f} | peak ={true_peak:6.1f} MiB | {t:6.1f} ms/b")
    print(f"  (Peak before eval: {peak_before_eval:.1f} MiB, Peak during eval: {peak_lr:.1f} MiB)")
