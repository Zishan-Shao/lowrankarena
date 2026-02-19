# adaptive_rank_selection.py  (paper-compliant, NAACL 2024)
# -----------------------------------------------------------------------------
# Changes vs adasvd_refactored:
#   1. parameter_budget() → scalar log form (paper Eq.8), receives SOFT masks
#   2. collect_op_metadata() — new function for PaperHN
#   3. PaperHN — fixed random z buffer + LayerNorm + meta_proj + GRU + heads
#   4. topk_like() — clamp(1, ...) to prevent k=0 zero-gradient
# -----------------------------------------------------------------------------

import os, sys, json, math, time, random, argparse
from typing import List, Tuple
from collections import Counter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig

# ------------------------------ Utils ----------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = True):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y = torch.sigmoid((logits + g) / tau)
    if not hard:
        return y
    y_hard = (y > 0.5).float()
    return (y_hard - y).detach() + y  # straight-through

# ----------------------- SVD wrappers for Linear -----------------------------
def svd_of_linear_weight(linear: nn.Linear, device: str):
    W = linear.weight.data  # [out, in]
    Win = W.t().float().to(device)  # [in, out]
    U, s, Vh = torch.linalg.svd(Win, full_matrices=False)  # U:[in,R], s:[R], Vh:[R,out]
    V = Vh.t().contiguous()
    return U, s, V  # FP32 on device

class MaskedSVDLinear(nn.Module):
    """Drop-in SVD wrapper. Forward expects mask set by context (self._current_mask)."""
    def __init__(self, linear: nn.Linear, device: str, rank_cap: int = None):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

        U, s, V = svd_of_linear_weight(linear, device)
        Rfull = s.numel()
        R = Rfull if rank_cap is None else min(rank_cap, Rfull)
        self.U = nn.Parameter(U[:, :R], requires_grad=False)      # [in,R]
        self.s = nn.Parameter(s[:R],   requires_grad=False)       # [R]
        self.V = nn.Parameter(V[:, :R], requires_grad=False)      # [out,R]
        self.R = R
        self._current_mask = None

    def forward(self, x: torch.Tensor):
        m = self._current_mask
        if m is None:
            raise RuntimeError("MaskedSVDLinear: _current_mask is None (masking context not set).")
        ms  = (m * self.s).to(x.dtype)                    # [R]
        US  = (self.U * ms.unsqueeze(0)).to(x.dtype)      # [in,R]
        mid = torch.matmul(x, US)                         # [...,R]
        y   = torch.matmul(mid, self.V.t().to(x.dtype))   # [...,out]
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)
        return y

    def param_count_given_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return (self.in_features + self.out_features) * torch.sum(mask)

# --------------------- Original HyperNetwork (kept for reference) ------------
class SimpleHN(nn.Module):
    """Original hypernetwork: Embedding + GRU → per-op heads → logits."""
    def __init__(self, op_sizes: List[int], feat_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.op_sizes = op_sizes
        self.L        = len(op_sizes)
        self.embed    = nn.Embedding(self.L, feat_dim)
        self.gru      = nn.GRU(input_size=feat_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.heads    = nn.ModuleList([nn.Linear(hidden, r) for r in op_sizes])

    def forward(self) -> List[torch.Tensor]:
        ids = torch.arange(self.L, device=self.embed.weight.device).long().unsqueeze(0)  # [1,L]
        z   = self.embed(ids)                                                             # [1,L,feat]
        h,_ = self.gru(z)                                                                 # [1,L,H]
        h   = h.squeeze(0)                                                                # [L,H]
        return [self.heads[i](h[i]) for i in range(self.L)]

# ----------------------- Paper-compliant HyperNetwork ------------------------
def collect_op_metadata(linear_list: List[Tuple[str, nn.Linear]]) -> List[List[float]]:
    """
    Build fixed metadata features per op: [in_norm, out_norm, one_hot_type×7].
    Shape: [L, 9].

    Type keys (index 0-5, 6=other):
      0: attention.self.query
      1: attention.self.key
      2: attention.self.value
      3: attention.output.dense
      4: intermediate.dense
      5: output.dense
      6: other
    """
    type_keys = [
        "attention.self.query",    # 0
        "attention.self.key",      # 1
        "attention.self.value",    # 2
        "attention.output.dense",  # 3
        "intermediate.dense",      # 4
        "output.dense",            # 5
    ]                              # 6 = other
    max_dim = 4096.0
    meta = []
    for name, lin in linear_list:
        one_hot = [0.0] * 7
        matched = False
        for idx, key in enumerate(type_keys):
            if key in name:
                one_hot[idx] = 1.0
                matched = True
                break
        if not matched:
            one_hot[6] = 1.0
        meta.append([lin.in_features / max_dim,
                     lin.out_features / max_dim] + one_hot)
    return meta  # List[List[float]], shape [L, 9]


class PaperHN(nn.Module):
    """
    Paper-compliant ARS hypernetwork (NAACL 2024).
    - z: fixed random buffer (not trained) — paper: "fixed random sampling z"
    - meta_proj: trainable linear on op metadata
    - LayerNorm (no learnable params) for scale alignment
    - GRU: cross-op budget competition context
    - per-op Linear heads → logits

    engineering_stable=False (default): strict paper
    engineering_stable=True (ablation):  learned alpha_z scalar gate on z
    """
    def __init__(self, op_sizes: List[int], op_metadata: List[List[float]],
                 feat_dim: int = 16, hidden: int = 64,
                 engineering_stable: bool = False,
                 budget: float = 0.5):
        super().__init__()
        self.L                  = len(op_sizes)
        meta_dim                = len(op_metadata[0])  # 9
        self.engineering_stable = engineering_stable
        gru_in_dim              = feat_dim + feat_dim   # z_part + meta_proj

        # Fixed random z: NOT trained (paper requirement)
        self.register_buffer('z',    torch.randn(self.L, feat_dim))
        self.register_buffer('meta', torch.tensor(op_metadata, dtype=torch.float32))

        # Trainable metadata projection
        self.meta_proj = nn.Linear(meta_dim, feat_dim, bias=False)

        if engineering_stable:
            # Learned scalar gate on z — ablation only
            self.alpha_z = nn.Parameter(torch.tensor(1.0))

        # LayerNorm without extra learnable params → pure scale alignment
        self.ln_in = nn.LayerNorm(gru_in_dim, elementwise_affine=False)

        # GRU: cross-op context for budget competition
        self.gru = nn.GRU(input_size=gru_in_dim, hidden_size=hidden,
                          num_layers=1, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, r) for r in op_sizes])
        # Budget-aware bias init: start ratio_soft just ABOVE budget so one-sided
        # log loss (paper Eq.8) is active from step 0 and can push ratio DOWN.
        # init_p = budget + 0.15, clamped to [0.55, 0.95].
        # bias = logit(init_p) = log(p/(1-p))
        init_p = float(max(0.55, min(0.95, budget + 0.15)))
        init_bias = math.log(init_p / (1.0 - init_p))
        for head in self.heads:
            nn.init.constant_(head.bias, init_bias)

    def forward(self) -> List[torch.Tensor]:
        m_out  = self.meta_proj(self.meta)                                    # [L, feat_dim]
        z_part = self.alpha_z * self.z if self.engineering_stable else self.z  # [L, feat_dim]
        inp    = self.ln_in(torch.cat([z_part, m_out], dim=-1))               # [L, 2*feat_dim]
        h, _   = self.gru(inp.unsqueeze(0))                                   # [1, L, hidden]
        h      = h.squeeze(0)                                                  # [L, hidden]
        return [self.heads[i](h[i]) for i in range(self.L)]

# -------------------- Model patcher: Linear -> MaskedSVDLinear ----------------
def collect_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    liners = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            liners.append((name, mod))
    return liners

def replace_with_masked(model: nn.Module, device: str, rank_cap_per_op: List[int],
                        original_names: List[str]):
    ops = []
    for name, Rcap in zip(original_names, rank_cap_per_op):
        parent = model
        *parents, last = name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        lin = getattr(parent, last)
        wrapped = MaskedSVDLinear(lin, device=device, rank_cap=Rcap)
        setattr(parent, last, wrapped)
        ops.append((name, wrapped))
    return model, ops

# ---------------------------- Loss functions ---------------------------------
def topk_like(mask: torch.Tensor, s: torch.Tensor,
              k_ref: torch.Tensor = None) -> torch.Tensor:
    """Top-k binary mask aligned to s ordering.

    k comes from k_ref.sum() if provided (e.g. hard mask), else from mask.sum().
    Using k_ref=masks_hard breaks the self-referential lock where k is always
    equal to the current soft sum — enabling budget loss to actually shrink k.
    k clamped to [1, R] to avoid zero-gradient.
    """
    with torch.no_grad():
        src = k_ref if k_ref is not None else mask
        k = int(src.sum().detach().round().clamp(1, mask.numel()).item())
        idx = torch.arange(mask.numel(), device=mask.device)
        m_top = torch.zeros_like(mask)
        m_top[idx[:k]] = 1.0
    return m_top

def alignment_loss(mask: torch.Tensor, s: torch.Tensor,
                   k_ref: torch.Tensor = None) -> torch.Tensor:
    """Alignment loss (paper Eq.7 spirit).

    mask: soft sigmoid — gradient flows through this.
    k_ref: hard Gumbel-sigmoid mask (detached) — determines target k.
    When k_ref is provided, k = k_ref.sum() instead of mask.sum(), so:
      budget forces logits down → hard k decreases → alignment must pull
      soft mask lower → feedback loop is unblocked.
    """
    m_top = topk_like(mask, s, k_ref=k_ref)
    return torch.sum(((mask - m_top) * s) ** 2)

def parameter_budget(op_list: List[MaskedSVDLinear], masks: List[torch.Tensor],
                     p: float) -> torch.Tensor:
    """
    Paper Eq.8: R(a,b) = log(max(T, Tmax) / Tmax), one-sided log form.

    masks MUST be SOFT (torch.sigmoid output) so gradients flow through T.
    T and Tmax are SCALARS (sum over all ops).
    One-sided: T < Tmax → log(Tmax/Tmax) = 0; T > Tmax → log(T/Tmax) > 0.
    """
    device = masks[0].device
    T          = torch.zeros((), device=device)
    T_original = torch.zeros((), device=device)
    for op, m in zip(op_list, masks):
        # m is soft sigmoid → sum gives expected rank (smooth gradient)
        T          = T + (op.in_features + op.out_features) * m.sum()
        T_original = T_original + float(op.in_features * op.out_features)
    Tmax = p * T_original                                     # target (scalar)
    return torch.log(torch.clamp(T, min=Tmax) / (Tmax + 1e-12))
