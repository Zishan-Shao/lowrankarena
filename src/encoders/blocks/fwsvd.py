from typing import Tuple, Dict, Optional, Callable
from collections import defaultdict

from tqdm import tqdm

import torch
import torch.linalg as LA
from torch.utils.data import DataLoader
from transformers import BertModel, PreTrainedModel


# this will compute the FWSVD for the FF Layers
def compute_row_sum_svd_decomposition(A: torch.Tensor, weights: Optional[torch.Tensor] = None, rank: Optional[int] = None):
    """Computes FWSVD from https://arxiv.org/pdf/2207.00112.pdf.

    Args: 
      A (torch.Tensor): matrix of size (H, W) to decompose, where H is the hidden dimension, W is the intermediate
      weights (Optional[torch.Tensor]): matrix of size (H, W) or (H,) - Fisher weights.
        If None (default), set to ones.
      rank (Optional[int]): approx. rank in SVD. If None (default), computes
        full-rank decomposition without compression.
    
    Returns:
      left_w (torch.Tensor): matrix [H, r] = I_hat_inv @ Ur @ Sr
      right_w (torch.Tensor): matrix [r, W] = Vr.T
    """
    h, w = A.shape
    orig_dtype = A.dtype

    if weights is None:
        weights = torch.ones(h)

    if weights.ndim > 1:
        weights = weights.sum(dim=1)

    i_hat = torch.diag(torch.sqrt(weights + 1e-5))
    i_hat_inv = LA.inv(i_hat)  # actually it's diagonal so we can just take 1 / i_hat

    u, s, v = LA.svd(i_hat @ A.to(i_hat.dtype), full_matrices=True)
    s = torch.diag(s)  # more convenient form

    if rank is not None:
        u = u[:, :rank]
        s = s[:rank, :rank]
        v = v[:rank]
    else:
        s_tmp = s
        s = torch.zeros_like(A)
        s[:min(h, w), :min(h, w)] = s_tmp

    left_w = (i_hat_inv @ (u @ s)).to(orig_dtype)
    right_w = v.to(orig_dtype)

    return left_w, right_w



# NEW: this help finds the fisher weights of multi-head attention
def estimate_fisher_weights_bert_with_attention(
    model: BertModel,
    dataloader: DataLoader,
    compute_full: bool = False,
    device: str = 'cuda'
):
    """
    Returns six dicts keyed by layer index:
      fisher_q, fisher_k, fisher_v  each of shape [d_model] (or summed to [dh] per head)
      fisher_int, fisher_out         each of shape [d_model] (or [intermediate] for FFN)
    """
    model = model.to(device).train()
    cfg   = model.config
    d_model = cfg.hidden_size
    H       = cfg.num_attention_heads
    dh      = d_model // H
    dint    = cfg.intermediate_size

    # initialize accumulators
    fisher_q   = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_k   = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_v   = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_int = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_out = defaultdict(lambda: torch.zeros(dint,   device=device))

    model_dtype = next(model.parameters()).dtype
    use_autocast = model_dtype in (torch.float16, torch.bfloat16)

    for batch in dataloader:
        # move inputs to device…
        inputs = {k: v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.autocast('cuda', dtype=model_dtype, enabled=use_autocast):
            outputs = model(**inputs)
            loss    = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        loss.backward()

        for i in range(cfg.num_hidden_layers):
            # attention: [out_features, in_features] grads => transpose to [in, out]
            q_grad = model.bert.encoder.layer[i].attention.self.query.weight.grad.data.t()  ** 2
            k_grad = model.bert.encoder.layer[i].attention.self.key.weight.grad.data.t()    ** 2
            v_grad = model.bert.encoder.layer[i].attention.self.value.weight.grad.data.t()  ** 2

            # flatten to vector of length d_model
            fisher_q[i]   += q_grad.sum(dim=1) if not compute_full else q_grad
            fisher_k[i]   += k_grad.sum(dim=1) if not compute_full else k_grad
            fisher_v[i]   += v_grad.sum(dim=1) if not compute_full else v_grad

            # FFN intermediate
            int_grad = model.bert.encoder.layer[i].intermediate.dense.weight.grad.data.t() ** 2
            out_grad = model.bert.encoder.layer[i].output.dense.weight.grad.data.t()      ** 2

            fisher_int[i] += int_grad.sum(dim=1) if not compute_full else int_grad
            fisher_out[i] += out_grad.sum(dim=1) if not compute_full else out_grad

        model.zero_grad()

    # normalize each dict to [0,1]
    def normalize(d):
        return {i: v / v.max() for i,v in d.items()}

    return (normalize(fisher_q),
            normalize(fisher_k),
            normalize(fisher_v),
            normalize(fisher_int),
            normalize(fisher_out))


# ── ModernBERT FWSVD (pre-norm, fused Wqkv, GeGLU FFN) ────────────────────

def estimate_fisher_weights_modernbert(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str = 'cuda',
):
    """Compute Fisher weights for ModernBERT layers.

    ModernBERT submodule layout per layer (model.model.layers[i]):
      layer.attn.Wqkv  — fused [3D, D]; split into Q/K/V each [D, D]
      layer.attn.Wo    — [D, D]  (kept dense in NaiveModernBertSVDBlock)
      layer.mlp.Wi     — [2*d_ffn, D]  (GeGLU gate+linear concat)
      layer.mlp.Wo     — [D, d_ffn]

    Returns five normalized dicts keyed by layer index:
      fisher_q, fisher_k, fisher_v  — shape [D]  (from Wqkv Q/K/V split)
      fisher_wi                      — shape [D]  (from mlp.Wi input dim)
      fisher_wo_ffn                  — shape [d_ffn]  (from mlp.Wo input dim)

    Note: Wo_attn is NOT returned because NaiveModernBertSVDBlock keeps it dense.
    """
    model = model.to(device).train()
    cfg = model.config
    d_model = cfg.hidden_size
    num_layers = cfg.num_hidden_layers
    d_ffn = model.model.layers[0].mlp.Wo.in_features  # ffn intermediate dim

    fisher_q      = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_k      = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_v      = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_wo_attn= defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_wi     = defaultdict(lambda: torch.zeros(d_model, device=device))
    fisher_wo_ffn = defaultdict(lambda: torch.zeros(d_ffn,   device=device))

    model_dtype = next(model.parameters()).dtype
    use_autocast = model_dtype in (torch.float16, torch.bfloat16)

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.autocast('cuda', dtype=model_dtype, enabled=use_autocast):
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        loss.backward()

        for i in range(num_layers):
            layer = model.model.layers[i]

            # Wqkv grad: [3D, D] — split into Q/K/V
            g = layer.attn.Wqkv.weight.grad.data          # [3D, D]
            gq, gk, gv = torch.chunk(g, 3, dim=0)         # each [D, D]
            # .t() → [D, D], .sum(dim=1) → [D]  (sum over output heads)
            fisher_q[i]   += (gq.t() ** 2).sum(dim=1)
            fisher_k[i]   += (gk.t() ** 2).sum(dim=1)
            fisher_v[i]   += (gv.t() ** 2).sum(dim=1)

            # attn.Wo grad: [D, D] → .t() → [D, D] → sum → [D]
            fisher_wo_attn[i] += (layer.attn.Wo.weight.grad.data.t() ** 2).sum(dim=1)

            # mlp.Wi grad: [2*d_ffn, D] → .t() → [D, 2*d_ffn] → sum → [D]
            fisher_wi[i] += (layer.mlp.Wi.weight.grad.data.t() ** 2).sum(dim=1)

            # mlp.Wo grad: [D, d_ffn] → .t() → [d_ffn, D] → sum → [d_ffn]
            fisher_wo_ffn[i] += (layer.mlp.Wo.weight.grad.data.t() ** 2).sum(dim=1)

        model.zero_grad()

    def normalize(d):
        return {i: v / (v.max() + 1e-8) for i, v in d.items()}

    return (normalize(fisher_q), normalize(fisher_k), normalize(fisher_v),
            normalize(fisher_wo_attn), normalize(fisher_wi), normalize(fisher_wo_ffn))


def build_fwsvd_helpers_modernbert(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str = 'cuda',
    eps: float = 1e-6,
):
    """Return (per_head_fn, low_rank_fn) for ModernBERT FWSVD.

    Interface is identical to build_fwsvd_helpers() in svd_helpers.py so that
    run_encoder_benchmark._build_fwsvd() can call it without further changes.

    Internally builds a data_ptr → fisher_vector map that the two closures
    look up when called from NaiveModernBertSVDBlock.__init__:
      - per_head_fn is called for Wq, Wk, Wv (split from Wqkv)
      - low_rank_fn is called for Wo_attn, Wi, and Wo_ffn
    """
    model = model.to(device).train()

    fisher_q, fisher_k, fisher_v, fisher_wo_attn, fisher_wi, fisher_wo_ffn = \
        estimate_fisher_weights_modernbert(model, dataloader, device)

    H = model.config.num_attention_heads
    fw_map: dict = {}

    for i, layer in enumerate(model.model.layers):
        # Split Wqkv [3D, D] into views for Q / K / V
        W = layer.attn.Wqkv.weight.data          # [3D, D]
        Wq, Wk, Wv = torch.chunk(W, 3, dim=0)   # each [D, D], views of W

        # .t() is a non-contiguous view; data_ptr == chunk's data_ptr
        fw_map[Wq.t().data_ptr()] = fisher_q[i]      + eps
        fw_map[Wk.t().data_ptr()] = fisher_k[i]      + eps
        fw_map[Wv.t().data_ptr()] = fisher_v[i]      + eps

        fw_map[layer.attn.Wo.weight.data.data_ptr()] = fisher_wo_attn[i] + eps

        # Wi.weight.data.t() has the same data_ptr as Wi.weight.data
        fw_map[layer.mlp.Wi.weight.data.data_ptr()] = fisher_wi[i]     + eps
        fw_map[layer.mlp.Wo.weight.data.data_ptr()] = fisher_wo_ffn[i] + eps

    def fwsvd_per_head(Wt: torch.Tensor, rank: int):
        """Fisher-weighted per-head SVD for Q / K / V projections."""
        ptr = Wt.data_ptr()
        w   = fw_map[ptr]               # [D]
        d_model = Wt.shape[0]
        dh      = Wt.shape[1] // H
        Wh      = Wt.view(d_model, H, dh)
        Us, Vs  = [], []
        for h in range(H):
            U, V = compute_row_sum_svd_decomposition(Wh[:, h, :], weights=w, rank=rank)
            Us.append(U)
            Vs.append(V)
        return torch.stack(Us, 0), torch.stack(Vs, 0)

    def fwsvd_low_rank(W: torch.Tensor, rank: int):
        """Fisher-weighted full-matrix SVD for Wi / Wo_ffn."""
        ptr = W.data_ptr()
        w   = fw_map[ptr]
        return compute_row_sum_svd_decomposition(W, weights=w, rank=rank)

    return fwsvd_per_head, fwsvd_low_rank
