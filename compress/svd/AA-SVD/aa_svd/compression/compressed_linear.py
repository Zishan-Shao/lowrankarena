import os
import json
import torch
import wandb
import logging
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from .utils import get_log_key

logger = logging.getLogger(__name__)


class CompressedLinear(nn.Module):
    """
    A custom linear layer that uses SVD for parameter reduction.
    This makes the compressed layers easily identifiable for later analysis.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = None, U: torch.Tensor = None, S: torch.tensor = None, V: torch.Tensor = None, bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if rank is None:
            assert U is not None and V is not None, "If rank is not specified, U and V must be provided"
            rank = U.shape[1]  # Assuming U is (out_features, rank)

        self.rank = rank

        # if self.rank >= (in_features*out_features)/min(in_features, out_features):
        #     self.W = nn.Linear(in_features, out_features, bias=bias)
        #     if U is not None and V is not None:
        #         if S is None:
        #             self.W.weight.data = U @ V.t()
        #         else:
        #             self.W.weight.data = U @ torch.diag(S) @ V.t()
        # else:
        self.W = None
        self.U = nn.Linear(rank, out_features, bias=bias)
        self.V = nn.Linear(in_features, rank, bias=False)

        if S is None:
            self.U.weight.data = U
            self.V.weight.data = V.t()
        else:
            self.U.weight.data = U@torch.diag(torch.sqrt(S))
            self.V.weight.data = (V@torch.diag(torch.sqrt(S))).t()

    def forward(self, x):
        x = self.U(self.V(x))
        return x

    def get_recon_weight(self):
        """Returns the reconstructed weight matrix"""
        if self.W is not None:
            return self.W.weight
        else:
            return self.U.weight @ self.V.weight

    def _get_reconstruction_error(self, original_w=None) -> float:
        """Calculate reconstruction error if using reduced rank"""
        W_reconstructed = self.get_recon_weight()
        weight = original_w.data
        error = torch.norm(weight - W_reconstructed).item()
        return error

    def save_tensor(self, tensor, path, name):
        save_path = os.path.join(path, f'{name}.pt')
        torch.save(tensor, save_path)

    @classmethod
    def load_tensor(cls, path, name):
        load_path = os.path.join(path, f'{name}.pt')

        if not os.path.exists(load_path):
            return None

        return torch.load(load_path)

    def save(self, path):
        """Saves the U and V matrices to the specified path"""

        os.makedirs(path, exist_ok=True)

        self.save_tensor(self.U.weight.data, path, 'U')

        self.save_tensor(self.V.weight.data.t(), path, 'V')
        print(f"SVD components saved to {path}")

    def load(self, path):
        """Loads the U and V matrices from the specified path"""
        self.U = type(self).load_tensor(path, 'U')
        self.S = type(self).load_tensor(path, 'S')
        self.V = type(self).load_tensor(path, 'V')

        # check if U and V are not None
        if self.U is None or self.V is None:
            raise ValueError(f"Files to load SVD components from {path} not available.")
        print(f"SVD components loaded from {path}")

    @classmethod
    def _load(cls, path):
        U = cls.load_tensor(path, 'U')
        S = cls.load_tensor(path, 'S')
        V = cls.load_tensor(path, 'V')

        # check if U and V are not None
        if U is None or V is None:
            raise ValueError(f"Files to load SVD components from {path} not available.")
        print(f"SVD components loaded from {path}")
        return U, S, V

    @classmethod
    def from_path(cls, path, bias=None):
        U, S, V = cls._load(path)
        return cls.from_linear(U, S, V, bias=bias)

    def _get_compression_ratio(self) -> float:
        """Calculate compression ratio if using reduced rank"""

        in_dim, out_dim, rank = self.in_features, self.out_features, self.rank

        bias_params = (1 if self.U.bias is not None else 0) * out_dim
        original_params = in_dim * out_dim + bias_params
        svd_params = rank * (in_dim + out_dim) + bias_params
        return svd_params / original_params

    def log_metrics(self, original_w, name, device=None):

        original_device = self.U.weight.device

        if device is not None:
            self.to(device)
            original_w = original_w.to(device)

        metrics = {
                    "rank": self.U.weight.shape[1],
                    "compression_ratio": self._get_compression_ratio(),
                    "recon_error": self._get_reconstruction_error(original_w)
                  }

        if device is not None:
            self.to(original_device)
            original_w = original_w.to(original_device)

        # log metrics
        logger.info(f"SVD Compression metrics for {name}: {json.dumps(metrics, indent=2)}")
        if wandb.run is not None:
            to_log = {}
            for k, v in metrics.items():
                log_key, layer_idx = get_log_key(name, k)
                to_log[log_key] = v
            to_log.update({'layer_idx': int(layer_idx)})
            wandb.log(to_log)

    def log_calibration_metrics(self, original_w, xTx, xhatTxhat, xTxhat, name, device=None):
        """
        Log calibration metrics for the compressed linear layer.
        """
        original_device = self.U.weight.device

        if device is not None:
            self.to(device)
            original_w = original_w.to(device)
            xTx = xTx.to(device)
            xhatTxhat = xhatTxhat.to(device)
            xTxhat = xTxhat.to(device)

        original_w = original_w.float()
        xTx = xTx.float()
        xhatTxhat = xhatTxhat.float()
        xTxhat = xTxhat.float()
        W_reconstructed = self.get_recon_weight().float()

        diff_W = original_w - W_reconstructed
        cal_metric_1 = torch.trace(diff_W @ xTx @ diff_W.t()).item()
        # cal_metric_2 = torch.trace(diff_W @ xhatTxhat @ diff_W.t()).item()

        # Gp_inv = torch.linalg.pinv(xhatTxhat)
        # What = W_reconstructed - original_w @ (xTxhat @ Gp_inv)
        # cal_metric_3_proj = torch.trace(What @ xhatTxhat @ What.T).item()

        # xhat = xTx - xTxhat @ (Gp_inv @ xTxhat.T)
        # # xhat = 0.5 * (xhat + xhat.T)  # symmetrize
        # cal_metric_3_rel = torch.trace(original_w @ xhat @ original_w.T).item()

        # cal_metric_3 = cal_metric_3_proj + cal_metric_3_rel

        # metrics = {
        #     "(w-w')x": cal_metric_1,
        #     "(w-w')x'": cal_metric_2,
        #     "(w'x'-wx)": cal_metric_3,
        #     "(w'x'-wx)_p": cal_metric_3_proj,
        #     "(w'x'-wx)_r": cal_metric_3_rel,
        # }

        # cal_metric_3 = norm(WX - W'X')^2 = norm(WX)^2 + norm(W'X')^2 - 2*trace(WX X'^T W'^T)
        WX_norm_sq = torch.trace(original_w @ xTx @ original_w.t()).item()
        WXhat_norm_sq = torch.trace(original_w @ xhatTxhat @ original_w.t()).item()
        W_hatX_hat_norm_sq = torch.trace(W_reconstructed @ xhatTxhat @ W_reconstructed.t()).item()
        cross_term = 2 * torch.trace(original_w @ xTxhat @ W_reconstructed.t()).item()
        cal_metric_3 = (WX_norm_sq + W_hatX_hat_norm_sq - cross_term)/(original_w.shape[0])  # normalize by output dim to match mse

        # cal_metric_4 = norm(WX - WX')^2 = norm(WX)^2 + norm(WX')^2 - 2*trace(WX X'^T W^T)
        cal_metric_4 = WX_norm_sq + WXhat_norm_sq - 2 * torch.trace(original_w @ xTxhat @ original_w.t()).item()
        cal_metric_4 /= (original_w.shape[0])  # normalize by output dim to match mse

        metrics = {
            "(w-w')x": cal_metric_1,
            "(wx-w'x')": cal_metric_3,
            "(wx-wx')": cal_metric_4,
            "wx_norm_sq": WX_norm_sq/(original_w.shape[0]),
            "wx'_norm_sq": WXhat_norm_sq/(original_w.shape[0]),
            "(wx)-(wx')": (WX_norm_sq - WXhat_norm_sq)/(original_w.shape[0]),
        }

        if device is not None:
            self.to(original_device)
            original_w = original_w.to(original_device)
            xTx = xTx.to(original_device)
            xhatTxhat = xhatTxhat.to(original_device)
            xTxhat = xTxhat.to(original_device)

        # log metrics
        logger.info(f"SVD Calibration metrics for {name}: {json.dumps(metrics, indent=2)}")
        if wandb.run is not None:
            to_log = {}
            for k, v in metrics.items():
                log_key, layer_idx = get_log_key(name, k)
                to_log[log_key] = v
            to_log.update({'layer_idx': int(layer_idx)})
            wandb.log(to_log)

    @classmethod
    def from_linear(cls, U, S, V, bias=None):
        """
        Create a CompressedLinear layer from two weight matrices.

        Args:
            left_w: Left weight matrix (in_features x rank)
            right_w: Right weight matrix (rank x out_features)
            S: Optional singular values for further compression (not used here)

        Returns:
            CompressedLinear: An instance of the CompressedLinear layer
        """
        in_features = V.shape[0]
        out_features = U.shape[0]
        rank = U.shape[1]  # Assuming U is (in_features, rank)
        module = cls(in_features, out_features, rank=rank, U=U, S=S, V=V, bias=True if bias is not None else False)

        if bias is not None:
            module.U.bias.data = bias

        return module


class QuantizedCompressedLinear(CompressedLinear):
    """
    Extends CompressedLinear with quantized storage only.

    In memory, fp16 weights are maintained identically to CompressedLinear
    (self.U and self.V as nn.Linear). Forward pass is fully inherited.

    On save, weights are quantized to 8-bit using per-column absmax (DOBI scheme):
        - U       (M, r): 8-bit quantized  → r int8 columns + r fp16 scales
        - V[:M]   (M, r): 8-bit quantized  → r int8 columns + r fp16 scales
        - V[M:]   (N-M, r): stored as fp16
    where M = out_features, N = in_features (requires out_features <= in_features).

    On load (from_path), quantized tensors are dequantized back to fp16 before
    constructing the module. Quantization benefits apply to storage only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = None,
        U: torch.Tensor = None,
        S: torch.Tensor = None,
        V: torch.Tensor = None,
        bias: bool = True,
    ):
        # Use parent's init: stores U, V as fp16 nn.Linear; forward is standard.
        super().__init__(in_features, out_features, rank=rank, U=U, S=S, V=V, bias=bias)

    # forward, get_recon_weight, _get_reconstruction_error: fully inherited.

    # ------------------------------------------------------------------
    # Quantization / dequantization helpers (used only in save / _load)
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_cols(
        matrix: torch.Tensor,  # (M, r) fp32, on CUDA
        code: torch.Tensor,
    ) -> tuple:
        """
        Quantize each column of *matrix* with an independent absmax scale.

        Stores result transposed: output q_T has shape (r, M) so that row j
        of q_T is the quantized version of column j of matrix.

        Returns:
            q_T    (r, M) int8  — quantized columns, transposed
            absmax (r,)   fp32  — per-column scale factors
        """
        _, r = matrix.shape
        q_cols, absmax_list = [], []
        for j in range(r):
            q, (absmax, _) = bnb.functional.quantize(
                matrix[:, j].contiguous(), code=code
            )
            q_cols.append(q)
            absmax_list.append(absmax)
        return torch.stack(q_cols, dim=0), torch.stack(absmax_list, dim=0)

    @staticmethod
    def _dequantize_cols(
        q_T: torch.Tensor,     # (r, M) int8
        absmax: torch.Tensor,  # (r,)   fp16/fp32
        code: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dequantize column-wise-quantized matrix back to (M, r) fp32.

        Mirrors the DOBI pattern:
            dq = dequantize_no_absmax(q_T, code)  # (r, M), values in [-1, 1]
            dq = dq * absmax[:, None]              # rescale each row by its absmax
            return dq.T                            # (M, r)
        """
        dq = bnb.functional.dequantize_no_absmax(q_T, code=code)  # (r, M) fp32
        dq = dq * absmax.float().unsqueeze(1)                      # (r, M) rescaled
        return dq.t().contiguous()                                 # (M, r)

    def _get_quantized_parts(self):
        """
        Quantize U and V using the asymmetric K = min(M, N) scheme.
        Returns raw int8 buffers and metadata, reused by save() and apply_quantization().

        Returns:
            UK_q      (r, K) int8   — quantized first K rows of U
            UK_absmax (r,)   fp32   — per-column scales for UK
            VK_q      (r, K) int8   — quantized first K rows of V
            VK_absmax (r,)   fp32   — per-column scales for VK
            rem       (|M-N|, r) fp16 — fp16 remainder (U[K:] if swapped, else V[K:])
            K         int
            swapped   bool          — True if remainder comes from U (M > N)
            code      Tensor        — dynamic map on CUDA
        """
        quant_device = torch.device('cuda') if torch.cuda.is_available() else None
        assert quant_device is not None, "Quantization requires a CUDA device"
        code = bnb.functional.create_dynamic_map().to(quant_device)

        M, N = self.out_features, self.in_features
        K = min(M, N)
        swapped = M > N

        U = self.U.weight.data.float().to(quant_device)      # (M, r)
        V = self.V.weight.data.t().float().to(quant_device)  # (N, r)  [V.weight is (r, N)]

        UK_q, UK_absmax = self._quantize_cols(U[:K], code)   # (r, K) int8, (r,) f32
        VK_q, VK_absmax = self._quantize_cols(V[:K], code)   # (r, K) int8, (r,) f32
        rem = (U[K:] if swapped else V[K:]).to(torch.float16) # (|M-N|, r) fp16

        return UK_q, UK_absmax, VK_q, VK_absmax, rem, K, swapped, code

    # ------------------------------------------------------------------
    # Simulate quantization noise in-place
    # ------------------------------------------------------------------

    def apply_quantization(self):
        """
        Quantize then dequantize U and V in-place, simulating the effect of
        quantized storage without saving/loading. Useful for evaluating
        quantization noise on model quality.
        """
        UK_q, UK_absmax, VK_q, VK_absmax, rem, K, swapped, code = self._get_quantized_parts()

        UK_dq = self._dequantize_cols(UK_q, UK_absmax, code)  # (K, r) fp32
        VK_dq = self._dequantize_cols(VK_q, VK_absmax, code)  # (K, r) fp32

        if swapped:
            U_new = torch.cat([UK_dq, rem.float()], dim=0)  # (M, r)
            V_new = VK_dq                                    # (N, r) = (K, r)
        else:
            U_new = UK_dq                                    # (M, r) = (K, r)
            V_new = torch.cat([VK_dq, rem.float()], dim=0)  # (N, r)

        self.U.weight.data = U_new.to(self.U.weight.dtype).to(self.U.weight.device)
        self.V.weight.data = V_new.t().to(self.V.weight.dtype).to(self.V.weight.device)
        logger.info("Applied quantization noise to U and V in-place")
    # ------------------------------------------------------------------
    # Persistence — overrides parent
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Quantize weights to int8 and save.

        Uses K = min(out_features, in_features) as the split point:
            - U[:K]  (K, r): 8-bit quantized
            - V[:K]  (K, r): 8-bit quantized
            - remainder: U[K:] if out_features > in_features, else V[K:], stored as fp16
        A 'swapped' flag records which matrix holds the fp16 remainder.
        """
        os.makedirs(path, exist_ok=True)

        UK_q, UK_absmax, VK_q, VK_absmax, rem, K, swapped, _ = self._get_quantized_parts()

        self.save_tensor(UK_q,                         path, 'UK_q')
        self.save_tensor(UK_absmax.to(torch.float16),  path, 'UK_absmax')
        self.save_tensor(VK_q,                         path, 'VK_q')
        self.save_tensor(VK_absmax.to(torch.float16),  path, 'VK_absmax')
        self.save_tensor(rem,                          path, 'rem')
        self.save_tensor(torch.tensor(swapped),        path, 'swapped')
        if self.U.bias is not None:
            self.save_tensor(self.U.bias.data, path, 'bias')
        print(f"QuantizedCompressedLinear saved to {path}")

    @classmethod
    def _load(cls, path):
        """Load quantized tensors, dequantize to fp16, return (U, S, V, bias).

        If quantized files (UK_q / VK_q) are not found, falls back to loading
        plain U / V tensors (as saved by CompressedLinear.save). The module is
        constructed from those fp16 weights without any dequantization step;
        quantization is applied only on the next save() or apply_quantization().
        """
        UK_q = cls.load_tensor(path, 'UK_q')
        VK_q = cls.load_tensor(path, 'VK_q')

        if UK_q is None or VK_q is None:
            # Fall back to plain U / V saved by CompressedLinear
            U, S, V = CompressedLinear._load(path)
            bias = cls.load_tensor(path, 'bias')
            print(f"QuantizedCompressedLinear: quantized files not found at {path}, loaded plain U/V tensors")
            return U, S, V, bias

        quant_device = torch.device('cuda') if torch.cuda.is_available() else None
        assert quant_device is not None, "QuantizedCompressedLinear._load requires a CUDA device"
        code = bnb.functional.create_dynamic_map().to(quant_device)

        UK_absmax = cls.load_tensor(path, 'UK_absmax')
        VK_absmax = cls.load_tensor(path, 'VK_absmax')
        rem       = cls.load_tensor(path, 'rem')
        swapped   = cls.load_tensor(path, 'swapped')

        UK = cls._dequantize_cols(UK_q.to(quant_device), UK_absmax.to(quant_device), code).to(torch.float16)  # (K, r)
        VK = cls._dequantize_cols(VK_q.to(quant_device), VK_absmax.to(quant_device), code).to(torch.float16)  # (K, r)

        if swapped.item():
            U = torch.cat([UK, rem.to(torch.float16)], dim=0)  # (M, r), M > N
            V = VK                                               # (N, r) = (K, r)
        else:
            U = UK                                               # (M, r) = (K, r)
            V = torch.cat([VK, rem.to(torch.float16)], dim=0)  # (N, r), N > M

        bias = cls.load_tensor(path, 'bias')
        print(f"QuantizedCompressedLinear loaded from {path}")
        return U, None, V, bias

    @classmethod
    def from_path(cls, path, bias=None):
        is_fallback = cls.load_tensor(path, 'UK_q') is None
        U, _, V, saved_bias = cls._load(path)
        if bias is None:
            bias = saved_bias
        module = cls.from_linear(U, None, V, bias=bias)
        if is_fallback:
            module.apply_quantization()
        return module

    @classmethod
    def from_linear(cls, U, S, V, bias=None):
        in_features  = V.shape[0]
        out_features = U.shape[0]
        module = cls(
            in_features, out_features, rank=U.shape[1],
            U=U, S=S, V=V,
            bias=(bias is not None),
        )
        if bias is not None:
            module.U.bias.data = bias
        return module

    def _get_compression_ratio(self) -> float:
        """Storage compression ratio relative to an uncompressed fp16 linear layer.

        All counts expressed in bytes:
            original  : (M*N + bias_p) * 2       fp16
            UK_q      :  r*K                      int8
            VK_q      :  r*K                      int8
            rem       :  |M-N|*r * 2              fp16
            bias      :  bias_p * 2               fp16
        """
        M, N, r = self.out_features, self.in_features, self.rank
        K = min(M, N)
        bias_p = M if self.U.bias is not None else 0
        original_bytes = (M * N + bias_p) * 2
        quant_bytes    = 2 * r * K + abs(M - N) * r * 2 + bias_p * 2
        return quant_bytes / original_bytes


class QuantizedCompressedLinearBnB(CompressedLinear):
    """
    Same asymmetric quantization scheme as QuantizedCompressedLinear, but uses
    bnb.nn.Linear8bitLt for U and V[:M] so the forward pass runs a fused int8
    CUDA matmul — no explicit dequantization step.

    For W = U @ V.T, U: (M, r), V: (N, r), with M = out_features <= N = in_features:

        - U        (M, r): stored in Linear8bitLt — row-wise int8, M scales of r elements
        - V[:M]    (M, r): stored in Linear8bitLt — row-wise int8, r scales of M elements
        - V[M:]  (N-M, r): stored as fp16 nn.Linear

    Granularity comparison vs QuantizedCompressedLinear (per-column):
        U:  Linear8bitLt gives M scales (one per output neuron) — finer than r col scales
        V1: both give r scales (equivalent)

    Forward (no dequantization):
        h   = V1_lin(x[..., :M]) + V2_lin(x[..., M:])   # (..., r)
        out = U_lin(h)                                    # (..., M)

    Note: requires CUDA. get_recon_weight() is accurate before first forward;
    after first forward, fp16 weights are freed and CB/SCB int8 state is used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = None,
        U: torch.Tensor = None,
        S: torch.Tensor = None,
        V: torch.Tensor = None,
        bias: bool = True,
    ):
        nn.Module.__init__(self)

        assert U is not None and V is not None, "U and V must be provided"
        assert out_features <= in_features, (
            f"QuantizedCompressedLinearBnB requires out_features ({out_features}) "
            f"<= in_features ({in_features}). Swap U/V if needed."
        )

        M, N = out_features, in_features
        if rank is None:
            rank = U.shape[1]
        self.in_features = N
        self.out_features = M
        self.rank = rank
        self.W = None

        if S is not None:
            sqrtS = torch.sqrt(S.float())
            U = U.float() @ torch.diag(sqrtS)
            V = V.float() @ torch.diag(sqrtS)

        # U: Linear8bitLt(rank → M).  weight shape = (M, rank) = U.
        # Copy into the existing Int8Params weight (do NOT replace it with a plain
        # nn.Parameter — that would strip the CB/SCB attributes bnb relies on).
        self.U_lin = bnb.nn.Linear8bitLt(rank, M, bias=bias, has_fp16_weights=True, threshold=6.0)
        self.U_lin.weight.data = U.to(torch.float16)
        if bias:
            self.U_lin.bias = nn.Parameter(torch.zeros(M))

        # V[:M]: Linear8bitLt(M → rank).  weight shape = (rank, M) = V[:M].T.
        self.V1_lin = bnb.nn.Linear8bitLt(M, rank, bias=False, has_fp16_weights=True, threshold=6.0)
        self.V1_lin.weight.data = V[:M].t().to(torch.float16)

        # V[M:]: fp16 nn.Linear(N-M → rank).  weight = (rank, N-M) = V[M:].T.
        self.V2_lin = nn.Linear(N - M, rank, bias=False)
        self.V2_lin.weight = nn.Parameter(V[M:].t().to(torch.float16), requires_grad=False)

    # ------------------------------------------------------------------
    # Core interface — overrides parent
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split x along the feature dim to route through the two V sub-layers
        h = self.V1_lin(x[..., :self.out_features]) + self.V2_lin(x[..., self.out_features:])
        return self.U_lin(h)

    def get_recon_weight(self) -> torch.Tensor:
        """
        Returns reconstructed W = U @ V.T (M, N) in fp32.
        Reads from fp16 weight parameters — accurate before the first forward call.
        After first forward, fp16 weights are freed by bnb; call this beforehand if needed.
        """
        U  = self.U_lin.weight.float()        # (M, r)
        V1 = self.V1_lin.weight.t().float()   # (M, r)  — weight is stored as (r, M)
        V2 = self.V2_lin.weight.t().float()   # (N-M, r)
        V  = torch.cat([V1, V2], dim=0)       # (N, r)
        return U @ V.t()                      # (M, N)

    def _get_compression_ratio(self) -> float:
        M, N, r = self.out_features, self.in_features, self.rank
        bias_p = M if self.U_lin.bias is not None else 0
        original_params = M * N + bias_p
        # U_lin:  M*r int8  + M fp32 row-scales  → (M*r + M*4) / 4 fp32-equiv
        # V1_lin: r*M int8  + r fp32 row-scales  → (r*M + r*4) / 4 fp32-equiv
        # V2_lin: (N-M)*r fp16                   → (N-M)*r / 2 fp32-equiv
        quant_fp32_equiv = (M*r + M*4)/4 + (r*M + r*4)/4 + (N-M)*r/2 + bias_p
        return quant_fp32_equiv / original_params

    # ------------------------------------------------------------------
    # Persistence — overrides parent
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.save_tensor(self.U_lin.weight.data,   path, 'U')
        self.save_tensor(self.V1_lin.weight.data,  path, 'V1_T')
        self.save_tensor(self.V2_lin.weight.data,  path, 'V2_T')
        if self.U_lin.bias is not None:
            self.save_tensor(self.U_lin.bias.data, path, 'bias')
        print(f"QuantizedCompressedLinearBnB saved to {path}")

    @classmethod
    def from_linear(cls, U, S, V, bias=None):
        in_features  = V.shape[0]
        out_features = U.shape[0]
        module = cls(
            in_features, out_features,
            U=U, S=S, V=V,
            bias=(bias is not None),
        )
        if bias is not None:
            module.U_lin.bias.data = bias
        return module


class GPTQLinear(nn.Module):
    """
    A linear layer with weights quantized via GPTQ (Frantar et al., 2022).

    Weights are quantized to `bits` (4 or 8) with per-group asymmetric quantization
    and stored as packed uint8 integers. Scales and integer zero points are stored
    per group (fp16). The forward pass dequantizes on the fly.

    Quantization convention (asymmetric):
        q   = clamp(round(w / scale + zero), 0, 2^bits - 1)   [integer]
        w'  = (q - zero) * scale                               [dequantized]
    where scale and zero are per-group, derived from the min/max of each group.

    Storage layout for 4-bit:
        qweight: (out_features, in_features // 2), uint8
                 low nibble  = column 2k
                 high nibble = column 2k+1
    Storage layout for 8-bit:
        qweight: (out_features, in_features), uint8
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        group_size: int,
        bias: bool = True,
    ):
        super().__init__()
        assert bits in (4, 8), f"bits must be 4 or 8, got {bits}"
        assert in_features % group_size == 0, (
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
        if bits == 4:
            assert in_features % 2 == 0, "in_features must be even for 4-bit packing"

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.maxq = (1 << bits) - 1

        num_groups = in_features // group_size

        if bits == 4:
            self.register_buffer(
                "qweight",
                torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
            )
        else:
            self.register_buffer(
                "qweight",
                torch.zeros(out_features, in_features, dtype=torch.uint8),
            )

        # Per-group quantization parameters (fp16 for storage efficiency)
        self.register_buffer("scales", torch.zeros(out_features, num_groups, dtype=torch.float16))
        self.register_buffer("zeros", torch.zeros(out_features, num_groups, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_buffer("bias", None)

    # ------------------------------------------------------------------
    # Packing / unpacking
    # ------------------------------------------------------------------

    def pack_weights(self, W_int: torch.Tensor):
        """
        Pack integer weight tensor W_int (out_features, in_features) with values
        in [0, maxq] into the qweight buffer.
        """
        W_uint8 = W_int.to(torch.uint8)
        if self.bits == 4:
            packed = (W_uint8[:, 0::2] & 0xF) | ((W_uint8[:, 1::2] & 0xF) << 4)
            self.qweight.copy_(packed)
        else:
            self.qweight.copy_(W_uint8)

    def unpack_weights(self) -> torch.Tensor:
        """
        Unpack qweight to (out_features, in_features) int32 tensor with values in [0, maxq].
        """
        if self.bits == 4:
            W_int = torch.empty(
                self.out_features, self.in_features,
                dtype=torch.int32, device=self.qweight.device,
            )
            W_int[:, 0::2] = (self.qweight & 0xF).to(torch.int32)
            W_int[:, 1::2] = ((self.qweight >> 4) & 0xF).to(torch.int32)
        else:
            W_int = self.qweight.to(torch.int32)
        return W_int

    # ------------------------------------------------------------------
    # Dequantization
    # ------------------------------------------------------------------

    def dequantize(self) -> torch.Tensor:
        """
        Dequantize weights to fp16: W = (q - zero) * scale.
        Returns (out_features, in_features) fp16 tensor.
        """
        W_int = self.unpack_weights().float()
        # Broadcast (out_features, num_groups) → (out_features, in_features)
        scales = self.scales.float().repeat_interleave(self.group_size, dim=1)
        zeros  = self.zeros.float().repeat_interleave(self.group_size, dim=1)
        return ((W_int - zeros) * scales).to(self.scales.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.dequantize()
        bias = self.bias.to(W.dtype) if self.bias is not None else None
        return F.linear(x, W, bias)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _get_compression_ratio(self) -> float:
        """Storage compression ratio relative to an uncompressed fp16 linear layer."""
        M, N = self.out_features, self.in_features
        num_groups = N // self.group_size
        bias_bytes = M * 2 if self.bias is not None else 0
        original_bytes = M * N * 2 + bias_bytes
        quant_bytes = M * N * self.bits // 8        # packed integer weights
        quant_bytes += M * num_groups * 2 * 2       # scales + zeros, fp16
        quant_bytes += bias_bytes
        return quant_bytes / original_bytes

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_quantized(
        cls,
        W_int: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        bits: int,
        group_size: int,
        bias: torch.Tensor = None,
    ) -> "GPTQLinear":
        """
        Construct a GPTQLinear from pre-quantized integer weights and quantization parameters.

        Args:
            W_int:      (out_features, in_features) int32 tensor, values in [0, 2^bits - 1]
            scales:     (out_features, num_groups) float32 scale per group
            zeros:      (out_features, num_groups) float32 integer zero point per group
            bits:       quantization bit-width (4 or 8)
            group_size: columns per quantization group
            bias:       optional (out_features,) bias tensor
        """
        out_features, in_features = W_int.shape
        module = cls(
            in_features, out_features,
            bits=bits, group_size=group_size,
            bias=(bias is not None),
        )
        module.pack_weights(W_int)
        module.scales.copy_(scales.to(torch.float16))
        module.zeros.copy_(zeros.to(torch.float16))
        if bias is not None:
            module.bias.copy_(bias.to(torch.float16))
        return module

