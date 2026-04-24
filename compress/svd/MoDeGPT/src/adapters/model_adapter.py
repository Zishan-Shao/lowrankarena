import torch
import os

from torch import Tensor
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass
import logging

from transformers.configuration_utils import PretrainedConfig

from src.adapters.CompressionConfig import CompressionConfig
from src.model_utils import dtype_p, calib_device

from datetime import datetime


def _resolve_metrics_paths(
    path: str = "./metrics/metrics.json",
    backup_dir: str = "./metrics/backups/",
    jsons_path: str = "./metrics/jsons/",
):
    metrics_root = os.environ.get("MODEGPT_METRICS_DIR", "").strip()
    if metrics_root:
        path = os.path.join(metrics_root, "metrics.json")
        backup_dir = os.path.join(metrics_root, "backups")
        jsons_path = os.path.join(metrics_root, "jsons")

    return path, backup_dir, jsons_path


@dataclass
class MLPTensors:
    up_proj: Tensor
    down_proj: Tensor
    gate_proj: Tensor

    def to(self, dtype):
        self.up_proj = self.up_proj.to(dtype=dtype)
        self.down_proj = self.down_proj.to(dtype=dtype)
        self.gate_proj = self.gate_proj.to(dtype=dtype)
        return self


@dataclass
class VOTensors:
    v_proj: Tensor
    o_proj: Tensor

    def to(self, dtype):
        self.v_proj = self.v_proj.to(dtype=dtype)
        self.o_proj = self.o_proj.to(dtype=dtype)
        return self


@dataclass
class QKTensors:
    query_proj: Tensor
    key_proj: Tensor

    def to(self, dtype):
        self.query_proj = self.query_proj.to(dtype=dtype)
        self.key_proj = self.key_proj.to(dtype=dtype)
        return self


@dataclass
class MLPComponents:
    block: nn.Module | None
    up_proj: nn.Module | nn.Linear
    down_proj: nn.Module | nn.Linear
    gate_proj: nn.Module | nn.Linear | None = None


@dataclass
class QKComponents:
    block: nn.Module | None
    query_proj: nn.Module | nn.Linear
    key_proj: nn.Module | nn.Linear


@dataclass
class VOComponents:
    block: nn.Module | None
    v_proj: nn.Module | nn.Linear
    o_proj: nn.Module | nn.Linear


@dataclass
class AttentionComponents:
    block: nn.Module
    q_proj: nn.Module
    k_proj: nn.Module
    v_proj: Optional[nn.Module] = None
    o_proj: Optional[nn.Module] = None


def build_metrics(_all_metrics: dict):
    run_name = datetime.now().strftime("%Y_%m_%d--%H_%M_%S")
    metrics = {
        "RunName": run_name,
        "RunDate": datetime.now().strftime("%b %d, %Y %I:%M %p"),
        "latent_moe_metrics": {},
    }
    _all_metrics[run_name] = metrics

    return metrics


class ModelAdapter(ABC):
    _metrics: dict

    _metrics = {}

    def __init__(self, model: nn.Module, tokenizer=None):
        self.model = model
        self.model_config: PretrainedConfig = model.config
        self.config: CompressionConfig = CompressionConfig()
        self.tokenizer = tokenizer
        self.calibs = None

        ModelAdapter.load_metrics()
        self.metrics = build_metrics(ModelAdapter._metrics)

        # self.metrics = {}
        # run_name = datetime.now().strftime("%Y_%m_%d--%H_%M_%S")

        # ModelAdapter._metrics[run_name] = self.metrics
        # self.metrics["RunDate"] = datetime.now().strftime("%b %d, %Y %I:%M %p")

    @staticmethod
    def from_model(model: nn.Module, tokenizer) -> "ModelAdapter":
        from .LlamaAdapter import LlamaAdapter
        from .OPTAdapter import OPTAdapter
        from .QwenAdapter import QwenAdapter

        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            return OPTAdapter(model, tokenizer=tokenizer)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # Could be Llama or Mixtral or others sharing this structure
            # Check config type or architecture string if available
            arch = getattr(model.config, "model_type", None)
            if "qwen3" in arch:
                return QwenAdapter(model, tokenizer=tokenizer)

            return LlamaAdapter(model, tokenizer=tokenizer)
        else:
            raise RuntimeError("Unsupported model architecture")

    @staticmethod
    def load_metrics(path="./metrics/metrics.json"):
        path, _, _ = _resolve_metrics_paths(path=path)
        if not ModelAdapter._metrics.keys() and os.path.exists(path):
            with open(path, "r") as f:
                import json

                ModelAdapter._metrics = json.load(f)

    @staticmethod
    def save_metrics_static(
        path: str = "./metrics/metrics.json",
        backup_dir: str = "./metrics/backups/",
        jsons_path="./metrics/jsons/",
        run_metrics: dict | None = None,
    ):
        import json
        import shutil

        path, backup_dir, jsons_path = _resolve_metrics_paths(
            path=path,
            backup_dir=backup_dir,
            jsons_path=jsons_path,
        )
        timestamp = datetime.now().strftime("%Y_%m_%d--%H_%M_%S")

        os.makedirs(backup_dir, exist_ok=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(ModelAdapter._metrics, f, indent=4)

        if run_metrics:
            os.makedirs(jsons_path, exist_ok=True)

            filename_date = run_metrics["RunName"]
            filename_note = (
                ""
                if "note" not in run_metrics or not run_metrics["note"]
                else run_metrics["note"][:15]
            )

            run_metrics_path = os.path.join(jsons_path, f"{filename_date}--{filename_note}.json")

            with open(run_metrics_path, "w") as f:
                json.dump(run_metrics, f, indent=4)

    def save_metrics(
        self,
        path: str = "./metrics/metrics.json",
        backup_dir: str = "./metrics/backups/",
    ):
        ModelAdapter.save_metrics_static(path=path, backup_dir=backup_dir, run_metrics=self.metrics)

    def save_layer(self, output_dir: str, suffix: str, weights: dict[str, Tensor], layer_idx):
        import os

        output_dir = os.path.expandvars(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        p = os.path.join(output_dir, f"layer_{layer_idx}_{suffix}")
        torch.save(weights, p)

    @torch.no_grad()
    def convert_model(
        self, saved_layers_dir: str = "./compressed_output/layers/", suffixes=["mlp", "qk", "vo"]
    ):
        saved_layers_dir = os.path.expandvars(saved_layers_dir)

        def tensor_to_linear(weight: Tensor) -> nn.Linear:
            linear = nn.Linear(
                in_features=weight.shape[1],
                out_features=weight.shape[0],
                device="cuda",
                dtype=torch.bfloat16,
                bias=False,
            )
            linear.weight.data.copy_(weight.to(torch.bfloat16))
            return linear

        for suffix in suffixes:
            for layer_idx in range(self.n_layers):
                path = os.path.join(saved_layers_dir, f"layer_{layer_idx}_{suffix}")
                compressed = torch.load(path, map_location="cuda:0")

                if suffix == "mlp":
                    self.replace_mlp_layers(
                        layer_idx,
                        new_up=tensor_to_linear(compressed["up"]),
                        new_down=tensor_to_linear(compressed["down"]),
                        new_gate=tensor_to_linear(compressed["gate"]),
                    )
                elif suffix == "qk":
                    self.replace_attn_layers(
                        layer_idx,
                        new_q=tensor_to_linear(compressed["q_proj"]),
                        new_k=tensor_to_linear(compressed["k_proj"]),
                        new_v=None,
                        new_o=None,
                    )
                elif suffix == "vo":
                    self.replace_attn_layers(
                        layer_idx,
                        new_q=None,
                        new_k=None,
                        new_v=tensor_to_linear(compressed["v_proj"]),
                        new_o=tensor_to_linear(compressed["o_proj"]),
                    )

    @property
    def n_kv_heads(self):
        getattr()
        if hasattr(self.model.config, "num_key_value_heads"):
            return self.model.config, "num_key_value_heads"
        elif hasattr(self.model.config, "num_key_value_heads"):
            return self.model.config, "num_attention_heads"
        else:
            raise NotImplementedError("Need to add the arch specific key for num_kv_heads")

    @abstractmethod
    def compute_layer_energy(self, layer_idx: int, Ca: Tensor | None = None) -> MLPTensors:
        pass

    @property
    def arch(self) -> str:
        return self.model_config.model_type

    @property
    def n_layers(self) -> int:
        return (
            getattr(self.model_config, "n_layer", None)
            or getattr(self.model_config, "num_hidden_layers", None)
            or getattr(self.model_config, "num_layers", None)
        )

    @property
    def n_heads(self) -> int:
        return getattr(self.model_config, "n_head", None) or getattr(
            self.model_config, "num_attention_heads", None
        )

    @property
    def d_model(self) -> int:
        return getattr(self.model_config, "hidden_size", None) or getattr(
            self.model_config, "dim", None
        )

    @property
    def d_int(self) -> int:
        return getattr(self.model_config, "intermediate_size", None)

    @property
    def head_dim(self) -> int:
        return self.model_config.head_dim
        # return self.d_model // self.n_heads

    @property
    def n_experts(self) -> int:
        if self.arch == "deepseek":
            return getattr(self.model_config, "n_routed_experts", 0)
        return getattr(self.model_config, "num_local_experts", 0)

    @property
    def n_kv_heads(self) -> int:
        return getattr(self.model_config, "num_key_value_heads", self.n_heads)

    @abstractmethod
    def calibrate_model(
        self, n_samples: int, batch_size: int, target_layers: list[int], dataset="wikitext"
    ):
        raise NotImplementedError

    @abstractmethod
    def get_transformer_blocks(self) -> nn.ModuleList:
        pass

    def get_n_inner(self) -> int:
        return self.model_config.intermediate_size

    @abstractmethod
    def register_hooks(
        self,
        layer_idx: int,
        block: nn.Module,
        cov_mlp_list: List[torch.Tensor],
        cov_q_list: List[List[torch.Tensor]],
        cov_k_list: List[List[torch.Tensor]],
        cov_x_list: List[torch.Tensor],
        handles: List[Any],
        logger: logging.Logger,
    ):
        pass

    def on_batch_end_step(self, layer_idx: int, x_in: torch.Tensor, cov_x_list: List[torch.Tensor]):
        pass

    @abstractmethod
    def get_mlp_components(self, layer_idx: int, expert_idx: int | None = None) -> MLPComponents:
        pass

    @abstractmethod
    def get_mlp_tensors(self, layer_idx: int, expert_idx: int | None = None) -> MLPComponents:
        pass

    @abstractmethod
    def get_vo_components(self, layer_idx: int, expert_idx: int | None = None) -> VOComponents:
        pass

    @abstractmethod
    def get_vo_tensors(self, layer_idx: int, expert_idx: int | None = None) -> VOTensors:
        pass

    @abstractmethod
    def get_qk_components(self, layer_idx: int, expert_idx: int | None = None) -> QKComponents:
        pass

    @abstractmethod
    def get_qk_tensors(self, layer_idx: int, expert_idx: int | None = None) -> QKTensors:
        pass

    @abstractmethod
    def replace_mlp_layers(
        self,
        layer_idx: int,
        new_up: nn.Module,
        new_down: nn.Module,
        new_gate: Optional[nn.Module] = None,
        expert_idx: Optional[int] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_attn_components(self, layer_idx: int) -> AttentionComponents:
        pass

    @abstractmethod
    def replace_attn_layers(
        self,
        layer_idx: int,
        new_q: Optional[nn.Module],
        new_k: Optional[nn.Module],
        new_v: Optional[nn.Module],
        new_o: Optional[nn.Module],
    ) -> None:
        pass

    @abstractmethod
    def get_qk_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q and K projection weights for a layer.

        Returns:
            Tuple of (W_q, W_k) weight tensors
        """
        pass

    @abstractmethod
    def get_vo_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get V and O projection weights for a layer.

        Returns:
            Tuple of (W_v, W_o) weight tensors
        """
        pass

    def slice_gate_dims(
        self,
        layer_idx: int,
        up_weights: torch.Tensor,
        down_weights: torch.Tensor,
        gate_weights: Optional[torch.Tensor],
        new_bias_u: Optional[torch.Tensor],
        new_bias_g: Optional[torch.Tensor],
        bias: bool = True,
        expert_idx: Optional[int] = None,
    ):
        comps = self.get_mlp_components(layer_idx, expert_idx=expert_idx)

        # up_weights: [out, in]
        new_layer_U = nn.Linear(
            in_features=up_weights.shape[1],
            out_features=up_weights.shape[0],
            device="cuda",
            dtype=torch.bfloat16,
            bias=True if comps.up_proj.bias is not None and bias else False,
        )
        new_layer_U.weight.data.copy_(up_weights.to(torch.bfloat16))
        if bias and new_bias_u is not None:
            if new_layer_U.bias is not None:
                new_layer_U.bias.data.copy_(new_bias_u)

        new_layer_G = None
        if gate_weights is not None:
            # If we have gate weights, we expect the component to support it (like llama)
            # or we are adding it? Original code only did this if arch == llama.
            # But gate_weights is None for other arches in caller usually.

            gate_has_bias = False
            if comps.gate_proj is not None and comps.gate_proj.bias is not None:
                gate_has_bias = True

            new_layer_G = nn.Linear(
                in_features=gate_weights.shape[1],
                out_features=gate_weights.shape[0],
                device="cuda",
                dtype=torch.bfloat16,
                bias=True if gate_has_bias and bias else False,
            )
            new_layer_G.weight.data.copy_(gate_weights.to(torch.bfloat16))
            if bias and new_bias_g is not None:
                if new_layer_G.bias is not None:
                    new_layer_G.bias.data.copy_(new_bias_g)

        new_layer_D = nn.Linear(
            in_features=down_weights.shape[1],
            out_features=down_weights.shape[0],
            device="cuda",
            dtype=torch.bfloat16,
            bias=True if comps.down_proj.bias is not None and bias else False,
        )
        new_layer_D.weight.data.copy_(down_weights.to(torch.bfloat16))
        if comps.down_proj.bias is not None and bias:
            if new_layer_D.bias is not None:
                new_layer_D.bias.data.copy_(comps.down_proj.bias.data)

        self.replace_mlp_layers(
            layer_idx, new_layer_U, new_layer_D, new_layer_G, expert_idx=expert_idx
        )

    def slice_qk_dims(
        self,
        layer_idx: int,
        new_heads_Q: List[torch.Tensor],
        new_heads_K: List[torch.Tensor],
        new_bias_Q: List[torch.Tensor] = [],
        new_bias_K: List[torch.Tensor] = [],
        bias: bool = True,
    ):
        comps = self.get_attn_components(layer_idx)

        Q_heads = torch.cat(new_heads_Q, dim=0).to(device="cuda", dtype=torch.bfloat16)
        K_heads = torch.cat(new_heads_K, dim=0).to(device="cuda", dtype=torch.bfloat16)

        bias_Q = None
        if len(new_bias_Q) > 0:
            bias_Q = torch.cat(new_bias_Q, dim=0).to(device="cuda", dtype=torch.fbloat16)

        bias_K = None
        if len(new_bias_K) > 0:
            bias_K = torch.cat(new_bias_K, dim=0).to(device="cuda", dtype=torch.bfloat16)

        new_layer_Q = nn.Linear(
            in_features=Q_heads.shape[1],
            out_features=Q_heads.shape[0],
            device="cuda",
            dtype=torch.bfloat16,
            bias=comps.q_proj.bias is not None and bias and (bias_Q is not None),
        )
        new_layer_Q.weight.data.copy_(Q_heads)
        if new_layer_Q.bias is not None and bias_Q is not None:
            new_layer_Q.bias.data.copy_(bias_Q)

        new_layer_K = nn.Linear(
            in_features=K_heads.shape[1],
            out_features=K_heads.shape[0],
            device="cuda",
            dtype=torch.bfloat16,
            bias=comps.k_proj.bias is not None and bias and (bias_K is not None),
        )
        new_layer_K.weight.data.copy_(K_heads)
        if new_layer_K.bias is not None and bias_K is not None:
            new_layer_K.bias.data.copy_(bias_K)

        self.replace_attn_layers(
            layer_idx, new_q=new_layer_Q, new_k=new_layer_K, new_v=None, new_o=None
        )

    def slice_vo_dims(
        self,
        layer_idx: int,
        new_heads_V: List[torch.Tensor],
        new_heads_O: List[torch.Tensor],
        bias: bool,
    ):
        comps = self.get_attn_components(layer_idx)
        original_o = comps.o_proj
        original_v = comps.v_proj

        V_heads = torch.cat(new_heads_V, dim=0).to(device="cuda", dtype=torch.bfloat16)
        O_heads = torch.cat(new_heads_O, dim=1).to(device="cuda", dtype=torch.bfloat16)

        new_layer_V = nn.Linear(
            in_features=V_heads.shape[1],
            out_features=V_heads.shape[0],
            device="cuda",
            dtype=torch.bfloat16,
            bias=False,  # As per original implementation, bias=False for V
        )
        new_layer_V.weight.data.copy_(V_heads)

        new_layer_O = nn.Linear(
            in_features=O_heads.shape[1],
            out_features=O_heads.shape[0],
            device="cuda",
            dtype=torch.bfloat16,
            bias=True if original_o.bias is not None and bias else False,
        )
        new_layer_O.weight.data.copy_(O_heads)
        if original_o.bias is not None and bias and new_layer_O.bias is not None:
            new_layer_O.bias.data.copy_(original_o.bias.data)

        self.replace_attn_layers(
            layer_idx, new_q=None, new_k=None, new_v=new_layer_V, new_o=new_layer_O
        )

    # -- Helper hooks --

    @staticmethod
    def _make_fc_hook(layer_idx, cov_mlp_list):
        @torch.no_grad()
        def hook(module, inp, out):
            act = torch.nn.functional.relu(out.to(dtype=dtype_p, device=calib_device))
            H = act.detach().to(dtype=dtype_p).view(-1, act.size(-1))
            cov_mlp_list[layer_idx] += (H.T @ H).to(device=calib_device)

        return hook

    @staticmethod
    def _make_proj_hook(layer_idx, cov_list, n_heads, head_dim, d_model):
        @torch.no_grad()
        def hook(module, inp, out):
            proj_out = out.detach().to(dtype=dtype_p, device="cuda")
            proj = proj_out.view(-1, n_heads, head_dim)
            proj = proj.permute(1, 0, 2)  # [n_heads, B*T, head_dim]
            C_proj = torch.bmm(proj.transpose(1, 2), proj).to(calib_device)
            for h in range(n_heads):
                cov_list[layer_idx][h] += C_proj[h, :, :]

        return hook
