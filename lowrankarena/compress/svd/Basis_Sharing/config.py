import argparse
import re
from typing import Any

import yaml


def add_args():
    paser = argparse.ArgumentParser(description="ShareModel")
    paser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    paser.add_argument(
        "--calibration_size",
        "--cs",
        help="calibration size",
        type=int,
        default=256,
    )
    paser.add_argument(
        "--dataset_name",
        help="dataset for load",
        type=str,
        default="wikitext",
    )
    paser.add_argument(
        "--dataset_cache_dir",
        help="change dataset cache dir",
        type=str,
        default=None,
    )
    paser.add_argument(
        "--gpu_guard",
        action="store_true",
        help="Reserve most free CUDA memory while the model is offloaded to CPU between calibration/decomposition phases.",
    )
    paser.add_argument(
        "--gpu_guard_keep_free_gib",
        type=float,
        default=60.0,
        help="When gpu_guard is enabled, leave at least this many GiB free for the next GPU phase.",
    )
    paser.add_argument(
        "--gpu_guard_reserve_fraction",
        type=float,
        default=0.95,
        help="Fraction of currently free memory to reserve beyond gpu_guard_keep_free_gib.",
    )
    paser.add_argument(
        "--gpu_guard_chunk_mib",
        type=int,
        default=256,
        help="Chunk size in MiB for the guard allocator.",
    )
    args, unknown = paser.parse_known_args()
    return args


class ShareConfig:
    LLAMA_MODEL_TYPES = {"llama", "llama2", "llama3", "llama3.1"}

    name_map = {
        'meta-llama/Llama-2-7b-hf': "llama2-7b",
        'meta-llama/Llama-3.1-8B': "llama3.1-8b",
        "jeffwan/llama-7b-hf": "llama2-7b",
        "jeffwan/llama-13b-hf": "llama2-13b",
        "jeffwan/llama-30b-hf": "llama2-30b",
        'gpt2': "gpt2",
        'facebook/opt-6.7b': 'opt-6.7b',
        "mistralai/Mistral-7B-v0.1": "mistral-7b"
    }

    weight_info = {
        "llama2-7b": {
            "self_attn.k_proj": (4096, 4096),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 4096),
            "self_attn.o_proj": (4096, 4096),
            "mlp.up_proj": (4096, 11008),
            "mlp.gate_proj": (4096, 11008),
            "mlp.down_proj": (11008, 4096),
        },

        "llama2-13b": {
            "self_attn.k_proj": (5120, 5120),
            "self_attn.q_proj": (5120, 5120),
            "self_attn.v_proj": (5120, 5120),
            "self_attn.o_proj": (5120, 5120),
            "mlp.up_proj": (5120, 13824),
            "mlp.gate_proj": (5120, 13824),
            "mlp.down_proj": (13824, 5120),
        },

        "llama2-30b": {
            "self_attn.k_proj": (6656, 6656),
            "self_attn.q_proj": (6656, 6656),
            "self_attn.v_proj": (6656, 6656),
            "self_attn.o_proj": (6656, 6656),
            "mlp.up_proj": (6656, 17920),
            "mlp.gate_proj": (6656, 17920),
            "mlp.down_proj": (17920, 6656),
        },

        "llama3.1-8b": {
            "self_attn.k_proj": (4096, 1024),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 1024),
            "self_attn.o_proj": (4096, 4096),
            "mlp.up_proj": (4096, 14336),
            "mlp.gate_proj": (4096, 14336),
            "mlp.down_proj": (14336, 4096),
        },

        "gpt2": {
            "attn.c_attn": (768, 2304),
            "attn.c_proj": (768, 768),
            "mlp.c_fc": (768, 3072),
            "mlp.c_proj": (3072, 768)
        },

        "opt-6.7b": {
            "self_attn.k_proj": (4096, 4096),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 4096),
            "self_attn.out_proj": (4096, 4096),
            "fc1": (4096, 16384),
            "fc2": (16384, 4096),
        },
        "mistral-7b": {
            "self_attn.k_proj": (4096, 1024),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 1024),
            "self_attn.o_proj": (4096, 4096),
            "mlp.up_proj": (4096, 14336),
            "mlp.gate_proj": (4096, 14336),
            "mlp.down_proj": (14336, 4096),
        },

    }

    @classmethod
    def is_llama_model_type(cls, model_type: Any) -> bool:
        return str(model_type).lower() in cls.LLAMA_MODEL_TYPES

    @staticmethod
    def _sanitize_project_name(value: str) -> str:
        value = value.strip().lower().replace("/", "-")
        value = re.sub(r"[^a-z0-9._-]+", "-", value)
        value = re.sub(r"-{2,}", "-", value).strip("-")
        return value or "basis-sharing"

    @classmethod
    def resolve_project_name(cls, model_name: str, hf_config: Any | None = None) -> str:
        alias = cls.name_map.get(model_name)
        if alias is not None:
            return alias
        if hf_config is not None:
            model_type = getattr(hf_config, "model_type", None)
            hidden_size = getattr(hf_config, "hidden_size", None)
            if model_type is not None:
                suffix = f"-{int(hidden_size)}" if hidden_size is not None else ""
                return cls._sanitize_project_name(f"{model_type}{suffix}")
        return cls._sanitize_project_name(model_name.rsplit("/", 1)[-1])

    @classmethod
    def _infer_weight_info_from_hf_config(cls, hf_config: Any) -> dict[str, tuple[int, int]] | None:
        model_type = str(getattr(hf_config, "model_type", "")).lower()

        if model_type in {"llama", "mistral"}:
            hidden_size = int(getattr(hf_config, "hidden_size"))
            intermediate_size = int(getattr(hf_config, "intermediate_size"))
            num_attention_heads = int(getattr(hf_config, "num_attention_heads"))
            num_key_value_heads = int(getattr(hf_config, "num_key_value_heads", num_attention_heads))
            head_dim = getattr(hf_config, "head_dim", None)
            if head_dim is None:
                head_dim = hidden_size // num_attention_heads
            kv_hidden_size = int(num_key_value_heads) * int(head_dim)
            return {
                "self_attn.k_proj": (hidden_size, kv_hidden_size),
                "self_attn.q_proj": (hidden_size, hidden_size),
                "self_attn.v_proj": (hidden_size, kv_hidden_size),
                "self_attn.o_proj": (hidden_size, hidden_size),
                "mlp.up_proj": (hidden_size, intermediate_size),
                "mlp.gate_proj": (hidden_size, intermediate_size),
                "mlp.down_proj": (intermediate_size, hidden_size),
            }

        if model_type == "opt":
            hidden_size = int(getattr(hf_config, "hidden_size"))
            intermediate_size = int(getattr(hf_config, "ffn_dim"))
            return {
                "self_attn.k_proj": (hidden_size, hidden_size),
                "self_attn.q_proj": (hidden_size, hidden_size),
                "self_attn.v_proj": (hidden_size, hidden_size),
                "self_attn.out_proj": (hidden_size, hidden_size),
                "fc1": (hidden_size, intermediate_size),
                "fc2": (intermediate_size, hidden_size),
            }

        return None

    @classmethod
    def resolve_weight_info(cls, model_name: str, hf_config: Any | None = None) -> dict[str, tuple[int, int]]:
        alias = cls.name_map.get(model_name)
        if alias in cls.weight_info:
            return dict(cls.weight_info[alias])

        if hf_config is not None:
            inferred = cls._infer_weight_info_from_hf_config(hf_config)
            if inferred is not None:
                return inferred

        if alias is not None:
            raise KeyError("Missing Basis Sharing weight metadata for model alias: {}".format(alias))
        raise KeyError(
            "Unable to resolve Basis Sharing weight metadata for model '{}'. "
            "Pass a compatible Hugging Face config or extend ShareConfig.".format(model_name)
        )

    def __init__(self, cmd_args):
        cmd_args_dict = cmd_args.__dict__
        self.configuration = self.load_yaml_config(cmd_args.yaml_config_file)
        self.set_attr_from_config(self.configuration)
        for arg_key, arg_val in cmd_args_dict.items():
            setattr(self, arg_key, arg_val)

    @staticmethod
    def load_yaml_config(yaml_path):
        with open(yaml_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Yaml error - check yaml file")

    def set_attr_from_config(self, configuration):
        for _, param_family in configuration.items():
            for key, val in param_family.items():
                setattr(self, key, val)
