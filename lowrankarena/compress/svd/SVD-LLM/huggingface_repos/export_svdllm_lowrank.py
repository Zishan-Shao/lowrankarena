import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers.activations as activations
from huggingface_hub import save_torch_state_dict
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SVDLLM_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
MODELING_ROOT = REPO_ROOT / "src" / "modeling"

sys.path.insert(0, str(SVDLLM_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))


MODULE_NAME_MAP = {
    "self_attn.q_u_proj": "self_attn.q_proj",
    "self_attn.q_v_proj": "self_attn.q_proj",
    "self_attn.k_u_proj": "self_attn.k_proj",
    "self_attn.k_v_proj": "self_attn.k_proj",
    "self_attn.v_u_proj": "self_attn.v_proj",
    "self_attn.v_v_proj": "self_attn.v_proj",
    "self_attn.o_u_proj": "self_attn.o_proj",
    "self_attn.o_v_proj": "self_attn.o_proj",
    "mlp.gate_u_proj": "mlp.gate_proj",
    "mlp.gate_v_proj": "mlp.gate_proj",
    "mlp.up_u_proj": "mlp.up_proj",
    "mlp.up_v_proj": "mlp.up_proj",
    "mlp.down_u_proj": "mlp.down_proj",
    "mlp.down_v_proj": "mlp.down_proj",
}

STATE_KEY_MAP = {
    "self_attn.q_u_proj.": "self_attn.q_proj.ALinear.",
    "self_attn.q_v_proj.": "self_attn.q_proj.BLinear.",
    "self_attn.k_u_proj.": "self_attn.k_proj.ALinear.",
    "self_attn.k_v_proj.": "self_attn.k_proj.BLinear.",
    "self_attn.v_u_proj.": "self_attn.v_proj.ALinear.",
    "self_attn.v_v_proj.": "self_attn.v_proj.BLinear.",
    "self_attn.o_u_proj.": "self_attn.o_proj.ALinear.",
    "self_attn.o_v_proj.": "self_attn.o_proj.BLinear.",
    "mlp.gate_u_proj.": "mlp.gate_proj.ALinear.",
    "mlp.gate_v_proj.": "mlp.gate_proj.BLinear.",
    "mlp.up_u_proj.": "mlp.up_proj.ALinear.",
    "mlp.up_v_proj.": "mlp.up_proj.BLinear.",
    "mlp.down_u_proj.": "mlp.down_proj.ALinear.",
    "mlp.down_v_proj.": "mlp.down_proj.BLinear.",
}


def _install_legacy_activation_compat():
    if hasattr(activations, "SiLUActivation"):
        return

    class SiLUActivation(torch.nn.Module):
        def forward(self, x):
            return F.silu(x)

    activations.SiLUActivation = SiLUActivation


def _prepend_pythonpaths(extra_pythonpaths: list[str]):
    for raw_path in reversed(extra_pythonpaths):
        path = Path(raw_path).expanduser().resolve()
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_checkpoint(checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            f"Expected a checkpoint dict with a 'model' entry, got: {type(payload)}"
        )
    return payload


def _canonical_low_rank_name(module_name: str) -> str | None:
    for old_suffix, new_suffix in MODULE_NAME_MAP.items():
        if module_name.endswith(old_suffix):
            return module_name[: -len(old_suffix)] + new_suffix
    return None


def _build_low_rank_specs(model) -> dict[str, dict[str, int]]:
    specs = {}
    for name, module in model.named_modules():
        target_name = _canonical_low_rank_name(name)
        if target_name is None:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue

        if name.endswith("_v_proj"):
            rank = int(module.out_features)
        else:
            rank = int(module.in_features)

        existing = specs.get(target_name)
        if existing is None:
            specs[target_name] = {"rank": rank}
        elif int(existing["rank"]) != rank:
            raise ValueError(
                f"Rank mismatch for {target_name}: {existing['rank']} vs {rank}"
            )
    return specs

def _remap_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if ".rotary_emb." in key:
            continue

        new_key = key
        for old_fragment, new_fragment in STATE_KEY_MAP.items():
            if old_fragment in key:
                new_key = key.replace(old_fragment, new_fragment)
                break

        if "_u_proj" in new_key or "_v_proj" in new_key:
            raise ValueError(f"Unmapped low-rank key remains: {new_key}")

        remapped[new_key] = value.detach().cpu().contiguous()
    return remapped


def _save_tokenizer(tokenizer, output_dir: Path, tokenizer_id: str | None):
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(output_dir)
            return
        except Exception:
            pass

    if tokenizer_id is None:
        raise ValueError(
            "Tokenizer was not usable from the checkpoint, and no fallback --tokenizer-id was provided."
        )

    fallback_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    fallback_tokenizer.save_pretrained(output_dir)


def _copy_model_code(output_dir: Path, source_dir: Path, filenames: tuple[str, ...]):
    for filename in filenames:
        source_path = (source_dir / filename).resolve()
        shutil.copy2(source_path, output_dir / Path(filename).name)


def _jsonify(value):
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        return {key: _jsonify(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _select_model_spec(source_model):
    model_type = getattr(source_model.config, "model_type", None)
    if model_type == "llama":
        from src.modeling.llama.configuration_lowrank_llama import LowRankLlamaConfig

        return {
            "config_cls": LowRankLlamaConfig,
            "auto_map": {
                "AutoConfig": "configuration_lowrank_llama.LowRankLlamaConfig",
                "AutoModel": "modeling_lowrank_llama.LowRankLlamaModel",
                "AutoModelForCausalLM": "modeling_lowrank_llama.LowRankLlamaForCausalLM",
            },
            "architectures": ["LowRankLlamaForCausalLM"],
            "source_dir": MODELING_ROOT / "llama",
            "copy_files": (
                "../common.py",
                "configuration_lowrank_llama.py",
                "modeling_lowrank_llama.py",
            ),
        }
    if model_type == "mistral":
        from src.modeling.mistral.configuration_lowrank_mistral import LowRankMistralConfig

        return {
            "config_cls": LowRankMistralConfig,
            "auto_map": {
                "AutoConfig": "configuration_lowrank_mistral.LowRankMistralConfig",
                "AutoModel": "modeling_lowrank_mistral.LowRankMistralModel",
                "AutoModelForCausalLM": "modeling_lowrank_mistral.LowRankMistralForCausalLM",
            },
            "architectures": ["LowRankMistralForCausalLM"],
            "source_dir": MODELING_ROOT / "mistral",
            "copy_files": (
                "../common.py",
                "configuration_lowrank_mistral.py",
                "modeling_lowrank_mistral.py",
            ),
        }
    if model_type == "qwen2":
        from src.modeling.qwen.configuration_lowrank_qwen2 import LowRankQwen2Config

        return {
            "config_cls": LowRankQwen2Config,
            "auto_map": {
                "AutoConfig": "configuration_lowrank_qwen2.LowRankQwen2Config",
                "AutoModel": "modeling_lowrank_qwen2.LowRankQwen2Model",
                "AutoModelForCausalLM": "modeling_lowrank_qwen2.LowRankQwen2ForCausalLM",
            },
            "architectures": ["LowRankQwen2ForCausalLM"],
            "source_dir": MODELING_ROOT / "qwen",
            "copy_files": (
                "../common.py",
                "configuration_lowrank_qwen2.py",
                "modeling_lowrank_qwen2.py",
            ),
        }
    if model_type == "qwen3":
        from src.modeling.qwen.configuration_lowrank_qwen3 import LowRankQwen3Config

        return {
            "config_cls": LowRankQwen3Config,
            "auto_map": {
                "AutoConfig": "configuration_lowrank_qwen3.LowRankQwen3Config",
                "AutoModel": "modeling_lowrank_qwen3.LowRankQwen3Model",
                "AutoModelForCausalLM": "modeling_lowrank_qwen3.LowRankQwen3ForCausalLM",
            },
            "architectures": ["LowRankQwen3ForCausalLM"],
            "source_dir": MODELING_ROOT / "qwen",
            "copy_files": (
                "../common.py",
                "configuration_lowrank_qwen3.py",
                "modeling_lowrank_qwen3.py",
            ),
        }
    raise ValueError(f"Unsupported base model_type for low-rank export: {model_type}")


def _build_config(source_model, low_rank_specs: dict[str, dict[str, int]], method_label: str):
    spec = _select_model_spec(source_model)
    raw_config = _jsonify(source_model.config.to_dict())
    config = spec["config_cls"].from_dict(raw_config)
    config.low_rank_modules = low_rank_specs
    config.low_rank_method = method_label
    config.low_rank_schema = "ABLinear"
    config.low_rank_format_version = 1
    config.auto_map = spec["auto_map"]
    config.architectures = spec["architectures"]
    return config, spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True, help="Path to the pickled SVDLLM checkpoint (.pt).")
    parser.add_argument("--output-dir", required=True, help="Where to write the HF low-rank export.")
    parser.add_argument(
        "--checkpoint-pythonpath",
        action="append",
        default=[],
        help="Extra source directories needed to unpickle the original checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Fallback tokenizer/model id when the checkpoint tokenizer cannot be saved.",
    )
    parser.add_argument(
        "--method-label",
        default="svdllm",
        help="Value stored in config.low_rank_method for downstream bookkeeping.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Shard size passed to Hugging Face state-dict serialization.",
    )
    parser.add_argument(
        "--unsafe-overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.unsafe_overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. Pass --unsafe-overwrite to reuse it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    _prepend_pythonpaths(args.checkpoint_pythonpath)
    _install_legacy_activation_compat()
    payload = _load_checkpoint(checkpoint_path)
    source_model = payload["model"]
    tokenizer = payload.get("tokenizer")

    low_rank_specs = _build_low_rank_specs(source_model)
    remapped_state_dict = _remap_state_dict(source_model.state_dict())
    config, model_spec = _build_config(source_model, low_rank_specs, method_label=args.method_label)

    print("[export] saving config and model code", flush=True)
    config.save_pretrained(output_dir)
    _copy_model_code(output_dir, source_dir=model_spec["source_dir"], filenames=model_spec["copy_files"])
    _save_tokenizer(tokenizer, output_dir, tokenizer_id=args.tokenizer_id or getattr(config, "_name_or_path", None))

    generation_config = getattr(source_model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output_dir)

    print("[export] writing safetensors shards", flush=True)
    save_torch_state_dict(
        remapped_state_dict,
        str(output_dir),
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
    )

    print(f"Saved low-rank HF export to {output_dir}")
    print(f"low_rank_method={config.low_rank_method}")
    print(f"low_rank_modules={len(low_rank_specs)}")


if __name__ == "__main__":
    main()
