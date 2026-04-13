from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WRAPPER_METADATA_NAME = "vllm_wrapper_meta.json"

MODEL_FILE = """from transformers import LlamaForCausalLM, LlamaModel

from .configuration_svdllm_llama import SVDLLMLlamaConfig
from .modeling_svdllm_common import replace_with_svd_linears


class SVDLLMLlamaModel(LlamaModel):
    config_class = SVDLLMLlamaConfig
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        self.svd_ranks = dict(getattr(config, "svd_ranks", {}) or {})
        self.replaced_svd_modules, self.missing_svd_modules = replace_with_svd_linears(self, self.svd_ranks)
        if self.missing_svd_modules:
            raise ValueError(
                "SVD-LLM replacement failed for modules: " + ", ".join(sorted(self.missing_svd_modules))
            )


class SVDLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = SVDLLMLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = SVDLLMLlamaModel(config)
"""


INIT_FILE = """from .configuration_svdllm_llama import SVDLLMLlamaConfig
from .modeling_svdllm_llama import SVDLLMLlamaForCausalLM, SVDLLMLlamaModel

__all__ = [
    "SVDLLMLlamaConfig",
    "SVDLLMLlamaModel",
    "SVDLLMLlamaForCausalLM",
]
"""


PASSTHROUGH_NAMES = [
    "__init__.py",
    "generation_config.json",
    "modeling_svdllm_common.py",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "pytorch_model.bin.index.json",
]


@dataclass(slots=True)
class WrapperMaterialization:
    output_dir: Path
    created: bool
    reused: bool
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a vLLM-compatible local wrapper for an SVD-LLM LLaMA checkpoint."
    )
    parser.add_argument("--source-model", required=True, help="Local source checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Where to write the local wrapper model.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing wrapper directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def symlink_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def build_config(source_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(source_config)
    auto_map = dict(config.get("auto_map") or {})
    auto_map["AutoConfig"] = "configuration_svdllm_llama.SVDLLMLlamaConfig"
    auto_map["AutoModel"] = "modeling_svdllm_llama.SVDLLMLlamaModel"
    auto_map["AutoModelForCausalLM"] = "modeling_svdllm_llama.SVDLLMLlamaForCausalLM"
    config["auto_map"] = auto_map
    # vLLM's Transformers backend validates architectures against its
    # registry before it consults auto_map. Declaring the generic
    # Transformers backend class keeps the model on the supported path
    # while the custom auto_map still points Transformers at our SVD code.
    config["architectures"] = ["TransformersForCausalLM"]
    return config


def build_wrapper_metadata(source_model: Path, source_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "wrapper_format": "svdllm_llama_vllm_transformers_v1",
        "source_model": str(source_model.resolve()),
        "source_model_type": source_config.get("model_type"),
        "source_checkpoint_name": source_model.name,
    }


def _validate_source_model(source_model: Path) -> dict[str, Any]:
    if not source_model.exists():
        raise FileNotFoundError(source_model)

    source_config_path = source_model / "config.json"
    if not source_config_path.exists():
        raise FileNotFoundError(source_config_path)

    source_config = load_json(source_config_path)
    model_type = source_config.get("model_type")
    if model_type != "svdllm-llama":
        raise ValueError(f"Expected model_type='svdllm-llama', got {model_type!r}")
    return source_config


def _looks_like_compatible_wrapper(output_dir: Path) -> tuple[bool, dict[str, Any] | None]:
    config_path = output_dir / "config.json"
    model_path = output_dir / "modeling_svdllm_llama.py"
    if not config_path.exists() or not model_path.exists():
        return False, None

    try:
        config = load_json(config_path)
    except Exception:
        return False, None

    auto_map = dict(config.get("auto_map") or {})
    is_compatible = (
        config.get("model_type") == "svdllm-llama"
        and list(config.get("architectures") or []) == ["TransformersForCausalLM"]
        and auto_map.get("AutoModel") == "modeling_svdllm_llama.SVDLLMLlamaModel"
        and auto_map.get("AutoModelForCausalLM") == "modeling_svdllm_llama.SVDLLMLlamaForCausalLM"
    )
    if not is_compatible:
        return False, None

    metadata_path = output_dir / WRAPPER_METADATA_NAME
    metadata = load_json(metadata_path) if metadata_path.exists() else None
    return True, metadata


def materialize_svdllm_llama_wrapper(
    source_model: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    allow_reuse: bool = True,
) -> WrapperMaterialization:
    source_model = Path(source_model).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    source_config = _validate_source_model(source_model)
    expected_metadata = build_wrapper_metadata(source_model, source_config)

    if output_dir.exists() and any(output_dir.iterdir()):
        is_compatible, existing_metadata = _looks_like_compatible_wrapper(output_dir)
        if not overwrite and allow_reuse and is_compatible:
            if existing_metadata is None or existing_metadata.get("source_model") == expected_metadata["source_model"]:
                return WrapperMaterialization(
                    output_dir=output_dir,
                    created=False,
                    reused=True,
                    metadata=existing_metadata or expected_metadata,
                )
        if not overwrite and not allow_reuse:
            raise FileExistsError(
                f"{output_dir} already exists and is not empty. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for name in PASSTHROUGH_NAMES:
        src = source_model / name
        if src.exists():
            symlink_file(src, output_dir / name)

    for shard in sorted(source_model.glob("pytorch_model-*.bin")):
        symlink_file(shard, output_dir / shard.name)

    config_src = source_model / "configuration_svdllm_llama.py"
    if not config_src.exists():
        raise FileNotFoundError(config_src)
    symlink_file(config_src, output_dir / config_src.name)

    write_text(output_dir / "__init__.py", INIT_FILE)
    write_text(output_dir / "modeling_svdllm_llama.py", MODEL_FILE)
    write_json(output_dir / "config.json", build_config(source_config))
    write_json(output_dir / WRAPPER_METADATA_NAME, expected_metadata)

    return WrapperMaterialization(
        output_dir=output_dir,
        created=True,
        reused=False,
        metadata=expected_metadata,
    )


def ensure_svdllm_llama_wrapper(source_model: str | Path, output_dir: str | Path) -> WrapperMaterialization:
    return materialize_svdllm_llama_wrapper(
        source_model,
        output_dir,
        overwrite=False,
        allow_reuse=True,
    )


def main() -> None:
    args = parse_args()
    materialized = materialize_svdllm_llama_wrapper(
        args.source_model,
        args.output_dir,
        overwrite=args.overwrite,
        allow_reuse=not args.overwrite,
    )
    if materialized.created:
        print(f"[ok] wrapper written to {materialized.output_dir}")
    else:
        print(f"[ok] wrapper already exists: {materialized.output_dir}")
    print("[note] use tokenizer_mode=slow when running vLLM for this checkpoint")


if __name__ == "__main__":
    main()
