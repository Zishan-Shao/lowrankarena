from __future__ import annotations

from pathlib import Path


_MODEL_FOLDER_MAP = {
    "llama_7b": "llama_7b",
    "huggyllama__llama_7b": "llama_7b",
    "huggyllama__llama_7b_hf": "llama_7b",
    "llama_3_1_8b": "llama31_8b",
    "meta_llama__llama_3_1_8b": "llama31_8b",
    "llama_3_1_8b_instruct": "llama31_8b_Instruct",
    "meta_llama__llama_3_1_8b_instruct": "llama31_8b_Instruct",
    "qwen3_8b": "Qwen3-8b",
    "qwen__qwen3_8b": "Qwen3-8b",
    "qwen3_8b_base": "Qwen3-8b-Base",
    "qwen__qwen3_8b_base": "Qwen3-8b-Base",
    "llama_2_7b_hf": "llama2_7b",
    "meta_llama__llama_2_7b_hf": "llama2_7b",
    "llama2_7b": "llama2_7b",
}


def _normalize_token(value: str) -> str:
    return value.strip().replace("-", "_").replace(".", "_").lower()


def canonical_model_repo_folder(model_alias: str) -> str:
    normalized = _normalize_token(model_alias)
    return _MODEL_FOLDER_MAP.get(normalized, model_alias)


def infer_path_in_repo(bundle_path: Path, explicit_path_in_repo: str = "", default_method: str = "DobiSVD") -> str:
    if explicit_path_in_repo:
        return explicit_path_in_repo.rstrip("/")

    resolved = bundle_path.resolve()
    parts = resolved.parts
    model_alias = ""
    method_name = default_method

    if "compressed_model" in parts:
        idx = parts.index("compressed_model")
        if idx + 1 < len(parts):
            model_alias = parts[idx + 1]
        if idx + 2 < len(parts):
            method_name = parts[idx + 2]

    if not model_alias:
        model_alias = bundle_path.stem

    top_level = canonical_model_repo_folder(model_alias)
    leaf_dir = bundle_path.stem
    return f"{top_level}/{method_name}/{leaf_dir}"
