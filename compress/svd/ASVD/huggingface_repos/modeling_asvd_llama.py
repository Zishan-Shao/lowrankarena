from __future__ import annotations

import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

from .configuration_asvd_llama import ASVDLlamaConfig


class ASVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        return self.ALinear(self.BLinear(input))


def _extract_rank(spec) -> int:
    if isinstance(spec, dict):
        if "rank" not in spec:
            raise ValueError(f"Missing rank in ASVD spec: {spec}")
        return int(spec["rank"])
    return int(spec)


def _resolve_parent(root: nn.Module, name: str):
    if "." not in name:
        return root, name
    parent_name, attr_name = name.rsplit(".", 1)
    return dict(root.named_modules())[parent_name], attr_name


def _apply_asvd_replacements(root: nn.Module, truncation_ranks: dict[str, object], *, strict: bool):
    replaced_modules = []
    missing_modules = []
    for name, spec in truncation_ranks.items():
        candidate_names = [name]
        if name.startswith("model."):
            candidate_names.append(name[len("model."):])

        resolved = None
        for candidate in candidate_names:
            try:
                resolved = _resolve_parent(root, candidate)
                break
            except KeyError:
                continue

        if resolved is None:
            missing_modules.append(name)
            continue

        parent, attr_name = resolved
        original = getattr(parent, attr_name)
        if not isinstance(original, nn.Linear):
            missing_modules.append(name)
            continue

        replacement = ASVDLinear(
            original.in_features,
            original.out_features,
            _extract_rank(spec),
            bias=original.bias is not None,
        )
        replacement.to(device=original.weight.device, dtype=original.weight.dtype)
        setattr(parent, attr_name, replacement)
        replaced_modules.append(name)

    if strict and missing_modules:
        raise ValueError("ASVD replacement failed for modules: " + ", ".join(sorted(missing_modules)))
    return replaced_modules, missing_modules


class ASVDLlamaModel(LlamaModel):
    config_class = ASVDLlamaConfig
    _supports_attention_backend = True

    def __init__(self, config: ASVDLlamaConfig):
        super().__init__(config)
        truncation_ranks = dict(getattr(config, "truncation_ranks", {}) or {})
        truncation_ranks.pop("lm_head", None)
        self.replaced_low_rank_modules, self.missing_low_rank_modules = _apply_asvd_replacements(
            self,
            truncation_ranks,
            strict=True,
        )


class ASVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = ASVDLlamaConfig

    def __init__(self, config: ASVDLlamaConfig):
        super().__init__(config)
        self.replaced_low_rank_modules, self.missing_low_rank_modules = _apply_asvd_replacements(
            self,
            getattr(config, "truncation_ranks", {}) or {},
            strict=True,
        )
