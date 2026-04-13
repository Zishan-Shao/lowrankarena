from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.rank = int(rank)
        self.BLinear = nn.Linear(in_features, self.rank, bias=False)
        self.ALinear = nn.Linear(self.rank, out_features, bias=bias)

    def forward(self, x):
        return self.ALinear(self.BLinear(x))


class SharedBasis(nn.Linear):
    def __init__(self, num_basis: int, in_features: int):
        super().__init__(in_features, num_basis, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class BasisCoefficient(nn.Linear):
    def __init__(self, out_features: int, num_basis: int, bias: bool = False):
        super().__init__(num_basis, out_features, bias=bias)
        self.out_features = int(out_features)

    def forward(self, x):
        output_shape = x.size()[:-1] + (self.out_features,)
        return F.linear(x, self.weight, self.bias).view(output_shape)


def build_basis_collection(groups, num_basis: int, in_features: int) -> nn.ModuleDict:
    module_dict = nn.ModuleDict()
    for group in groups:
        basis = SharedBasis(num_basis, in_features)
        for item in group:
            module_dict[str(item)] = basis
    return module_dict


def resolve_parent(root: nn.Module, name: str):
    if "." not in name:
        return root, name
    parent_name, attr_name = name.rsplit(".", 1)
    parent = dict(root.named_modules())[parent_name]
    return parent, attr_name


def extract_rank(spec) -> int:
    if isinstance(spec, dict):
        if "rank" not in spec:
            raise ValueError(f"Missing rank in low-rank spec: {spec}")
        return int(spec["rank"])
    return int(spec)


def apply_low_rank_replacements(root: nn.Module, module_specs: dict[str, object]):
    for name, spec in module_specs.items():
        candidate_names = [name]
        if name.startswith("model."):
            candidate_names.append(name[len("model."):])

        resolved = None
        for candidate in candidate_names:
            try:
                resolved = resolve_parent(root, candidate)
                break
            except KeyError:
                continue

        if resolved is None:
            continue

        parent, attr_name = resolved
        original = getattr(parent, attr_name)
        if not isinstance(original, nn.Linear):
            continue

        replacement = LowRankLinear(
            in_features=original.in_features,
            out_features=original.out_features,
            rank=extract_rank(spec),
            bias=original.bias is not None,
        )
        replacement.to(device=original.weight.device, dtype=original.weight.dtype)
        setattr(parent, attr_name, replacement)
