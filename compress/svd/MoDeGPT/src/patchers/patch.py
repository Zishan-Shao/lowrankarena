import logging

import torch

# from transformers.models.opt.modeling_opt import OPTDecoderLayer
from src.compression_utils import get_gate_projs
import copy
from src.adapters.model_adapter import ModelAdapter

logger = logging.getLogger("MoDeGPT")


def patch_config(model: torch.nn.Module):
    og_config = copy.deepcopy(model.config)
    config = model.config

    adapter = ModelAdapter.from_model(model)

    n_layers, arch = adapter.n_layers, adapter.arch

    # handle qk/vo separately in case only one stage is used
    q_ranks = []
    k_ranks = []
    v_ranks = []
    o_ranks = []
    gate_ranks = []
    for layer in range(n_layers):
        query_weight, key_weight = adapter.get_qk_weights(layer)
        value_weight, output_weight = adapter.get_vo_weights(layer)

        q_rank = query_weight.shape[0]  # q/k
        k_rank = key_weight.shape[0]  # q/k

        v_rank = value_weight.shape[0]  # v/o
        o_rank = output_weight.shape[1]

        q_ranks.append(q_rank)
        k_ranks.append(k_rank)
        v_ranks.append(v_rank)
        o_ranks.append(o_rank)
        if arch == "mixtral":
            experts_rank = []
            n_experts = adapter.n_experts
            for expert_idx in range(n_experts):
                gate_rank = adapter.get_mlp_rank(layer, expert_idx)
                experts_rank.append(gate_rank)
            gate_ranks.append(experts_rank)
        elif arch == "deepseek":
            gate_rank = adapter.get_mlp_components(layer, 0).up_proj.weight.shape[0]
            if layer != 0:
                gate_ranks.append([gate_rank for _ in range(adapter.n_experts)])
            else:
                gate_ranks.append(gate_rank)
        else:
            _, up_proj, _, _, _ = get_gate_projs(model, layer)
            gate_rank = up_proj.weight.shape[0]  # u/d
            gate_ranks.append(gate_rank)

    # just break these values so we know where's something's going wrong later
    config.ffn_dim = -1
    config.q_ranks = q_ranks
    config.k_ranks = k_ranks
    config.v_ranks = v_ranks
    config.o_ranks = o_ranks
    config.gate_ranks = gate_ranks
    if arch == "opt":
        config.auto_map = {"AutoModelForCausalLM": "OPTRebuild.OPTForCausalLM"}
    if arch == "llama":
        config.auto_map = {"AutoModelForCausalLM": "LlamaRebuild.LlamaForCausalLM"}
    if arch == "mixtral":
        config.auto_map = {"AutoModelForCausalLM": "MixtralRebuild.MixtralForCausalLM"}
    if arch == "deepseek":
        config.auto_map["AutoModelForCausalLM"] = "DeepseekRebuild.DeepseekForCausalLM"

    return og_config
