import torch
from tqdm import tqdm
from calib import Calib
from config import ShareConfig


def _maybe_release_guard(gpu_guard, reason: str):
    if gpu_guard is not None:
        gpu_guard.release_for_gpu_work(reason)


def _maybe_reserve_guard(gpu_guard, reason: str):
    if gpu_guard is not None:
        gpu_guard.reserve_idle(reason)


class Group:
    def __init__(self, std_model, group_member, name, step, model_type, s, invs, device=None, gpu_guard=None):
        """
        :param std_model: the original model
        :param group_member: the layers which share2 the same parameter
        :param names: list, share2 model name
        :param steps: list, the col num of each name
        """
        self.member = group_member
        self.model_type = model_type
        self.name = name
        self.step = step
        self.basis = None
        self.coefficient = None
        self.sigma = None
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gpu_guard = gpu_guard
        self._init_basis_coefficient(std_model, s, invs)

    def _init_gpt2(self, std_model):
        assert self.model_type == 'gpt2'
        w = []
        model = std_model.transformer.h
        for layer in self.member:
            data = model[layer].get_submodule(self.name).weight.data
            w.append(data)
        return w

    def _init_llama2(self, std_model):
        assert ShareConfig.is_llama_model_type(self.model_type)
        w = []
        model = std_model.model.layers
        for layer in self.member:
            data = model[layer].get_submodule(self.name).weight.data
            w.append(data.T)
        return w

    def _init_opt(self, std_model):
        assert self.model_type == "opt"
        w = []
        model = std_model.model.decoder.layers
        for layer in self.member:
            data = model[layer].get_submodule(self.name).weight.data
            w.append(data.T)
        return w

    def _init_mistral(self, std_model):
        assert self.model_type == 'mistral'
        w = []
        model = std_model.model.layers
        for layer in self.member:
            data = model[layer].get_submodule(self.name).weight.data
            w.append(data.T)
        return w

    def _init_basis_coefficient(self, std_model, s, invs):
        if self.model_type == 'gpt2':
            w = self._init_gpt2(std_model)
        elif ShareConfig.is_llama_model_type(self.model_type):
            w = self._init_llama2(std_model)
        elif self.model_type == "opt":
            w = self._init_opt(std_model)
        elif self.model_type == "mistral":
            w = self._init_mistral(std_model)
        else:
            raise NotImplementedError

        _maybe_release_guard(self.gpu_guard, f"basis decomposition for {self.name} group {self.member}")
        w = torch.cat([item.to(self.device) for item in w], -1).double()
        s = s.to(self.device)
        invs = invs.to(self.device)
        w = s @ w
        u, sigma, v = torch.svd(w)
        self.basis = torch.matmul(invs @ u, torch.diag(sigma)).float().cpu()
        self.coefficient = v.T.float().cpu()
        del w, s, invs, u, sigma, v
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        _maybe_reserve_guard(self.gpu_guard, f"basis decomposition for {self.name} group {self.member} finished")

    def _get_coefficient_split(self):
        res = {}
        offset = self.step
        for i, layer in enumerate(self.member):
            res[layer] = {}
            start = offset * i
            co_attn = self.coefficient[:, start:start + offset]
            res[layer][self.name] = co_attn
        return res

    def change_basis(self, model, num_basis, basis_name):
        if self.model_type == 'gpt2':
            tmp_model = model.transformer
        elif ShareConfig.is_llama_model_type(self.model_type):
            tmp_model = model.model
        elif self.model_type == "opt":
            tmp_model = model.model.decoder
        elif self.model_type == "mistral":
            tmp_model = model.model
        else:
            raise NotImplementedError
        tmp_model.get_submodule(basis_name)[str(self.member[0])].set_weight(self.basis[:, :num_basis])
        return model

    def change_coefficient(self, model, num_basis):
        if self.model_type == 'gpt2':
            tmp_model = model.transformer.h
        elif ShareConfig.is_llama_model_type(self.model_type):
            tmp_model = model.model.layers
        elif self.model_type == "opt":
            tmp_model = model.model.decoder.layers
        elif self.model_type == "mistral":
            tmp_model = model.model.layers
        else:
            raise NotImplementedError
        co = self._get_coefficient_split()
        for i, layer in enumerate(self.member):
            weight = co[layer][self.name][:num_basis, :]
            tmp_model[layer].get_submodule(self.name).set_weight(weight)
        return model


def change_model(std_model, model, model_type, groups, name, step, num_basis, basis_name, calib_path, device=None,
                 gpu_guard=None):
    _maybe_reserve_guard(gpu_guard, f"waiting to start basis decomposition for {name}")
    for group in tqdm(groups):
        s, inv_s = Calib.get_s_inv_s(group, name, model_type, calib_path)
        item = Group(std_model, group, name=name, step=step, model_type=model_type, s=s, invs=inv_s, device=device,
                     gpu_guard=gpu_guard)
        model = item.change_basis(model, num_basis, basis_name)
        model = item.change_coefficient(model, num_basis)
    return model


def update_model(std_model, model, model_type, groups, name, step, num_basis, basis_name, calib_path, device=None,
                 gpu_guard=None):
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if model_type == "gpt2":
        tmp_std_model = std_model.transformer.h
        tmp_model = model.model.trtansformer.h
        tmp = model.model.transformer
    elif ShareConfig.is_llama_model_type(model_type) or model_type == "mistral":
        tmp_std_model = std_model.model.layers
        tmp_model = model.model.layers
        tmp = model.model
    elif model_type == "opt":
        tmp_std_model = std_model.model.decoder.layers
        tmp_model = model.model.decoder.layers
        tmp = model.model.decoder
    else:
        raise NotImplementedError
    _maybe_reserve_guard(gpu_guard, f"waiting to start coefficient update for {name}")
    for group in tqdm(groups):
        _maybe_release_guard(gpu_guard, f"coefficient update for {name} group {group}")
        w = []
        for layer_idx in group:
            if model_type == "gpt2":
                data = tmp_std_model[layer_idx].get_submodule(name).weight.data
            else:
                data = tmp_std_model[layer_idx].get_submodule(name).weight.data.T
            w.append(data)

        u = tmp.get_submodule(basis_name)[str(group[0])].weight.data.T.to(device).double()
        assert u.shape[1] == num_basis

        w = torch.cat([item.to(device) for item in w], -1).double()
        if basis_name == "q_basis" or basis_name == "v_basis":
            xtx = Calib.get_calib_data(group, "k_basis", calib_path)
        elif basis_name == "gate_basis":
            xtx = Calib.get_calib_data(group, "up_basis", calib_path)
        else:
            xtx = Calib.get_calib_data(group, basis_name, calib_path)
        xtx = xtx.to(device).double()

        inv = torch.inverse(u.T @ xtx @ u)
        vt = w.T @ xtx @ u @ inv
        v = vt.T.float().cpu()

        for i, layer_idx in enumerate(group):
            data = v[:, i * step:(i + 1) * step]
            tmp_model[layer_idx].get_submodule(name).set_weight(data)
        del w, u, xtx, inv, vt, v
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _maybe_reserve_guard(gpu_guard, f"coefficient update for {name} group {group} finished")

    return model
