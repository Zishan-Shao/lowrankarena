import os.path
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from accelerate import load_checkpoint_and_dispatch

from config import ShareConfig
from utils import load_tokenizer, match_state_dict
from calib import Calib
from prepare_data import prepare_data
from utils import compute_num_basis
from group import change_model, update_model
from memory_guard import build_cuda_memory_guard

try:
    from models.gpt2 import ShareGPT2LMHeadModel
except ImportError:
    ShareGPT2LMHeadModel = None

try:
    from models.llama import ShareLlamaForCausalLM
except ImportError:
    ShareLlamaForCausalLM = None

try:
    from models.opt import ShareOPTForCausalLM
except ImportError:
    ShareOPTForCausalLM = None

try:
    from models.mistral import ShareMistralForCausalLM
except ImportError:
    ShareMistralForCausalLM = None


def _require_model_class(model_class, model_type):
    if model_class is None:
        raise ImportError(
            "Basis Sharing runtime for model_type '{}' failed to import. "
            "Check the local transformers compatibility for that backend.".format(model_type)
        )
    return model_class


def _resolve_runtime_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_release_guard(gpu_guard, reason: str):
    if gpu_guard is not None:
        gpu_guard.release_for_gpu_work(reason)


def _maybe_reserve_guard(gpu_guard, reason: str):
    if gpu_guard is not None:
        gpu_guard.reserve_idle(reason)


def _offload_model_to_cpu(model, gpu_guard=None, reason: str = "model offloaded to CPU"):
    _maybe_release_guard(gpu_guard, f"offloading for {reason}")
    try:
        model = model.to("cpu")
    except Exception:
        try:
            model = model.cpu()
        except Exception as exc:
            print(f"Warning: failed to offload model for guard: {exc}")
            return model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _maybe_reserve_guard(gpu_guard, reason)
    return model


def _build_gpu_guard(config):
    device = _resolve_runtime_device()
    return build_cuda_memory_guard(
        device,
        enabled=getattr(config, "gpu_guard", False),
        keep_free_gib=getattr(config, "gpu_guard_keep_free_gib", 60.0),
        reserve_fraction=getattr(config, "gpu_guard_reserve_fraction", 0.95),
        chunk_mib=getattr(config, "gpu_guard_chunk_mib", 256),
    )


def do_update_model(config, model, dataset, tokenizer, data_collator):
    gpu_guard = _build_gpu_guard(config)
    decomp_device = _resolve_runtime_device()
    try:
        if os.path.exists(config.updated_model_path):
            print("Start load model!")
            print("Load: {}".format(config.updated_model_path))
            if config.model_type == "gpt2":
                model = _require_model_class(ShareGPT2LMHeadModel, config.model_type).from_pretrained(
                    config.updated_model_path, device_map='auto', torch_dtype="auto"
                )
            elif ShareConfig.is_llama_model_type(config.model_type):
                model = _require_model_class(ShareLlamaForCausalLM, config.model_type).from_pretrained(
                    config.updated_model_path, device_map='auto', torch_dtype="auto"
                )
            elif config.model_type == "opt":
                model = _require_model_class(ShareOPTForCausalLM, config.model_type).from_pretrained(
                    config.updated_model_path, device_map='auto', torch_dtype="auto"
                )
            elif config.model_type == "mistral":
                model = _require_model_class(ShareMistralForCausalLM, config.model_type).from_pretrained(
                    config.updated_model_path, device_map='auto', torch_dtype="auto"
                )
            else:
                raise ValueError
        else:
            std_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="cpu", torch_dtype="auto")
            std_model.config.use_cache = False
            model = load_checkpoint_and_dispatch(model, config.untrained_model_path, device_map="auto")

            torch.manual_seed(2023)
            index = torch.randperm(len(dataset))
            index = index[:config.calibration_size]
            subset = Subset(dataset, index)
            dataloader = DataLoader(subset, batch_size=config.calib_batch_size, shuffle=False, collate_fn=data_collator,
                                    pin_memory=True, num_workers=4)

            if config.build_update_calib:
                print("Start build update calib!")
                names = config.share_part + config.private_part
                basis_name = []
                for name in names:
                    if name == "q" or name == "v" or name == "gate":
                        continue
                    basis_name.append(name + "_basis")

                Calib.build_update_dataset(model, dataloader, basis_name, config.model_type, config.update_calib_path)

            model = _offload_model_to_cpu(model, gpu_guard, "update calibration built and compressed model offloaded")
            model_config = model.config
            weight_info = ShareConfig.resolve_weight_info(config.model_name, model_config)

            names = config.share_part + config.private_part
            for name in names:
                print("Update {}".format(name))
                model = update_model(std_model=std_model,
                                     model=model,
                                     model_type=config.model_type,
                                     groups=getattr(model_config, name + "_groups"),
                                     name=getattr(config, name + "_name"),
                                     step=weight_info[getattr(config, name + "_name")][1],
                                     num_basis=getattr(model_config, "num_basis_" + name),
                                     basis_name=name + "_basis",
                                     calib_path=config.update_calib_path,
                                     device=decomp_device,
                                     gpu_guard=gpu_guard,
                                     )
            if config.save_updated_model:
                model.save_pretrained(config.updated_model_path, safe_serialization=False)
                tokenizer.save_pretrained(config.updated_model_path)
    finally:
        gpu_guard.close()
    return model


def create_model(config):
    if os.path.exists(config.untrained_model_path):
        model_path = config.untrained_model_path
        print("Start load model!")
        print("Start load: {}".format(config.untrained_model_path))
        if config.model_type == "gpt2":
            model = _require_model_class(ShareGPT2LMHeadModel, config.model_type).from_pretrained(
                model_path, device_map='auto', torch_dtype="auto"
            )
        elif ShareConfig.is_llama_model_type(config.model_type):
            if "30b" in config.untrained_model_path:
                model = _require_model_class(ShareLlamaForCausalLM, config.model_type).from_pretrained(
                    model_path, device_map='auto', torch_dtype=torch.float16
                )
            else:
                model = _require_model_class(ShareLlamaForCausalLM, config.model_type).from_pretrained(
                    model_path, device_map='cpu', torch_dtype="auto"
                )
        elif config.model_type == "opt":
            model = _require_model_class(ShareOPTForCausalLM, config.model_type).from_pretrained(
                model_path, device_map='auto', torch_dtype="auto"
            )
        elif config.model_type == "mistral":
            model = _require_model_class(ShareMistralForCausalLM, config.model_type).from_pretrained(
                model_path, device_map='auto', torch_dtype="auto"
            )
        else:
            raise ValueError

    else:
        gpu_guard = _build_gpu_guard(config)
        decomp_device = _resolve_runtime_device()
        try:
            tokenizer = load_tokenizer(config.model_name)
            print("Start create model!")
            model_config = AutoConfig.from_pretrained(config.model_name)
            model_config.use_cache = False
            if config.model_name == "jeffwan/llama-30b-hf":
                std_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto",
                                                                 torch_dtype=torch.float16)
            else:
                std_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto",
                                                                 torch_dtype="auto")

            if config.build_calib:
                train_dataset, val_dataset, tokenized_test, data_collator = prepare_data(config.dataset_name, tokenizer,
                                                                                         config.context_length,
                                                                                         config.dataset_cache_dir)
                torch.manual_seed(2023)
                index = torch.randperm(len(train_dataset))
                index = index[:config.calibration_size]
                subset = Subset(train_dataset, index)
                dataloader = DataLoader(subset, batch_size=config.calib_batch_size, shuffle=False,
                                        collate_fn=data_collator, pin_memory=True, num_workers=4)

                print("Start create calib!")
                calib_names = []
                if hasattr(config, "k_name"):
                    calib_names.append(config.k_name)
                if hasattr(config, "attn_name"):
                    calib_names.append(config.attn_name)
                calib_names.append(config.o_name)
                calib_names.append(config.up_name)
                calib_names.append(config.down_name)
                Calib.build_calibration_dataset(std_model, dataloader, calib_names, config.model_type, config.calib_path)
                print("Calib build done!")

            std_model = _offload_model_to_cpu(std_model, gpu_guard, "calibration built and base model offloaded")
            weight_info = ShareConfig.resolve_weight_info(config.model_name, model_config)

            names = config.share_part
            for name in names:
                print("Config for {}".format(name))
                nx, nf = weight_info[getattr(config, name + "_name")]
                num_group = model_config.num_hidden_layers // config.group_size
                rest = model_config.num_hidden_layers % config.group_size
                gs = config.group_size
                group = [[gs * i + j for j in range(config.group_size)] for i in range(num_group)]
                if rest != 0:
                    group += [[num_group * config.group_size + i for i in range(rest)]]
                setattr(model_config, name + "_groups", group)
                num_basis = compute_num_basis(nx, nf, config.group_size, config.compression_ratio)
                setattr(model_config, "num_basis_" + name, num_basis)
                print("num_basis {}".format(num_basis))

            names = config.private_part
            for name in names:
                print("Config for {}".format(name))
                setattr(model_config, name + "_groups", [[i] for i in range(model_config.num_hidden_layers)])
                nx, nf = weight_info[getattr(config, name + "_name")]
                num_basis = compute_num_basis(nx, nf, 1, config.compression_ratio)
                setattr(model_config, "num_basis_" + name, num_basis)
                print("num_basis {}".format(num_basis))

            if ShareConfig.is_llama_model_type(config.model_type):
                if "30b" in config.model_name:
                    model_config.torch_dtype = torch.float16
                model = _require_model_class(ShareLlamaForCausalLM, config.model_type)(model_config)
            elif config.model_type == "gpt2":
                model = _require_model_class(ShareGPT2LMHeadModel, config.model_type)(model_config)
            elif config.model_type == "opt":
                model = _require_model_class(ShareOPTForCausalLM, config.model_type)(model_config)
            elif config.model_type == "mistral":
                model = _require_model_class(ShareMistralForCausalLM, config.model_type)(model_config)
            else:
                raise NotImplementedError

            print("Model init finished!")
            if not hasattr(config, "tfs"):
                matched_state_dict, _ = match_state_dict(model.state_dict(), std_model.state_dict())
                model.load_state_dict(matched_state_dict, strict=False)

                names = config.share_part + config.private_part
                for name in names:
                    print("Change {}".format(name))
                    model = change_model(std_model=std_model,
                                         model=model,
                                         model_type=config.model_type,
                                         groups=getattr(model_config, name + "_groups"),
                                         name=getattr(config, name + "_name"),
                                         step=weight_info[getattr(config, name + "_name")][1],
                                         num_basis=getattr(model_config, "num_basis_" + name),
                                         basis_name=name + "_basis",
                                         calib_path=config.calib_path,
                                         device=decomp_device,
                                         gpu_guard=gpu_guard,
                                         )

                if config.save_untrained_model:
                    model.save_pretrained(config.untrained_model_path, safe_serialization=False)
                    tokenizer.save_pretrained(config.untrained_model_path)
        finally:
            gpu_guard.close()

    return model
