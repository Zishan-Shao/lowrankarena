import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributions as dist
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
import numpy as np
import math
import random
from tqdm import tqdm
from pathlib import Path
import gc
import json    
from datetime import datetime
import time
import re

from utils.datautils import prepare_train_loaders
from modules.IncrementalPCA import IncrementalPCAonGPU
from modules.module import SVDTransformLayer
from modules.remapping import DOBI_quantize





def canonical_model_name(model_id_tail):
    # Local cache folders are often prefixed (e.g., "huggyllama__llama-7b").
    base = model_id_tail.split("__")[-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")


def main(args):
    # setting random seed of numpy and torch
    SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    DEV_GPU = torch.device('cuda:0')
    DEV_CPU= torch.device('cpu')
    DATASET_NAME = args.training_dataset
    model_id = args.model_id
    lower_id = model_id.split('/')[-1]

    # set path
    path_head_folder = Path(args.path_head_folder)
    path_head_folder_output = Path(args.path_head_folder_output)
    data_cache_dir = path_head_folder / 'data_cache' / DATASET_NAME
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    assert data_cache_dir.exists() is True
    dataset_cache_dir = path_head_folder / 'datasets' / DATASET_NAME
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    assert dataset_cache_dir.exists() is True
    output_dir = path_head_folder_output / 'training_output' / lower_id
    assert output_dir.exists() is True
    TA_tarined_model_output_dir = output_dir / args.training_result_path
    para_json = TA_tarined_model_output_dir /'para_config.json'
    with open(para_json, 'r') as file:
        para_data = json.load(file)
        
    # for model
    model_load_dtype = para_data['dtype_settings']['model_load_dtype']
    if model_load_dtype == 'f16':
        model_load_dtype = torch.float16
    computeSVD_dtype = para_data['dtype_settings']['computeSVD_dtype']
    if computeSVD_dtype == 'f32':
        computeSVD_dtype = torch.float32
    use_safetensors = False
    use_cpu_svd = os.getenv("DOBI_SVD_ON_CPU", "0") == "1"
    max_gamma_env = os.getenv("DOBI_MAX_GAMMA", "").strip()
    max_gamma = int(max_gamma_env) if max_gamma_env else None
    
    # for dataset
    NSAMPLES_train = para_data['dataset_processing']['NSAMPLES_train']
    SEQ_LEN = para_data['dataset_processing']['SEQ_LEN']
    NSAMPLES_val = para_data['dataset_processing']['NSAMPLES_val']
    target_compression_ratio = para_data["target_compression_ratio"]
    SAVE = args.SAVE
    RECREATE = args.RECREATE
    BETA = para_data['BETA']

    args.seq_len = SEQ_LEN
    
    if not args.remapping:
        model_alias = args.model_alias.strip() if args.model_alias else canonical_model_name(lower_id)
        method_name = args.method_name.strip()
        ratio_str = str(target_compression_ratio)
        save_dir = path_head_folder_output / 'compressed_model' / model_alias / method_name
        save_dir.mkdir(parents=True, exist_ok=True)
        output_dir = save_dir
        output_filename = f"{args.owner}_{model_alias}_{method_name}_{ratio_str}.{args.output_format}"
        output_save_path = output_dir / output_filename
    if args.remapping:
        save_dir = path_head_folder_output / 'compressed_model' / lower_id
        save_dir.mkdir(parents=True, exist_ok=True)
        output_dir = save_dir/ f"DobiSVD-{lower_id}-{target_compression_ratio}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_save_path = output_dir / "remapping_weight.pt"
    print("The compressed model will be saved in", output_save_path)
    
    model_no_svd_layer_dic = {}
    
    if "llama" in lower_id or "Llama" in lower_id:
        model_no_svd_layer_dic[lower_id] = ['lm_head']
    elif "opt" in lower_id:
        model_no_svd_layer_dic[lower_id] = ['project_out', 'project_in']
    else:
        # Keep output head untouched for generic decoder-only LMs (e.g., Qwen).
        model_no_svd_layer_dic[lower_id] = ['lm_head']
    no_svd_layers = model_no_svd_layer_dic[lower_id]


    # load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=model_load_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(DEV_GPU)
    
    # load json
    gamma_json_file = TA_tarined_model_output_dir / 'best_gamma.json'
    with open(gamma_json_file, 'r') as file:
        gamma_json = json.load(file)


    # prepared dataset
    tokenized_traindata, tokenized_valdata = prepare_train_loaders(tokenizer, DATASET_NAME, data_cache_dir, dataset_cache_dir, args)

    
    V_PCA_dict = {}
    PCA_tensor_dict = {}
    V_accu_list_dict = {}


    def process_batch(ipca,A,name): 
        ipca.partial_fit(A)
        principal_components = ipca.components_
        principal_components_tensor = torch.tensor(principal_components, device=A.device)
        PCA_tensor_dict[name] = principal_components_tensor.to(DEV_CPU)
        del principal_components_tensor
        del A
        del principal_components
    
    
    def svd_forward(self, x):
        # print('--------------',self.name,'-----------')
        if x.dim() == 3:
            x = x.squeeze(0)
        assert x.dim() == 2
        W_T = self.weight.to(computeSVD_dtype).T
        A = x.to(computeSVD_dtype) @ W_T
        
        gamma_range = math.ceil(self.gamma)
        real_gamma = min(A.shape[0], A.shape[1], max(1, gamma_range))
        Uf, Sf, Vf = torch.svd_lowrank(A, q=int(real_gamma), niter = 2)
        
        sequence = torch.arange(1, len(Sf)+1, device = A.device, dtype = A.dtype)
        Trunc = (0.5*torch.tanh(self.BETA * (real_gamma - sequence))+0.5)
        S_transformed = Sf*Trunc
        S_diag = torch.diag_embed(S_transformed)
        x_transformed = Uf @ S_diag @Vf.T
        x_transformed = x_transformed.to(model_load_dtype)

        # Use IPCA to Collect representive V_f
        Vf1=Vf.detach().to(DEV_CPU)
        V_accu_list_dict[self.name].append(Vf1)
        if len(V_accu_list_dict[self.name])==self.Ngamma:
            Vf_full = torch.concatenate(V_accu_list_dict[self.name][:self.Ngamma], axis=1).to(DEV_GPU)
            process_batch(V_PCA_dict[self.name], Vf_full.T, self.name)
            V_accu_list_dict[self.name]=[]
            del Vf_full
            gc.collect()
            torch.cuda.empty_cache()
            
        return x_transformed
    
    
    # Collect V_A by using IPCA
    for name, module in tqdm(model.named_modules(), desc="Add SVD attribute to modules"):
        if isinstance(module, nn.Linear) and all(x not in name for x in no_svd_layers):
            gamma = torch.tensor(gamma_json[name], dtype=computeSVD_dtype)
            module.forward = svd_forward.__get__(module, nn.Linear)
            module.register_buffer('gamma', gamma)
            module.register_buffer('BETA', torch.tensor(BETA))
            m,n =module.weight.shape
            
            cut = min(module.in_features, module.out_features)/SEQ_LEN
            # Some models (e.g. Qwen k/v projections) have cut < 1.
            # Ngamma must stay >= 1, otherwise IPCA accumulation never triggers.
            module.Ngamma = max(1, int(cut))
            if cut>1:
                real_gamma = min(m,n, max(1,cut*math.ceil(gamma)))
            else:
                real_gamma = min(m,n, max(1,math.ceil(gamma)))
            # print(real_gamma)
            V_PCA_dict[name] = IncrementalPCAonGPU(n_components=int(real_gamma))
            V_accu_list_dict[name] = []
            module.name = name
            
    model.to(DEV_GPU)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # weight updating
    for batch in tqdm(tokenized_traindata):
        # cnt += 1
        batch = {k: v.to(DEV_GPU).unsqueeze(0) for k, v in batch.items()}
        with torch.no_grad():
            model(**batch)

    
    def change_weight(gamma, W, name):
        W = W.to(computeSVD_dtype)
        principal_components_tensor = PCA_tensor_dict[name]
        V_pca =principal_components_tensor.T.cuda().to(computeSVD_dtype)
        sequence = torch.arange(1, V_pca.shape[1]+1).to(DEV_GPU)
        m,n =W.shape
        real_gamma = min(m,n, max(1, gamma))
        if max_gamma is not None:
            real_gamma = min(real_gamma, max_gamma)
        Trunc = (0.5*torch.tanh(BETA * (real_gamma - sequence))+0.5)
        
        G=torch.diag(Trunc)
        W_new = (W.T @ V_pca @ G @V_pca.T).T
        W_new = W_new.to(model_load_dtype)
        return W_new
    
    # Release first-pass model/IPCA GPU states before reloading dense model.
    model.to(DEV_CPU)
    del model
    V_PCA_dict.clear()
    V_accu_list_dict.clear()
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=model_load_dtype)
    model.to(DEV_GPU)
    for name, module in tqdm(model.named_modules(), desc="Update model weight"):
        if isinstance(module, nn.Linear) and all(x not in name for x in no_svd_layers):
            cut = min(module.in_features, module.out_features)/SEQ_LEN
            if cut>1:
                gamma = cut * torch.tensor(gamma_json[name], dtype=computeSVD_dtype)
            else:
                gamma = torch.tensor(gamma_json[name], dtype=computeSVD_dtype)
            if max_gamma is not None:
                gamma = min(float(gamma), max_gamma)
            W=module.weight.data.detach()
            new_weight = change_weight(gamma, W, name)
            module.weight.data = new_weight
            del W
        
        
    # update and save model
    if not args.remapping:
        for name, module in tqdm(model.named_modules(), desc="Decomposition weights"):
            if isinstance(module, nn.Linear) and all(x not in name for x in no_svd_layers):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                min_dim = min(module.in_features, module.out_features)
                cut = min(module.in_features, module.out_features) / SEQ_LEN
                base_gamma = float(gamma_json[name])
                if cut > 1:
                    gamma = cut * math.ceil(base_gamma)
                else:
                    gamma = math.ceil(base_gamma)
                gamma = min(float(gamma), float(min_dim))
                if max_gamma is not None:
                    gamma = min(float(gamma), max_gamma)
                svd_weight = module.weight.data.detach().cpu() if use_cpu_svd else module.weight.data.detach()
                NewLayer = SVDTransformLayer(gamma =int(gamma), weight = svd_weight, 
                                             bias = module.bias, name = name, device = model.device)
                setattr(parent, attr_name, NewLayer)
                del svd_weight
                del module
                if use_cpu_svd:
                    torch.cuda.empty_cache()
            
        
        torch.save({'model': model, 'tokenizer': tokenizer}, output_save_path)
        
    if args.remapping:
        mapping_info = {}
        for name, module in tqdm(model.named_modules(), desc="Remapping weights"):
            if isinstance(module, nn.Linear) and all(x not in name for x in ['lm_head']):
                cut = min(module.in_features, module.out_features)/2048
                if cut>1:
                    gamma = cut * torch.tensor(gamma_json[name], dtype=computeSVD_dtype)
                else:
                    gamma = torch.tensor(gamma_json[name], dtype=computeSVD_dtype)
                    
                W=module.weight.data.detach()
                us_quan, vt_quan, us_absmax, vt_absmax, tuple_info = DOBI_quantize(W, int(gamma), code = None)
                mapping_info[name] ={}
                mapping_info[name]["us_quan"]=us_quan
                mapping_info[name]["vt_quan"]=vt_quan
                mapping_info[name]["us_absmax"]=us_absmax
                mapping_info[name]["vt_absmax"]=vt_absmax
                mapping_info[name]["tuple_info"]=tuple_info
                
        torch.save(mapping_info, output_save_path)
        del model
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=model_load_dtype)
        orig_sd = model.state_dict()
        new_sd = {}
        for k, v in orig_sd.items():
            if "self_attn." in k or "mlp." in k:
                pass
            else:
                new_sd[k] = v
        model.config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir) 
        torch.save(new_sd, f"{output_dir}/pytorch_model.bin")
        
    print("done")



    











if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
#==========================================================================    
    # Settings related to the model.
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID",
    )

    parser.add_argument(
        "--remapping",
        action="store_true",
        default=False,
        help="whether to use remaping to save model",
    )
    
#==========================================================================    
    # Settings related to the Path.
    parser.add_argument(
        '--path_head_folder',
        type=str,
        default='./',
        help='path of the model and dataset'
    )

    parser.add_argument(
        '--path_head_folder_output',
        type=str,
        default='./results',
        help='path of the output result'
    )

    parser.add_argument(
        '--training_result_path',
        type=str,
        required=True,
        help='name of the k json'
    )
    parser.add_argument(
        "--owner",
        type=str,
        default=os.environ.get("USER", "owner"),
        help="owner tag in output filename",
    )
    parser.add_argument(
        "--model_alias",
        type=str,
        default="",
        help="model alias in output path/filename, default derives from model_id tail",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="DobiSVD",
        help="method tag in output path/filename",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="pt",
        help="file extension for non-remap compressed model artifact",
    )

#==========================================================================      
    # Settings related to the device.
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
#==========================================================================
    # Settings related to the datasets
    parser.add_argument(
        "--n_train_samples",
        type=int,
        default=256,
        help="number of samples used for finetuning",
    )
    
    parser.add_argument(
        "--n_eval_samples",
        type=int,
        default=256,
        help="number of samples used for evaluation",
    )
    
    parser.add_argument(
        "--training_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "alpaca", "selfgen"],
        help="finetuning dataset",
    )

    parser.add_argument(
        "--SAVE",
        action="store_true",
        default=True,
        help="whether to save the generated dataset",
    )

    parser.add_argument(
        "--RECREATE",
        action="store_true",
        default=False,
        help="whether to regenerate the dataset",
    )

 
    
 
    args = parser.parse_args()

    main(args)
