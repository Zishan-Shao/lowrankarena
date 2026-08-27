import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import load_file
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_and_replace_layer(model: nn.Module, layer_name: str, replacement_module: nn.Module):
    """
    Finds a layer/module within a model by its qualified name and replaces it
    with a new module in place. Handles both attribute access (e.g., model.mlp)
    and indexed access within ModuleLists/Sequentials (e.g., model.layers.0).

    Args:
        model: The parent model (e.g., the full Llama model).
        layer_name: The qualified name of the layer to replace
                    (e.g., "model.layers.10.mlp.down_proj").
        replacement_module: The new module (e.g., an nn.Sequential) to insert.

    Raises:
        AttributeError: If an attribute name in the path doesn't exist.
        IndexError: If an index in the path is out of bounds.
        ValueError: If a part of the path expected to be an index is not a valid integer.
    """
    attrs = layer_name.split('.')

    parent_module = model
    for i, attr_name in enumerate(attrs[:-1]):
        try:
            if attr_name.isdigit():
                idx = int(attr_name)
                parent_module = parent_module[idx]
            else:
                parent_module = getattr(parent_module, attr_name)
        except (AttributeError, IndexError, ValueError) as e:
            logging.error(f"Error accessing parent module part '{attr_name}' (index {i}) in layer name '{layer_name}'.")
            logging.error(f"Parent module type at this point: {type(parent_module)}")
            if isinstance(e, AttributeError):
                 logging.error(f"Available attributes/children: {dir(parent_module)}")
            elif isinstance(e, IndexError):
                 logging.error(f"Available indices length: {len(parent_module) if hasattr(parent_module, '__len__') else 'N/A'}")
            raise e

    final_attr_name = attrs[-1]

    try:
        if final_attr_name.isdigit():
            idx = int(final_attr_name)
            logging.info(f"Replacing '{layer_name}' (index {idx} within parent {type(parent_module).__name__})")
            parent_module[idx] = replacement_module
        else:
            # Replacing a submodule by attribute name
            logging.info(f"Replacing '{layer_name}' (attribute '{final_attr_name}' within parent {type(parent_module).__name__})")
            setattr(parent_module, final_attr_name, replacement_module)
    except (AttributeError, IndexError, ValueError) as e:
        logging.error(f"Error replacing final module part '{final_attr_name}' (index {len(attrs)-1}) in layer name '{layer_name}'.")
        logging.error(f"Parent module type: {type(parent_module)}")
        if isinstance(e, AttributeError):
             logging.error(f"Available attributes/children: {dir(parent_module)}")
        elif isinstance(e, IndexError):
             logging.error(f"Available indices length: {len(parent_module) if hasattr(parent_module, '__len__') else 'N/A'}")
        raise e # Re-raise the original exception

    logging.debug(f"Successfully replaced '{layer_name}'.") # Use debug for success message

def is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def factorize_layer_kron_svd(
    layer: nn.Linear,
    XF: np.ndarray,
    YF: np.ndarray,
    rank: int,
    reg_alpha: float = 1e-5,
    max_reg_tries: int = 10,
    alpha_increase_factor: float = 10.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> nn.Sequential:
    """
    Factorizes a linear layer using Kronecker-factored SVD.
    
    Args:
        layer: Original nn.Linear layer.
        XF: Kronecker factor X (input side).
        YF: Kronecker factor Y (output side).
        rank: Target rank for the decomposition.
        reg_alpha: Regularization strength for positive definiteness.
        max_reg_tries: Maximum attempts to regularize.
        alpha_increase_factor: How much to increase alpha each retry.
        device: Device for the new layers.
        dtype: Data type for the new layers.

    Returns:
        An nn.Sequential module containing two new linear layers.
    """
    print ("factorize_layer_kron_svd")
    logging.info(f"factorize_layer_kron_svd")
    W = layer.weight.data.to(device='cpu', dtype=torch.float32).numpy()
    bias = layer.bias.data.to(device='cpu', dtype=torch.float32).numpy() if layer.bias is not None else None
    out_features, in_features = W.shape

    assert YF.shape == (out_features, out_features), f"Y factor shape mismatch for layer: {YF.shape}"
    assert XF.shape == (in_features, in_features), f"X factor shape mismatch for layer: {XF.shape}"

    print(f"Regularizing factors for layer (initial alpha={reg_alpha:.2e})...")
    XF_reg, YF_reg = XF.copy(), YF.copy()

    def regularize_factor(XF_reg, factor, diag_mean, max_reg_tries, reg_alpha, eye):
        for i in range(max_reg_tries + 1):
            if is_positive_definite(XF_reg):
                print(f"  Factor is positive definite (alpha={reg_alpha:.2e})")
                logging.info(f"  Factor is positive definite (alpha={reg_alpha:.2e})")
                break
            if i == max_reg_tries:
                raise RuntimeError(f"Failed to regularize factor after {max_reg_tries} attempts.")
            print(f"  Regularizing factor (try {i+1}, alpha={reg_alpha:.2e})")
            logging.info(f"  Regularizing factor (try {i+1}, alpha={reg_alpha:.2e})")
            reg_alpha *= alpha_increase_factor
            XF_reg = (1 - reg_alpha) * factor + reg_alpha * eye * diag_mean
        return XF_reg

    eye_X = np.eye(in_features, dtype=np.float32)
    diag_mean_X = max(np.mean(np.diag(XF)), 1e-6)
    XF_reg = regularize_factor(XF_reg, XF, diag_mean_X, max_reg_tries, reg_alpha, eye_X)

    eye_Y = np.eye(out_features, dtype=np.float32)
    diag_mean_Y = max(np.mean(np.diag(YF)), 1e-6)
    YF_reg = regularize_factor(YF_reg, YF, diag_mean_Y, max_reg_tries, reg_alpha, eye_Y)

    try:
        X_chol, Y_chol = np.linalg.cholesky(XF_reg), np.linalg.cholesky(YF_reg)
        print("  Cholesky decomposition successful.")
    except np.linalg.LinAlgError as e:
        print(f"ERROR: Cholesky decomposition failed: {e}")
        raise e

    print(f"  Performing SVD on transformed matrix ({out_features}x{in_features})...")
    W_tilde = Y_chol.T @ W @ X_chol
    U_hat, S_hat, Vt_hat = np.linalg.svd(W_tilde, full_matrices=False)
    print(f"  SVD complete. Singular values (top 5): {S_hat[:5]}")

    rank = min(rank, len(S_hat)) if rank > 0 else len(S_hat)
    S_hat_positive = np.maximum(S_hat[:rank], 1e-8)
    S_hat_sqrt = np.sqrt(S_hat_positive)
    S_hat_diag = np.diag(S_hat_sqrt)

    try:
        inv_X_chol, inv_Y_chol_T = np.linalg.inv(X_chol), np.linalg.inv(Y_chol.T)
        print("  Cholesky factor inverses computed.")
    except np.linalg.LinAlgError as e:
        print(f"ERROR: Failed to invert Cholesky factors: {e}")
        raise e

    W1 = S_hat_diag @ Vt_hat[:rank, :] @ inv_X_chol
    W2 = inv_Y_chol_T @ U_hat[:, :rank] @ S_hat_diag

    W1_tensor = torch.tensor(W1, dtype=dtype, device=device)
    W2_tensor = torch.tensor(W2, dtype=dtype, device=device)

    linear1 = nn.Linear(in_features=in_features, out_features=rank, bias=False)
    linear1.weight = nn.Parameter(W1_tensor)

    linear2 = nn.Linear(in_features=rank, out_features=out_features, bias=(bias is not None))
    linear2.weight = nn.Parameter(W2_tensor)
    if bias is not None:
        linear2.bias = nn.Parameter(torch.tensor(bias, dtype=dtype, device=device))

    factorized_layer = nn.Sequential(linear1, linear2)
    factorized_layer.to(device=device, dtype=dtype)

    print(f"  Factorization complete for layer. New shapes: ({rank},{in_features}), ({out_features},{rank})")
    return factorized_layer


def calculate_rank_from_ratio(
    in_features: int,
    out_features: int,
    param_ratio: float,
    rank_alignment: int = 1,
    min_rank: int = 1,
) -> int:
    """
    Calculates the SVD rank 'r' needed to achieve a target parameter ratio.

    The factorization W (n x m) -> W2 (n x r) @ W1 (r x m) results in
    r * (n + m) parameters. We find r such that r * (n + m) ≈ (n * m) * param_ratio.

    Args:
        in_features (m): Input features of the linear layer.
        out_features (n): Output features of the linear layer.
        param_ratio: The desired fraction of original parameters to keep (0.0 to 1.0).
        rank_alignment: The rank will be rounded up to the nearest multiple of this value.
        min_rank: The minimum allowed rank.

    Returns:
        The calculated rank.
    """
    if param_ratio <= 0:
        return min_rank
    if param_ratio >= 1:
        # No compression needed, return maximum possible rank (or close to it)
        # Technically max rank is min(n, m), but returning this might prevent
        # any SVD if ratio is exactly 1.0. Let's return a large number,
        # the SVD function will cap it anyway. Or just return min(n,m).
        return min(out_features, in_features)

    original_params = in_features * out_features
    target_params = original_params * param_ratio

    # Formula: target_params = rank * (in_features + out_features)
    # Solve for rank: rank = target_params / (in_features + out_features)
    denominator = in_features + out_features
    if denominator == 0: # Avoid division by zero for empty layers
        return min_rank

    rank_float = target_params / denominator

    # Apply rank alignment: Round up to the nearest multiple
    if rank_alignment > 1:
        rank = int(math.ceil(rank_float / rank_alignment) * rank_alignment)
    else:
        rank = int(math.ceil(rank_float))

    max_rank = min(out_features, in_features)
    rank = max(min_rank, min(rank, max_rank))

    return rank

def compress_llama_with_kron_svd(
    model_name_or_path: str,
    kron_factors_dir: str,
    sensitivity_dict: dict, # Changed from target_rank
    output_dir: str= "./llama10",
    rank_alignment: int = 8, # Add rank alignment parameter (e.g., 8 for efficiency)
    min_rank: int = 1,       # Minimum rank allowed
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Loads a Llama model, applies Kronecker-factored SVD compression using
    per-layer ranks derived from a sensitivity dictionary, and saves the result.
    """
    kron_factors_path = Path(kron_factors_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    original_dtype = next(model.parameters()).dtype
    logging.info(f"Model loaded with dtype {original_dtype}")
    model.to(device)
    logging.info(f"Model moved to {device}")

    total_params_before = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters before compression: {total_params_before}")


    linear_layers_to_compress = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Keep lm_head check consistent with sensitivity dict keys
            # If lm_head is in sensitivity_dict, it should be processed.
            # If not, it won't be found in the dict later anyway.
            linear_layers_to_compress[name] = module
            # if 'lm_head' not in name: # Modify this filter if lm_head is in your sensitivity dict
            #      linear_layers_to_compress[name] = module
            # else:
            #     logging.info(f"Excluding {name} based on name filter (lm_head)")


    logging.info(f"Found {len(linear_layers_to_compress)} linear layers to potentially compress.")

    layers_processed_count = 0
    layers_failed_count = 0
    layers_skipped_count = 0

    for layer_name, layer_module in tqdm(linear_layers_to_compress.items(), desc="Compressing Layers"):

        if layer_name not in sensitivity_dict:
            logging.warning(f"Skipping layer {layer_name}: Not found in sensitivity dictionary.")
            layers_skipped_count += 1
            continue

        param_ratio = sensitivity_dict[layer_name]
        if param_ratio >= 1.0: # Treat ratio >= 1 as no compression
             logging.info(f"Skipping layer {layer_name}: Sensitivity ratio >= 1.0 ({param_ratio:.2f}). No compression.")
             layers_skipped_count +=1
             continue # Skip SVD if ratio is 1 or more


        in_features = layer_module.in_features
        out_features = layer_module.out_features
        target_rank = calculate_rank_from_ratio(
            in_features=in_features,
            out_features=out_features,
            param_ratio=param_ratio,
            rank_alignment=rank_alignment,
            min_rank=min_rank
        )
        logging.info(f"Layer: {layer_name} | Ratio: {param_ratio:.3f} -> Target Rank: {target_rank} (Align: {rank_alignment})")

        # --- Load Factors ---
        factor_filename = kron_factors_path / f"{layer_name.replace('.', '_')}.safetensors"

        print ("factor_filename", factor_filename, factor_filename.exists())
        logging.info(f"factor_filename: {factor_filename}")
        logging.info(f"exists: {factor_filename.exists()}")
        
        if not factor_filename.exists():
            logging.warning(f"Skipping layer {layer_name}: Factor file not found at {factor_filename}")
            layers_skipped_count += 1
            continue

        # --- Perform Factorization ---
        logging.debug(f"Processing layer: {layer_name} with rank {target_rank}") # Use debug level
        print (f"Layer: {layer_name} | Ratio: {param_ratio:.3f} -> Target Rank: {target_rank} (Align: {rank_alignment})")
        try:
            # Load factors
            factors = load_file(factor_filename)
            # Ensure factors are float32 numpy arrays for Cholesky/SVD

            XF = factors['YF'].to(dtype=torch.float32).cpu().numpy()
            YF = factors['XF'].to(dtype=torch.float32).cpu().numpy()

            # Perform factorization
            # Pass the calculated target_rank
            # Ensure factorize_layer_kron_svd uses float32 internally for stability
            # and returns layers on the correct device/dtype
            cur_dtype = layer_module.weight.dtype
            print ("layer module", layer_module)
            factorized_sequential = factorize_layer_kron_svd(
                layer=layer_module, # Pass the layer already on the target device
                XF=XF,
                YF=YF,
                rank=target_rank, # Use calculated rank
                device=device,       # New layers will be on the target device
                dtype=cur_dtype # Match the layer's current dtype on device
                # Add regularization parameters if needed: reg_alpha, max_reg_tries
            )

            print ("factorized_sequential", factorized_sequential, flush = True)
            logging.info(f"factorized_sequential '{factorized_sequential}')")
            # Replace the original layer
            if layer_name.endswith(".weight"):
                layer_name = layer_name.rsplit(".weight", 1)[0]
            find_and_replace_layer(model, layer_name, factorized_sequential)
            layers_processed_count += 1

        except Exception as e:
            logging.error(f"Failed to process layer {layer_name}: {e}", exc_info=True) # Log traceback
            layers_failed_count += 1
            # Decide whether to continue or stop on failure
            # continue

    logging.info(
        f"Compression finished. "
        f"Processed: {layers_processed_count}, "
        f"Skipped (Ratio>=1 or No Sensitivity/Factors): {layers_skipped_count}, "
        f"Failed: {layers_failed_count}"
    )
    total_params_after = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters after compression: {total_params_after}")
    logging.info(f"C rate : {total_params_after/total_params_before}")
    
    logging.info(f"Saving compressed model to {output_dir}")
    #model.save_pretrained(output_dir)
    #tokenizer.save_pretrained(output_dir)
    logging.info("Compressed model and tokenizer saved.")
    return model, tokenizer

def load_sensitivity_dict(path: str) -> Dict[str, float]:
    """Load sensitivity dictionary from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
    
class IndexDataset(Dataset):
    """Simple dataset class for indexed tensors."""
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def get_test_dataset(name: str, tokenizer, seq_len: int = 2048) -> IndexDataset:
    """Prepare test dataset for perplexity evaluation."""
    def process_data(samples, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    
    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        return process_data(test_data, 'text')
    if 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        return process_data(test_data, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        return process_data(test_data[0:2000], 'text')
    else:
        raise ValueError(f"Unknown dataset: {name}")

@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module, 
    dataset: IndexDataset, 
    batch_size: int = 4,
    device: str = "cuda"
) -> float:
    """
    Evaluate model perplexity on a dataset.
    
    Args:
        model: The model to evaluate
        dataset: Dataset of tokenized inputs
        batch_size: Batch size for evaluation
        device: Device for evaluation
        
    Returns:
        Perplexity score (lower is better)
    """
    model.to(device)
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    nlls = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = batch.to(device)

        outputs = model(batch, use_cache=False)
        logits = outputs.logits
        
        if not torch.isfinite(logits).all():
            logging.warning("Skipping batch with non-finite logits")
            continue
            
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        nlls.append(loss)

    print("nlls.shape", nlls[0].shape)
    print("Mean NLL:", torch.cat(nlls, dim=-1).mean().item())
    
    nll = torch.cat(nlls, dim=-1).mean()
    ppl = torch.exp(nll).item()
    
    return ppl

def evaluate_model(
    model: nn.Module,
    tokenizer,
    datasets: List[str] = ['wikitext2', 'ptb', 'c4'],
    seq_len: int = 2048,
    batch_size: int = 4,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model on multiple datasets.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for preprocessing
        datasets: List of dataset names to evaluate on
        seq_len: Maximum sequence length
        batch_size: Batch size for evaluation
        device: Device for evaluation
        
    Returns:
        Dictionary mapping dataset names to perplexity scores
    """
    # Save initial model state
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(device)
    
    results = {}
    for dataset_name in datasets:
        logging.info(f"Evaluating on {dataset_name}")
        try:
            # Prepare dataset
            test_dataset = get_test_dataset(dataset_name, tokenizer, seq_len)
            
            # Evaluate perplexity
            ppl = evaluate_perplexity(
                model, 
                test_dataset, 
                batch_size=batch_size,
                device=device
            )
            
            results[dataset_name] = ppl
            logging.info(f"{dataset_name} perplexity: {ppl:.4f}")
            
        except Exception as e:
            logging.error(f"Failed to evaluate on {dataset_name}: {e}")
            results[dataset_name] = float('nan')
    
    # Restore model state
    model.config.use_cache = use_cache
    
    return results

if __name__ == "__main__":
    MODEL_NAME = "unsloth/llama-2-7b-chat" 
    KRON_FACTORS_DIR = "/home/jovyan/shares/SR004.nfs2/chekalina/FisherKronecker/grads_output/llama-2-7b-chat/fisher_factors_output_1404"
    OUTPUT_DIR = f"./{MODEL_NAME.split('/')[-1]}-kron-svd-sensitive_10"
    RANK_ALIGNMENT = 8
    MIN_RANK = 1
    
    device="cuda:0"
    # Compress model
    kron_factors_path = Path(KRON_FACTORS_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    original_dtype = next(model.parameters()).dtype
    logging.info(f"Model loaded with dtype {original_dtype}")
    model.to(device)
    logging.info(f"Model moved to {device}")

    total_params_before = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters before compression: {total_params_before}")


    dct ={}
    
    for layer_name, layer_module in model.named_modules():
        if isinstance(layer_module, torch.nn.Linear) and layer_name == "model.layers.0.self_attn.q_proj":
            try:
                layer_number = int(layer_name.split('.')[2])
            except:
                continue

            layer_results = {}

            for param_ratio in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                logging.info(layer_name, param_ratio)
    
                in_features = layer_module.in_features
                out_features = layer_module.out_features
                target_rank = calculate_rank_from_ratio(
                    in_features=in_features,
                    out_features=out_features,
                    param_ratio=param_ratio,
                    rank_alignment=RANK_ALIGNMENT,
                    min_rank=MIN_RANK
                )
                logging.info(f"Layer: {layer_name} | Ratio: {param_ratio:.3f} -> Target Rank: {target_rank} (Align: {RANK_ALIGNMENT})")
        
                # --- Load Factors ---
                factor_filename = kron_factors_path / f"{layer_name.replace('.', '_')}.safetensors"
        
                print ("factor_filename", factor_filename, factor_filename.exists())
                logging.info(f"factor_filename: {factor_filename}")
                logging.info(f"exists: {factor_filename.exists()}")
                
                if not factor_filename.exists():
                    logging.warning(f"Skipping layer {layer_name}: Factor file not found at {factor_filename}")
                    layers_skipped_count += 1
                    continue
        
                # --- Perform Factorization ---
                logging.debug(f"Processing layer: {layer_name} with rank {target_rank}") # Use debug level
                print (f"Layer: {layer_name} | Ratio: {param_ratio:.3f} -> Target Rank: {target_rank} (Align: {RANK_ALIGNMENT})")
                try:
                    # Load factors
                    factors = load_file(factor_filename)
                    # Ensure factors are float32 numpy arrays for Cholesky/SVD
                    XF = factors['XF'].to(dtype=torch.float32).cpu().numpy()
                    YF = factors['YF'].to(dtype=torch.float32).cpu().numpy()
                    
                    # Perform factorization
                    # Pass the calculated target_rank
                    # Ensure factorize_layer_kron_svd uses float32 internally for stability
                    # and returns layers on the correct device/dtype
                    cur_dtype = layer_module.weight.dtype
                    print ("layer module", layer_module)
                    factorized_sequential = factorize_layer_kron_svd(
                        layer=layer_module, # Pass the layer already on the target device
                        XF=XF,
                        YF=YF,
                        rank=target_rank, # Use calculated rank
                        device=device,       # New layers will be on the target device
                        dtype=cur_dtype # Match the layer's current dtype on device
                        # Add regularization parameters if needed: reg_alpha, max_reg_tries
                    )
        
                    print ("factorized_sequential", factorized_sequential, flush = True)
                    logging.info(f"factorized_sequential '{factorized_sequential}')")
                    # Replace the original layer
                    if layer_name.endswith(".weight"):
                        layer_name = layer_name.rsplit(".weight", 1)[0]
                    find_and_replace_layer(model, layer_name, factorized_sequential)
    
                except Exception as e:
                    logging.error(f"Failed to process layer {layer_name}: {e}", exc_info=True) # Log traceback
                    # Decide whether to continue or stop on failure
                    # continue
                logging.info(f"Compression finished. ")
        
                evaluation_results = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    datasets=['wikitext2', 'ptb'],  # Limited for testing
                    batch_size=8,
                    device="cuda:0",
                )
        
                # Print evaluation results
                logging.info(f"Evaluation results:")
                res = []
                for dataset, ppl in evaluation_results.items():
                    logging.info(f"  {dataset}: {ppl:.4f}")
                    res.append(ppl)
                layer_results[str(param_ratio)] = res  
                model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
                # Store all results for this layer
                dct[layer_name] = layer_results
                
                # Save intermediate results to JSON file after completing each layer
                json_path = "/home/jovyan/shares/SR004.nfs2/chekalina/FisherKronecker/llama/compression_sensitivity_results_FIXED.json"
                logging.info(f"Saving intermediate results to {json_path}")
                # First try to load existing results if any

                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                    # Update with new results
                    existing_results.update(dct)
                    # Write back the combined results
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(existing_results, f)
                else:
                    # First write
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(dct, f)
                
    print("\n\n\nFinal results dictionary:")
    print(dct)
    
    # Save final results to JSON file
    with open("/home/jovyan/shares/SR004.nfs2/chekalina/FisherKronecker/llama/compression_sensitivity_results_FIXED.json", "w", encoding="utf-8") as f:
        json.dump(dct, f)
