import argparse
import gc
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_positive_definite(matrix: np.ndarray) -> bool:
    """Checks if a matrix is positive definite using Cholesky decomposition."""
    if not isinstance(matrix, np.ndarray):
        logging.error(f"Input to is_positive_definite must be a numpy array, got {type(matrix)}")
        return False
    if matrix.size == 0:
        logging.warning("Input matrix is empty.")
        return False
    if not np.all(np.isfinite(matrix)):
        logging.warning("Input matrix contains non-finite values.")
        return False
    if matrix.shape[0] != matrix.shape[1]:
         logging.warning(f"Input matrix is not square ({matrix.shape}).")
         return False
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def calculate_rank_from_ratio(
    in_features: int,
    out_features: int,
    param_ratio: float,
    rank_alignment: int = 1,
    min_rank: int = 1,
) -> int:
    """Calculates SVD rank 'r' for a target parameter ratio."""
    if param_ratio <= 0: return min_rank
    if param_ratio >= 1: return min(out_features, in_features)
    original_params = in_features * out_features
    if original_params == 0: return min_rank
    target_params = original_params * param_ratio
    denominator = in_features + out_features
    if denominator == 0: return min_rank
    rank_float = target_params / denominator
    if rank_alignment > 1:
        rank = int(math.ceil(rank_float / rank_alignment)) * rank_alignment
    else:
        rank = int(math.ceil(rank_float))
    max_rank = min(out_features, in_features)
    rank = max(min_rank, min(rank, max_rank))
    return rank

def regularize_factor_inplace(factor_reg, factor_orig, diag_mean, max_tries, initial_alpha, alpha_incr, eye_mat, name):
    """Helper to regularize a factor until positive definite."""
    current_alpha = initial_alpha
    for i in range(max_tries + 1):
        if is_positive_definite(factor_reg):
            logging.debug(f"  Factor {name} PD (alpha={current_alpha:.2e})")
            return True
        if i == max_tries:
            logging.error(f"Failed to regularize {name} after {max_tries} attempts.")
            return False
        logging.debug(f"  Regularizing {name} (try {i+1}, alpha={current_alpha:.2e})")
        factor_reg[:] = (1.0 - current_alpha) * factor_orig + current_alpha * eye_mat * diag_mean
        current_alpha += alpha_incr
    return False

def prepare_kron_svd_components(
    XF: np.ndarray, YF: np.ndarray, out_features: int, in_features: int,
    reg_alpha: float = 1e-1, max_reg_tries: int = 10, alpha_increase_factor: float = 1e-1
) -> Optional[Dict[str, np.ndarray]]:
    """
    Regularizes factors, computes Cholesky decompositions and inverses.
    Done ONCE per layer before iterating through ranks.

    Returns:
        Dictionary with 'X_chol', 'Y_chol', 'inv_X_chol', 'inv_Y_chol_T'
        or None if preparation fails.
    """
    logging.debug("Preparing KronSVD components (regularization, Cholesky, inverse)...")
    if not (isinstance(YF, np.ndarray) and YF.shape == (out_features, out_features)):
         logging.error(f"Y factor shape mismatch or type. Expected ({out_features},{out_features}), got {YF.shape if isinstance(YF, np.ndarray) else type(YF)}")
         return None
    if not (isinstance(XF, np.ndarray) and XF.shape == (in_features, in_features)):
        logging.error(f"X factor shape mismatch or type. Expected ({in_features},{in_features}), got {XF.shape if isinstance(XF, np.ndarray) else type(XF)}")
        return None

    XF_reg = XF.copy()
    YF_reg = YF.copy()

    try:
        eye_X = np.eye(in_features, dtype=np.float32)
        diag_mean_X = max(np.mean(np.diag(XF)), 1e-6)
        success_x = regularize_factor_inplace(XF_reg, XF, diag_mean_X, max_reg_tries, reg_alpha, alpha_increase_factor, eye_X, "XF")

        eye_Y = np.eye(out_features, dtype=np.float32)
        diag_mean_Y = max(np.mean(np.diag(YF)), 1e-6)
        success_y = regularize_factor_inplace(YF_reg, YF, diag_mean_Y, max_reg_tries, reg_alpha, alpha_increase_factor, eye_Y, "YF")

        if not success_x or not success_y:
             raise RuntimeError("Regularization failed for one or both factors.")

    except Exception as e:
         logging.error(f"Error during regularization: {e}", exc_info=True)
         return None

    try:
        X_chol = np.linalg.cholesky(XF_reg)
        Y_chol = np.linalg.cholesky(YF_reg)
        logging.debug("  Cholesky decomposition successful.")
    except np.linalg.LinAlgError as e:
        logging.error(f"ERROR: Cholesky failed after regularization: {e}")
        return None

    try:
        inv_X_chol = np.linalg.inv(X_chol)
        inv_Y_chol_T = np.linalg.inv(Y_chol.T)
        logging.debug("  Cholesky factor inverses computed.")
    except np.linalg.LinAlgError as e:
        logging.error(f"ERROR: Failed to invert Cholesky factors: {e}")
        return None

    return {
        "X_chol": X_chol,
        "Y_chol": Y_chol,
        "inv_X_chol": inv_X_chol,
        "inv_Y_chol_T": inv_Y_chol_T
    }


def build_factored_layer_from_components(
    layer: nn.Linear,
    components: Dict[str, np.ndarray],
    rank: int,
    device: torch.device,
    dtype: torch.dtype
) -> Optional[nn.Sequential]:
    """
    Builds the factorized nn.Sequential layer using precomputed components
    and the target rank. Performs the SVD step.
    """
    logging.debug(f"Building factored layer with rank {rank}...")
    W = layer.weight.data.to(device='cpu', dtype=torch.float32).numpy()
    bias = layer.bias.data.to(device='cpu', dtype=torch.float32).numpy() if layer.bias is not None else None
    out_features, in_features = W.shape

    X_chol = components["X_chol"]
    Y_chol = components["Y_chol"]
    inv_X_chol = components["inv_X_chol"]
    inv_Y_chol_T = components["inv_Y_chol_T"]

    try:
        W_tilde = Y_chol.T @ W @ X_chol
        U_hat, S_hat, Vt_hat = np.linalg.svd(W_tilde, full_matrices=False)
        logging.debug(f"  SVD complete for rank {rank}. Singular values (top 5): {S_hat[:5]}")
    except Exception as e:
        logging.error(f"ERROR: SVD failed on transformed matrix for rank {rank}: {e}")
        return None

    max_svd_rank = len(S_hat)
    effective_rank = min(rank, max_svd_rank) if rank > 0 else max_svd_rank
    if effective_rank == 0: effective_rank = 1

    S_hat_positive = np.maximum(S_hat[:effective_rank], 1e-8)
    S_hat_sqrt = np.sqrt(S_hat_positive)
    S_hat_diag = np.diag(S_hat_sqrt)

    # W1 = S_hat_diag @ Vt_hat[:effective_rank, :] @ inv_X_chol # Rank x In
    # W2 = inv_Y_chol_T @ U_hat[:, :effective_rank] @ S_hat_diag # Out x Rank
    try:
        W1 = S_hat_diag @ Vt_hat[:effective_rank, :] @ inv_X_chol
        W2 = inv_Y_chol_T @ U_hat[:, :effective_rank] @ S_hat_diag
    except Exception as e:
        logging.error(f"Error during weight matrix calculation for rank {rank}: {e}")
        return None

    W1_tensor = torch.tensor(W1, dtype=dtype)
    W2_tensor = torch.tensor(W2, dtype=dtype)

    linear1 = nn.Linear(in_features=in_features, out_features=effective_rank, bias=False)
    linear1.weight = nn.Parameter(W1_tensor)

    linear2 = nn.Linear(in_features=effective_rank, out_features=out_features, bias=(bias is not None))
    linear2.weight = nn.Parameter(W2_tensor)
    if bias is not None:
        linear2.bias = nn.Parameter(torch.tensor(bias, dtype=dtype))

    factorized_layer = nn.Sequential(linear1, linear2)
    factorized_layer.to(device=device, dtype=dtype)
    logging.debug(f"  Factorization complete. New shapes: ({effective_rank},{in_features}), ({out_features},{effective_rank})")
    return factorized_layer

def get_calib_train_data(model_name, dataset_name, tokenizer, nsamples, seqlen=2048, seed=42, batch_size=1, dataset_cache_dir=None, file_cache_dir="cache"):
    """Loads calibration data, adapted for list of dicts output."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    safe_model_name = model_name.replace('/', '_').replace('-', '_')
    cache_dir_path = Path(file_cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    # Use more specific cache name including model
    cache_file = cache_dir_path / f"{safe_model_name}_{dataset_name}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"

    if cache_file.exists():
        logging.info(f"Loading calibration data from cache: {cache_file}")
        try:
            traindataset = torch.load(cache_file)
            # Verify format
            if isinstance(traindataset, list) and len(traindataset) > 0 and isinstance(traindataset[0], dict) and 'input_ids' in traindataset[0]:
                 logging.info(f"Loaded {len(traindataset)} samples from cache.")
                 return traindataset
            else:
                 logging.warning(f"Cache file {cache_file} has unexpected format. Regenerating.")
        except Exception as e:
            logging.warning(f"Failed to load cache file {cache_file}: {e}. Regenerating.")

    logging.info(f"Generating calibration data for {dataset_name}...")
    if dataset_name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
        text = traindata["text"]
    elif dataset_name == "ptb":
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
        text = traindata["sentence"]
    elif dataset_name == "c4":
        try:
             c4_path = Path("utils/c4-train.json")
             if not c4_path.exists():
                  raise FileNotFoundError("C4 train file not found at utils/c4-train.json. Please adjust path or download.")
             traindata = load_dataset("json", data_files=str(c4_path))['train']
             text = traindata["text"]
        except Exception as e:
             logging.error(f"Failed to load C4 dataset: {e}. Make sure it's accessible.")
             raise e
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    # Tokenize full text once (might require significant RAM)
    logging.info("Tokenizing full dataset...")
    # Use truncation and max_length defensively, though ideally the slices avoid needing it
    full_tokenized = tokenizer("\n\n".join(text), return_tensors="pt", max_length=10*1024*1024, truncation=False) # Adjust max_length based on RAM
    token_ids = full_tokenized.input_ids[0]
    logging.info(f"Full dataset tokenized. Total tokens: {token_ids.numel()}")
    del full_tokenized, text # Free memory
    gc.collect()

    traindataset = []
    indices = list(range(0, token_ids.numel() - seqlen - 1))
    random.shuffle(indices) # Shuffle indices to pick random start points

    samples_gathered = 0
    current_batch_ids = []

    for i in indices:
        j = i + seqlen
        inp = token_ids[i:j].unsqueeze(0) # Shape (1, seqlen)

        current_batch_ids.append(inp)

        if len(current_batch_ids) == batch_size:
             batch_tensor = torch.cat(current_batch_ids, dim=0) # Shape (batch_size, seqlen)
             attention_mask = torch.ones_like(batch_tensor)
             traindataset.append({"input_ids": batch_tensor, "attention_mask": attention_mask})
             samples_gathered += batch_size
             current_batch_ids = [] # Reset for next batch
             logging.debug(f"Gathered {samples_gathered}/{nsamples} samples...")

        if samples_gathered >= nsamples:
             break # Stop once we have enough samples

    if len(current_batch_ids) > 0: # Add any remaining samples
         batch_tensor = torch.cat(current_batch_ids, dim=0)
         attention_mask = torch.ones_like(batch_tensor)
         traindataset.append({"input_ids": batch_tensor, "attention_mask": attention_mask})
         samples_gathered += len(current_batch_ids)

    if samples_gathered < nsamples:
        logging.warning(f"Could only gather {samples_gathered} samples, requested {nsamples}.")

    logging.info(f"Saving {len(traindataset)} batches ({samples_gathered} total samples) to cache: {cache_file}")
    torch.save(traindataset, cache_file)
    return traindataset


@torch.no_grad()
def evaluate_perplexity(model: PreTrainedModel, input_ids: torch.Tensor, batch_size: int, device: torch.device = "cuda") -> float:
    """
    Evaluates perplexity on the provided input_ids tensor.
    Assumes input_ids is a large tensor (e.g., concatenated calibration data).
    Processes in chunks matching the model's sequence length.
    """
    model.eval()
    model.to(device)

    seq_len = model.config.max_position_embeddings # Use model's context length
    total_len = input_ids.size(0) # Assumes input_ids is (N*seq_len)

    nlls = []
    num_processed = 0

    # Process in chunks, potentially smaller than full batch_size if input_ids is not a multiple
    effective_batch_size = min(batch_size, 16) # Limit PPL batch size to avoid OOM

    # Reshape or iterate to get batches of shape [eff_batch_size, seq_len]
    # This simplified approach processes the whole concatenated tensor seq_len by seq_len
    # A more standard PPL would process independent sequences. This measures overall NLL increase.

    stride = seq_len

    logging.debug(f"Evaluating PPL with seq_len={seq_len}, stride={stride}, total_len={total_len}")

    for i in tqdm(range(0, total_len, stride * effective_batch_size), desc="Perplexity Eval", leave=False, mininterval=2.0):
        batch_input_list = []
        batch_label_list = []
        for b in range(effective_batch_size):
             start_ix = i + b * stride
             end_ix = start_ix + seq_len
             if end_ix > total_len:
                  break

             chunk = input_ids[start_ix:end_ix].unsqueeze(0) # Shape [1, seq_len]
             labels = chunk.clone()
             # labels[:, :-1] = -100 # Standard Causal LM label masking - apply if model needs it
             # If just using loss, no need to manually shift/mask usually if passing labels=

             batch_input_list.append(chunk)
             batch_label_list.append(labels)

        if not batch_input_list:
            break # No more full chunks

        batch_input = torch.cat(batch_input_list, dim=0).to(device)
        batch_label = torch.cat(batch_label_list, dim=0).to(device)


        try:
            outputs = model(batch_input, labels=batch_label, use_cache=False)
            neg_log_likelihood = outputs.loss

            if torch.isinf(neg_log_likelihood) or torch.isnan(neg_log_likelihood):
                 logging.warning(f"Infinite or NaN loss encountered in PPL batch {i // (stride*effective_batch_size)}. Skipping.")
                 continue

            nlls.append(neg_log_likelihood)
            num_processed += batch_input.numel()

        except torch.cuda.OutOfMemoryError:
            logging.error("OOM error during perplexity evaluation. Try reducing --eval_batch_size.")
            return float('nan')
        except Exception as e:
            logging.error(f"Error during perplexity evaluation: {e}", exc_info=True)
            return float('nan')


    if not nlls:
        logging.warning("No batches processed successfully for perplexity evaluation.")
        return float('nan')

    mean_nll = torch.stack(nlls).mean().item()

    if mean_nll == 0:
        logging.warning("Mean NLL is 0. This might indicate an issue.")
        return 1.0

    ppl = math.exp(mean_nll)
    logging.debug(f"Calculated PPL: {ppl:.4f} from Mean NLL: {mean_nll:.4f}")

    return ppl



@torch.no_grad()
def calib_sensitivity_ppl_kronsvd(model, calib_loader, args, use_cache=True):
    model_id = model.config._name_or_path
    safe_model_id = model_id.replace('/','_')
    cache_file = Path(args.output_dir) / f"cache_{safe_model_id}_sensitivity_kronsvd_{args.n_calib_samples}_{args.calib_dataset}.pt"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and use_cache:
        logging.info(f"Loading sensitivity results from cache: {cache_file}")
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict

    logging.info("Starting sensitivity calibration using KronSVD...")
    model.eval()
    target_device = args.device
    original_dtype = next(model.parameters()).dtype
    logging.info(f"Model device: {target_device}, dtype: {original_dtype}")

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules_to_process = [model]
    processed_modules = set()

    while modules_to_process:
        submodule = modules_to_process.pop(0)
        if submodule in processed_modules: continue
        processed_modules.add(submodule)

        module_id = id(submodule)
        is_factorized = False
        if isinstance(submodule, nn.Sequential) and len(submodule) == 2:
             if isinstance(submodule[0], nn.Linear) and isinstance(submodule[1], nn.Linear):
                  potential_rank = submodule[0].out_features
                  if potential_rank < min(submodule[0].in_features, submodule[1].out_features):
                       is_factorized = True


        if is_factorized:
             full_name = full_name_dict.get(submodule, f"Sequential_{module_id}")
             logging.debug(f"Skipping potentially already factorized Sequential block: {full_name}")
             continue

        for name, raw_module in submodule.named_children():
            if raw_module in processed_modules: continue

            if isinstance(raw_module, nn.Linear):
                 full_name = full_name_dict.get(raw_module)
                 if full_name:
                      linear_info[raw_module] = {"father": submodule, "name": name, "full_name": full_name}
                 else:
                      logging.warning(f"Could not find full name for linear layer {name} in {type(submodule).__name__}")
                 processed_modules.add(raw_module)
            elif list(raw_module.children()):
                 modules_to_process.append(raw_module)
            else:
                 processed_modules.add(raw_module)


    logging.info(f"Found {len(linear_info)} linear layers to calibrate.")

    all_input_ids = []
    logging.info(f"Loading calibration data batches...")
    for i, batch in enumerate(calib_loader):
        all_input_ids.append(batch["input_ids"])
        if i * args.calib_batch_size >= args.n_calib_samples:
             break
    if not all_input_ids:
        raise ValueError("Calibration loader did not yield any input_ids.")

    input_ids = torch.cat(all_input_ids, 0) # Shape [N_samples, seq_len]
    input_ids_flat = input_ids.view(-1) # Shape [N_samples * seq_len]
    logging.info(f"Calibration input_ids shape (samples, seq_len): {input_ids.shape}")
    logging.info(f"Flattened input_ids shape for PPL: {input_ids_flat.shape}")
    input_ids_flat = input_ids_flat.to(target_device) # Move data to device

    param_ratio_candidates = [i / 20.0 for i in range(1, 20)] # 0.05 to 0.95

    sensitivity_dict = {}
    kron_factors_dir = Path(args.kron_factors_dir)
    if not kron_factors_dir.is_dir():
        raise FileNotFoundError(f"Kronecker factors directory not found: {kron_factors_dir}")

    total_combinations = len(linear_info) * len(param_ratio_candidates)
    pbar = tqdm(total=total_combinations, desc="Calibrating Layers (KronSVD)")

    reg_alpha = getattr(args, 'reg_alpha', 1e-1)
    max_reg_tries = getattr(args, 'max_reg_tries', 10)
    alpha_increase_factor = getattr(args, 'alpha_increase_factor', 1e-1)
    eval_batch_size = getattr(args, 'eval_batch_size', 4)


    for raw_linear, info in linear_info.items():
        layer_full_name = info["full_name"]
        sensitivity_dict[layer_full_name] = {}
        logging.debug(f"Processing layer: {layer_full_name}")

        factor_filename = kron_factors_dir / f"{layer_full_name.replace('.', '_')}.safetensors"
        if not factor_filename.exists():
            logging.warning(f"Factor file not found for {layer_full_name}. Skipping.")
            for pr in param_ratio_candidates: sensitivity_dict[layer_full_name][pr] = float('nan')
            pbar.update(len(param_ratio_candidates))
            continue

        try:
            factors = load_file(factor_filename, device='cpu')
            XF = factors['XF'].to(dtype=torch.float32).numpy()
            YF = factors['YF'].to(dtype=torch.float32).numpy()

            logging.debug(f"Precomputing Kronecker Factors for {layer_full_name}.")
            precomputed_components = prepare_kron_svd_components(
                 XF, YF, raw_linear.out_features, raw_linear.in_features,
                 reg_alpha, max_reg_tries, alpha_increase_factor
            )

        except Exception as e:
            logging.error(f"Failed to load factors for {layer_full_name}: {e}. Skipping.", exc_info=True)
            for pr in param_ratio_candidates: sensitivity_dict[layer_full_name][pr] = float('nan')
            pbar.update(len(param_ratio_candidates))
            del XF, YF, factors # Clean up factors if loaded
            continue # Skip to next layer

        if precomputed_components is None:
             logging.error(f"Component preparation failed for {layer_full_name}. Skipping ratios.")
             for pr in param_ratio_candidates: sensitivity_dict[layer_full_name][pr] = float('nan')
             pbar.update(len(param_ratio_candidates))
             del XF, YF, factors
             continue

        for param_ratio in param_ratio_candidates:
            current_ppl = float('nan')
            target_rank = -1
            try:
                target_rank = calculate_rank_from_ratio(
                    raw_linear.in_features, raw_linear.out_features, param_ratio, args.rank_align)

                factorized_sequential = build_factored_layer_from_components(
                    layer=raw_linear,
                    components=precomputed_components, # Pass precomputed dict
                    rank=target_rank,
                    device=target_device,
                    dtype=original_dtype,
                )

                if factorized_sequential is None:
                     # Error occurred during build_factored_layer...
                     raise RuntimeError(f"Failed to build factorized layer for rank {target_rank}")

                setattr(info["father"], info["name"], factorized_sequential)

                current_ppl = evaluate_perplexity(model, input_ids_flat, eval_batch_size, device=target_device)
                logging.info(f"  {layer_full_name} Ratio: {param_ratio:.2f}, Rank: {target_rank}, PPL: {current_ppl:.4f}")

            except Exception as e:
                logging.error(f"Failed: {layer_full_name} (Ratio:{param_ratio:.2f}, Rank:{target_rank}): {e}", exc_info=False)
            finally:
                setattr(info["father"], info["name"], raw_linear)
                sensitivity_dict[layer_full_name][param_ratio] = current_ppl
                pbar.update(1)
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        del XF, YF, factors
        gc.collect()

    pbar.close()
    logging.info(f"Sensitivity calibration finished. Saving results to {cache_file}")
    try:
        torch.save(sensitivity_dict, cache_file)
        json_cache_file = cache_file.with_suffix('.json')
        sensitivity_dict_json = {
            layer: {float(k): (float(v) if not math.isnan(v) else None) for k, v in ratios.items()}
            for layer, ratios in sensitivity_dict.items()
        }
        with open(json_cache_file, 'w') as f:
             json.dump(sensitivity_dict_json, f, indent=2)
        logging.info(f"Results also saved in JSON format to {json_cache_file}")

    except Exception as e:
         logging.error(f"Failed to save sensitivity cache file: {e}", exc_info=True)

    del input_ids, input_ids_flat
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return sensitivity_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Run KronSVD Sensitivity Calibration")
    parser.add_argument('--model_name_or_path', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Model identifier')
    parser.add_argument('--kron_factors_dir', type=str, required=True, help='Directory containing Kronecker factor .safetensors files')
    parser.add_argument('--calib_dataset', type=str, default='wikitext2', choices=['wikitext2', 'ptb', 'c4'], help='Calibration dataset name')
    parser.add_argument('--n_calib_samples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--calib_batch_size', type=int, default=4, help='Batch size for loading calibration data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size for perplexity evaluation during calibration')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length for calibration data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for computation (e.g., "cuda", "cuda:0", "cpu")')
    parser.add_argument('--rank_align', type=int, default=8, help='Rank alignment factor')
    parser.add_argument('--output_dir', type=str, default="./sensitivity_results", help='Directory to save sensitivity results')
    parser.add_argument('--no_cache', action='store_true', help='Do not use cached sensitivity results')
    parser.add_argument('--torch_dtype', type=str, default="bfloat16", help='Data type for model loading (e.g., "float16", "bfloat16", "float32")')
    # Optional KronSVD params
    parser.add_argument('--reg_alpha', type=float, default=1e-1, help='Initial regularization alpha for KronSVD factors')
    parser.add_argument('--max_reg_tries', type=int, default=100, help='Max regularization attempts for KronSVD factors')
    parser.add_argument('--alpha_increase_factor', type=float, default=5e-2, help='Increase factor for regularization alpha per try')

    return parser.parse_args()

def main():
    args = parse_args()
    logging.info(f"Script arguments: {args}")
    device = torch.device(args.device)
    logging.info(f"Loading model: {args.model_name_or_path}")
    try:
        model_dtype = getattr(torch, args.torch_dtype)
    except AttributeError:
        logging.warning(f"Invalid torch_dtype '{args.torch_dtype}'. Defaulting to float32.")
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=model_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if tokenizer.vocab_size > model.config.vocab_size:
         logging.warning(f"Tokenizer vocab size ({tokenizer.vocab_size}) > model embedding size ({model.config.vocab_size}). Resizing model embeddings.")
         model.resize_token_embeddings(tokenizer.vocab_size)
         model.tie_weights()
    elif tokenizer.vocab_size < model.config.vocab_size:
         logging.warning(f"Tokenizer vocab size ({tokenizer.vocab_size}) < model embedding size ({model.config.vocab_size}). This might indicate a mismatch.")


    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logging.info("Tokenizer missing pad token; setting to eos token.")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        else:
            logging.info("Tokenizer missing pad token and eos token. Adding a new pad token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id


    model.to(device)
    model.eval()

    logging.info(f"Loading calibration data: {args.calib_dataset}")
    calib_data_loader = get_calib_train_data(
        model_name=args.model_name_or_path,
        dataset_name=args.calib_dataset,
        tokenizer=tokenizer,
        nsamples=args.n_calib_samples,
        seqlen=args.seq_len,
        seed=42,
        batch_size=args.calib_batch_size,
        file_cache_dir=os.path.join(args.output_dir, "calib_cache")
    )

    sensitivity_data = calib_sensitivity_ppl_kronsvd(
        model=model,
        calib_loader=calib_data_loader,
        args=args,
        use_cache=(not args.no_cache)
    )


    if sensitivity_data:
        logging.info("Sensitivity calibration complete.")
        example_layer = list(sensitivity_data.keys())[0]
        if example_layer:
             logging.info(f"Example Sensitivity for layer '{example_layer}':")
             for ratio, ppl in sensitivity_data[example_layer].items():
                  print(f"  Ratio {ratio:.2f}: PPL {ppl:.4f}")

        logging.info(f"Sensitivity results saved in: {args.output_dir}")
    else:
        logging.error("Sensitivity calibration failed to produce results.")

if __name__ == "__main__":
    main()
