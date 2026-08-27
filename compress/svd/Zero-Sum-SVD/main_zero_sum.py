import copy, datetime, json, os, sys, time
from collections import defaultdict, Counter
from contextlib import contextmanager
from typing import Dict, Optional

import torch
from tqdm import tqdm
from utils.data_utils import get_calib_train_data
from utils.model_utils import get_model_from_huggingface, get_model_from_local
from utils.eval_utils import ppl_eval
from utils.model_utils import LowRankLinear, get_layers, find_target_modules_in_layer
from utils.correction_utils import (extract_dense_weight, factorize_linear, replace_module,
    compute_dense_params, aggregate_stage_counts, run_svdllm_stage, run_gd_correction_after_truncation)
from utils.quant_utils import (quantize_target_modules_inplace, dequantize_target_modules_inplace,
    quant_dequant_roundtrip_targets_, _intermediate_eval_dtype)
from compression.args import define_compression_args, validate_args, print_args_summary, build_experiment_tag
from compression.profiling import compute_profiling_matrices, build_L_cache
from compression.gradient import recompute_importance, get_or_recompute_importance
from compression.planning import (build_module_states, stage_plan, build_stage_rank_snapshot,
    compute_revert_info, log_module_distribution)
from compression.corrections import run_between_stage_corrections
from compression.evaluation import (
    commonsense_eval,
    set_deterministic_seeds,
    _sanitize_for_json,
    _extract_commonsense_metrics,
)
from compression.truncation import apply_truncations, run_vanilla_svd_stage

# Track modules that were explicitly substituted with teacher weights
SUBSTITUTED_MODULE_KEYS = set()

# Timing infrastructure for --time_analysis mode
TIMES = defaultdict(float)
STAGE_TIMES = defaultdict(lambda: defaultdict(float))


def _sync():
    """Synchronize GPU to ensure all work is complete for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _time_block(name: str, stage: Optional[int] = None, args=None):
    """Context manager for timing code blocks with GPU synchronization."""
    if args is None or not getattr(args, "time_analysis", False):
        yield
        return
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        dt = time.perf_counter() - t0
        TIMES[name] += dt
        if stage is not None:
            STAGE_TIMES[stage][name] += dt


# ---------------------------------------------------------------------------
# Adapter for base models that don't return logits
# ---------------------------------------------------------------------------

class CausalLMFromBase(torch.nn.Module):
    """Adapter that wraps a base model and lm_head to always return logits."""
    def __init__(self, base_model: torch.nn.Module, lm_head: torch.nn.Module):
        super().__init__()
        self.model = base_model
        self.lm_head = lm_head

        # expose config if present
        if hasattr(base_model, "config"):
            self.config = base_model.config

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hs = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        logits = self.lm_head(hs)
        return type("TmpOut", (), {"logits": logits, "past_key_values": getattr(out, "past_key_values", None)})

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(*args, **kwargs)
        return None

    def gradient_checkpointing_disable(self, *args, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable(*args, **kwargs)
        return None

    @property
    def is_gradient_checkpointing(self):
        return bool(getattr(self.model, "is_gradient_checkpointing", False))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main():
    # Use modular argument parsing from compression.args
    parser = define_compression_args()
    args = parser.parse_args()

    # Set deterministic seeds for reproducibility across GPUs
    set_deterministic_seeds(args.seed)

    # Validate arguments
    validate_args(args)

    # Print algorithm arguments summary
    print_args_summary(args)

    dev = torch.device(args.DEV)
    os.makedirs(args.save_path, exist_ok=True)

    # Build experiment directory tag
    exp_tag = build_experiment_tag(args)
    exp_dir = os.path.join(args.save_path, exp_tag)
    os.makedirs(exp_dir, exist_ok=True)

    # Helper to save a stage checkpoint after truncation
    def _save_stage_checkpoint(model, tokenizer, exp_dir: str, stage_idx: int):
        ckpt_suffix = "_int8.pt" if getattr(args, "quantize_8_bit", False) else "_fp16.pt"
        ckpt_name = f"stage{stage_idx:02d}_after_truncation{ckpt_suffix}"
        ckpt_path = os.path.join(exp_dir, ckpt_name)

        if getattr(args, "quantize_8_bit", False):
            print("🔢 Quantizing (targets only) for intermediate checkpoint storage...")

            # quantize targets, then save the quantized snapshot
            model = model.to(dev)
            mapping_info = quantize_target_modules_inplace(args, model, dev)

            save_blob = {
                "model": model.cpu(),
                "tokenizer": tokenizer,
                "model_name": args.model,
                "mapping_info": mapping_info,
                "quantize_8_bit": True,
            }
            torch.save(save_blob, ckpt_path)
            print(f"💾 Saved quantized checkpoint to {ckpt_path}")

            # restore model back for continued compute
            model = model.to(dev)
            tmp = {"model": model, "model_name": args.model, "mapping_info": mapping_info}
            dequantize_target_modules_inplace(tmp, dev=dev)
            model.to(dtype=torch.bfloat16)
            return model

        # original behavior when quant is off
        torch.save({"model": model.cpu(), "tokenizer": tokenizer}, ckpt_path)
        print(f"💾 Saved checkpoint after truncation for stage {stage_idx} to {ckpt_path}")
        return model

    # Save all experiment configuration to JSON file with timestamp
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    config_filename = f"experiment_config_{timestamp_str}.json"
    config_path = os.path.join(exp_dir, config_filename)
    config_data = vars(args).copy()  # Convert namespace to dict

    # Add additional metadata
    config_data["experiment_timestamp"] = timestamp.isoformat()
    config_data["experiment_directory"] = exp_tag
    config_data["command_line"] = " ".join(sys.argv)
    config_data["python_version"] = sys.version

    # Convert non-serializable values to strings
    for key, value in config_data.items():
        if isinstance(value, torch.device):
            config_data[key] = str(value)
        elif value is None:
            config_data[key] = "null"

    if not getattr(args, "time_analysis", False):
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config_data, config_file, indent=2, default=str)
        print(f"💾 Saved experiment configuration to {config_path}")

    if args.model_path:
        model, tokenizer = get_model_from_local(args.model_path)
    else:
        model, tokenizer = get_model_from_huggingface(args.model)

    model = model.to(dev).eval()
    if getattr(args, "quantize_8_bit", False):
        model = model.to(dtype=torch.bfloat16)
        print("🔢 Running in bfloat16 for 8-bit quantization mode")
    if not hasattr(model, "seqlen"):
        setattr(model, "seqlen", args.model_seq_len)

    # Build and cache the base calibration data once
    base_cali_data = get_calib_train_data(
        args.dataset,
        tokenizer,
        args.nsamples,
        seqlen=args.model_seq_len,
        seed=args.seed,
        split=getattr(args, "dataset_split", "train"),
        text_fields=getattr(args, "dataset_text_fields", None),
    )
    # store as a list for reuse across stages
    args.base_calib_data = list(base_cali_data)

    # Debug: hash calibration data to verify consistency across machines
    import hashlib
    calib_hash = hashlib.md5(
        torch.cat([b["input_ids"].flatten() for b in args.base_calib_data]).numpy().tobytes()
    ).hexdigest()
    print(f"🔍 Calibration data hash: {calib_hash}")

    target_storage_dtype = torch.float16
    evaluation_dtype = torch.float32

    init_dense = compute_dense_params(args.model, model)
    print(f"📏 Initial dense parameter count: {init_dense:,}\n" )

    # Initial commonsense evaluation on uncompressed model
    initial_cs_results = None
    if (not getattr(args, "time_analysis", False)) and getattr(args, "initial_evaluate_commonsense", False):
        tasks_csv = getattr(args, "commonsense_tasks", "arc_easy,arc_challenge,openbookqa,winogrande,hellaswag,piqa")
        print(f"\n🧠 Evaluating commonsense on UNCOMPRESSED model (fp32): {tasks_csv}")
        prev_dtype = next(model.parameters()).dtype
        model = model.to(dtype=evaluation_dtype)  # fp32
        initial_cs_results = commonsense_eval(model, tokenizer, tasks_csv=tasks_csv, device=args.DEV)
        model = model.to(dtype=prev_dtype)  # restore

        # Print a compact summary
        try:
            res = initial_cs_results.get("results", {})
            print("\n🧠 Initial (uncompressed) commonsense results:")
            for task_name, metrics in res.items():
                if isinstance(metrics, dict):
                    keys = [k for k in ["acc", "acc_norm", "exact_match", "mc1", "mc2"] if k in metrics]
                    if keys:
                        msg = ", ".join([f"{k}={metrics[k]:.4f}" for k in keys if isinstance(metrics[k], (int, float))])
                        print(f"   {task_name}: {msg}")
                    else:
                        print(f"   {task_name}: {metrics}")
                else:
                    print(f"   {task_name}: {metrics}")
        except Exception:
            print("   Initial commonsense evaluation finished")
            print(initial_cs_results)

        # Save initial (uncompressed) commonsense results to JSON
        if initial_cs_results is not None:
            uncompressed_cs_path = os.path.join(exp_dir, "uncompressed_commonsense_results.json")
            uncompressed_cs_data = {
                "model": args.model,
                "evaluation_type": "uncompressed_baseline",
                "tasks": tasks_csv,
                "results": _extract_commonsense_metrics(initial_cs_results),
                "raw_results": _sanitize_for_json(initial_cs_results.get("results", {})),
            }
            with open(uncompressed_cs_path, "w", encoding="utf-8") as f:
                json.dump(uncompressed_cs_data, f, indent=2)
            print(f"💾 Saved uncompressed commonsense results to: {uncompressed_cs_path}")

    # Initial PPL evaluation on uncompressed model
    if (not getattr(args, "time_analysis", False)) and getattr(args, "initial_eval_ppl", False):
        datasets = [d.strip() for d in args.final_eval_datasets.split(",") if d.strip()]
        print(f"\n📊 Evaluating PPL on UNCOMPRESSED model: {datasets}")

        prev_dtype = next(model.parameters()).dtype
        model = model.to(dtype=torch.float32)  # Use fp32 for baseline accuracy

        initial_ppl_results = ppl_eval(
            model, tokenizer,
            datasets=datasets,
            model_seq_len=args.model_seq_len,
            batch_size=4,
            device=args.DEV,
        )

        model = model.to(dtype=prev_dtype)  # Restore original dtype

        # Print results
        print("\n📊 Initial (uncompressed) PPL results:")
        for ds_name, ppl_val in initial_ppl_results.items():
            print(f"   {ds_name}: {ppl_val:.4f}")

        # Save to JSON
        uncompressed_ppl_path = os.path.join(exp_dir, "uncompressed_ppl_results.json")
        uncompressed_ppl_data = {
            "model": args.model,
            "evaluation_type": "uncompressed_baseline",
            "datasets": datasets,
            "perplexity": initial_ppl_results,
        }
        with open(uncompressed_ppl_path, "w", encoding="utf-8") as f:
            json.dump(uncompressed_ppl_data, f, indent=2)
        print(f"💾 Saved uncompressed PPL results to: {uncompressed_ppl_path}")

    teacher_weights: Dict[str, torch.Tensor] = {}
    teacher_biases: Dict[str, Optional[torch.Tensor]] = {}
    for i, layer in enumerate(get_layers(args.model, model)):
        # Use find_target_modules_in_layer to properly handle both nn.Linear and LowRankLinear
        for name, mod in find_target_modules_in_layer(layer).items():
            key = f"layer{i}.{name}"
            teacher_weights[key] = extract_dense_weight(mod).detach().cpu()
            # For LowRankLinear, we want the bias from u_proj if it exists
            if isinstance(mod, LowRankLinear):
                bias = mod.u_proj.bias
            else:
                bias = getattr(mod, "bias", None)
            teacher_biases[key] = None if bias is None else bias.detach().cpu()

    # Print target layer names if requested
    if getattr(args, "print_target_layers", False):
        print(f"\n📋 Target layers for compression ({len(teacher_weights)} modules):")
        for idx, key in enumerate(sorted(teacher_weights.keys())):
            print(f"   {idx+1:3d}. {key}")
        print()

    # Print all linear layers (target and non-target) if requested
    if getattr(args, "print_all_linear_layers", False):
        target_keys = set(teacher_weights.keys())
        all_linear = []
        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Linear, LowRankLinear)):
                # Check if this is a target layer
                is_target = any(name.endswith(tgt) for tgt in target_keys) or name in target_keys
                # Try to match layer-based key format
                for tkey in target_keys:
                    if tkey.split(".")[-1] in name:
                        is_target = True
                        break
                shape = tuple(mod.weight.shape) if hasattr(mod, 'weight') else "N/A"
                all_linear.append((name, type(mod).__name__, shape, is_target))

        print(f"\n📋 All linear layers in model ({len(all_linear)} modules):")
        print(f"   {'#':>3}  {'Name':<60} {'Type':<15} {'Shape':<20} {'Target'}")
        print(f"   {'-'*3}  {'-'*60} {'-'*15} {'-'*20} {'-'*6}")
        for idx, (name, typ, shape, is_target) in enumerate(all_linear):
            marker = "✓" if is_target else ""
            print(f"   {idx+1:3d}. {name:<60} {typ:<15} {str(shape):<20} {marker}")
        print()

    # Initialize importance and profiling as None - will compute on-the-fly
    importance = None
    profiling_mat = None
    svd_cache = None
    L_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    states = {}

    # For SVDLLM/vanilla_svd selection mode: skip importance entirely (uses original SVDLLM logic)
    # For other modes: compute importance upfront
    if args.selection_mode in ("svdllm", "vanilla_svd"):
        print(f"\n⚙️  {args.selection_mode} selection mode: skipping importance/profiling computation")
    else:
        print("\n⚙️  Computing importance and profiling on the fly...")
        with _time_block("importance+profiling", stage=0, args=args):
            importance, profiling_mat, L_cache, svd_cache = recompute_importance(args, model, tokenizer, dev)

    total_dense_baseline = init_dense
    remap_target_retained = int((1.0 - float(args.global_prune_ratio)) * float(total_dense_baseline))
    print(f"🔢 Total dense params under consideration: {total_dense_baseline:,}")
    if getattr(args, "remap", False):
        print(f"🎯 Remap target retained: {remap_target_retained:,} (using max(m,n) cost)")
    print(f"🎯 Per stage target drop ratio: {args.global_prune_ratio:.3f}\n")

    for stage_idx in range(1, args.num_stages + 1):

        # ===== SVDLLM SPECIAL PATH =====
        if args.selection_mode == "svdllm":
            with _time_block("svdllm_stage", stage=stage_idx, args=args):
                model = run_svdllm_stage(
                    args, model, tokenizer, dev,
                    stage_idx=stage_idx,
                    total_dense_baseline=total_dense_baseline,
                    compute_profiling_matrices_fn=compute_profiling_matrices,
                    build_L_cache_fn=build_L_cache,
                    ppl_eval_fn=ppl_eval,
                )
            # Eval BEFORE gd_correction (when both flags are set)
            if (not getattr(args, "time_analysis", False)) and getattr(args, "gd_correction_mode", False) and args.eval_ppl_per_outer_stage:
                print(f"\n   📊 [BEFORE gd_correction] Evaluating perplexity on wikitext2 + c4 + ptb (outer stage {stage_idx}/{args.num_stages})...")
                if getattr(args, "quantize_8_bit", False):
                    quant_dequant_roundtrip_targets_(args, model, dev)
                target_eval_dtype = _intermediate_eval_dtype(args)
                model_dtype = next(model.parameters()).dtype
                if model_dtype != target_eval_dtype:
                    model = model.to(dtype=target_eval_dtype)
                ppl_before_gd = ppl_eval(
                    model, tokenizer,
                    datasets=['wikitext2', 'c4', 'ptb'],
                    model_seq_len=args.model_seq_len,
                    batch_size=4,
                    device=args.DEV,
                )
                for ds_name, ppl_val in ppl_before_gd.items():
                    print(f"   ✓ [BEFORE gd_correction] {ds_name} PPL: {ppl_val:.4f}")
                if model_dtype != target_eval_dtype:
                    model = model.to(dtype=model_dtype)
            # Optional gradient direction correction after SVDLLM truncation
            if getattr(args, "gd_correction_mode", False):
                with _time_block("gd_correction", stage=stage_idx, args=args):
                    run_gd_correction_after_truncation(
                        args, model, tokenizer, dev,
                        stage_idx=stage_idx,
                        inner_idx=1,
                    )
            current_dense = compute_dense_params(args.model, model)
            print(
                f"   → Dense params after SVDLLM stage {stage_idx}: "
                f"{current_dense:,} "
                f"(removed {total_dense_baseline - current_dense:,} from baseline)"
            )
            # Evaluate perplexity AFTER gd_correction / BEFORE pull_subspace (wikitext2 + c4 + ptb)
            if (not getattr(args, "time_analysis", False)) and args.eval_ppl_per_outer_stage:
                gd_label = "[AFTER gd_correction] " if getattr(args, "gd_correction_mode", False) else ""
                print(f"\n   📊 {gd_label}[BEFORE pull_subspace] Evaluating perplexity on wikitext2 + c4 + ptb (outer stage {stage_idx}/{args.num_stages})...")
                if getattr(args, "quantize_8_bit", False):
                    quant_dequant_roundtrip_targets_(args, model, dev)
                target_eval_dtype = _intermediate_eval_dtype(args)
                model_dtype = next(model.parameters()).dtype
                if model_dtype != target_eval_dtype:
                    model = model.to(dtype=target_eval_dtype)
                outer_stage_ppl = ppl_eval(
                    model, tokenizer,
                    datasets=['wikitext2', 'c4', 'ptb'],
                    model_seq_len=args.model_seq_len,
                    batch_size=4,
                    device=args.DEV,
                )
                for ds_name, ppl_val in outer_stage_ppl.items():
                    print(f"   ✓ {gd_label}[BEFORE pull_subspace] {ds_name} PPL: {ppl_val:.4f}")
                if model_dtype != target_eval_dtype:
                    model = model.to(dtype=model_dtype)
            # Save checkpoint after truncation, before between-stage corrections
            if (not getattr(args, "time_analysis", False)) and getattr(args, "save_after_truncation", False):
                model_dtype = next(model.parameters()).dtype
                if model_dtype != torch.float16:
                    model = model.half()
                model = _save_stage_checkpoint(model, tokenizer, exp_dir, stage_idx)
                model = model.to(dev)
            is_final_stage = (stage_idx == args.num_stages)
            with _time_block("between_stage_corrections", stage=stage_idx, args=args):
                run_between_stage_corrections(
                    args, model, tokenizer, dev, teacher_weights, stage_idx, is_final_stage,
                    substituted_keys=SUBSTITUTED_MODULE_KEYS,
                )
            # Evaluate perplexity AFTER pull_subspace (wikitext2 only)
            if (not getattr(args, "time_analysis", False)) and args.eval_ppl_per_outer_stage:
                print(f"\n   📊 [AFTER pull_subspace] Evaluating perplexity on wikitext2 (outer stage {stage_idx}/{args.num_stages})...")
                if getattr(args, "quantize_8_bit", False):
                    quant_dequant_roundtrip_targets_(args, model, dev)
                target_eval_dtype = _intermediate_eval_dtype(args)
                model_dtype = next(model.parameters()).dtype
                if model_dtype != target_eval_dtype:
                    model = model.to(dtype=target_eval_dtype)
                outer_stage_ppl_after = ppl_eval(
                    model, tokenizer,
                    datasets=['wikitext2'],
                    model_seq_len=args.model_seq_len,
                    batch_size=4,
                    device=args.DEV,
                )
                for ds_name, ppl_val in outer_stage_ppl_after.items():
                    print(f"   ✓ [AFTER pull_subspace] {ds_name} PPL: {ppl_val:.4f}")
                if model_dtype != target_eval_dtype:
                    model = model.to(dtype=model_dtype)
            continue

        # ===== VANILLA SVD SPECIAL PATH =====
        if args.selection_mode == "vanilla_svd":
            with _time_block("vanilla_svd_stage", stage=stage_idx, args=args):
                model = run_vanilla_svd_stage(
                    args, model, tokenizer, dev,
                    stage_idx=stage_idx,
                )

            # Optional gd correction after truncation
            if getattr(args, "gd_correction_mode", False):
                with _time_block("gd_correction", stage=stage_idx, args=args):
                    run_gd_correction_after_truncation(
                        args, model, tokenizer, dev,
                        stage_idx=stage_idx,
                        inner_idx=1,
                    )

            current_dense = compute_dense_params(args.model, model)
            print(
                f"   → Dense params after vanilla_svd stage {stage_idx}: "
                f"{current_dense:,} "
                f"(removed {total_dense_baseline - current_dense:,} from baseline)"
            )

            if (not getattr(args, "time_analysis", False)) and args.eval_ppl_per_outer_stage:
                print(f"\n   📊 Evaluating perplexity on wikitext2 + c4 + ptb (outer stage {stage_idx}/{args.num_stages})...")
                if getattr(args, "quantize_8_bit", False):
                    quant_dequant_roundtrip_targets_(args, model, dev)
                target_eval_dtype = _intermediate_eval_dtype(args)
                model_dtype = next(model.parameters()).dtype
                if model_dtype != target_eval_dtype:
                    model = model.to(dtype=target_eval_dtype)
                outer_stage_ppl = ppl_eval(
                    model, tokenizer,
                    datasets=['wikitext2', 'c4', 'ptb'],
                    model_seq_len=args.model_seq_len,
                    batch_size=4,
                    device=args.DEV,
                )
                for ds_name, ppl_val in outer_stage_ppl.items():
                    print(f"   ✓ {ds_name} PPL: {ppl_val:.4f}")
                if model_dtype != target_eval_dtype:
                    model = model.to(dtype=model_dtype)

            # Save checkpoint after truncation, before between-stage corrections
            if (not getattr(args, "time_analysis", False)) and getattr(args, "save_after_truncation", False):
                model_dtype = next(model.parameters()).dtype
                if model_dtype != torch.float16:
                    model = model.half()
                model = _save_stage_checkpoint(model, tokenizer, exp_dir, stage_idx)
                model = model.to(dev)

            is_final_stage = (stage_idx == args.num_stages)
            with _time_block("between_stage_corrections", stage=stage_idx, args=args):
                run_between_stage_corrections(
                    args, model, tokenizer, dev, teacher_weights, stage_idx, is_final_stage,
                    substituted_keys=SUBSTITUTED_MODULE_KEYS,
                )
            continue

        # ===== UNIFIED STAGE LOGIC (handles both single and multi inner stages) =====
        # Compute stage-level target
        stage_start_dense = compute_dense_params(args.model, model)
        stage_total_target_drop = int(stage_start_dense * args.global_prune_ratio)
        if stage_total_target_drop <= 0:
            print("⚠️  Stage target drop is zero. Stopping.")
            break

        print(f"\n{'='*70}")
        print(
            f"🚉 Stage {stage_idx}/{args.num_stages} | "
            f"total_target_drop={stage_total_target_drop:,} dense params "
            f"(current total {stage_start_dense:,})"
        )
        print(f"{'='*70}")

        # Initialize accumulators
        stage_counts_total: Counter = Counter()
        stage_dense_effects_total: Dict[str, int] = defaultdict(int)
        states = {}

        # Get importance (reuse or recompute)
        with _time_block("importance+profiling", stage=stage_idx, args=args):
            current_importance, current_profiling, L_cache_new, current_svd_cache = get_or_recompute_importance(
                args, model, tokenizer, dev, stage_idx,
                importance, profiling_mat, svd_cache
            )
        importance = current_importance
        profiling_mat = current_profiling
        if L_cache_new is not None:
            L_cache = L_cache_new
        svd_cache = current_svd_cache

        # Build module states
        states = build_module_states(
            current_importance, args, model, args.model,
            current_ranks=None,
        )
        if not states:
            print("⚠️  No prunable modules detected. Stopping.")
            break

        # Run stage plan
        selected, balance, dense_removed, dense_effects = stage_plan(
            states,
            stage_quota=stage_total_target_drop,
            remaining_param_budget=stage_total_target_drop,
            stage_idx=stage_idx,
            selection_mode=args.selection_mode,
            remap=getattr(args, "remap", False),
            target_retained=(remap_target_retained if getattr(args, "remap", False) else None),
        )

        if not selected:
            print("⚠️  Stage produced no selections.")
            break

        # Accumulate
        stage_counts = aggregate_stage_counts(selected)
        stage_counts_total.update(stage_counts)
        for k, v in dense_effects.items():
            stage_dense_effects_total[k] += v

        print(
            f"   → Selected {len(selected)} singular values "
            f"({sum(stage_counts.values())} drops across {len(stage_counts)} modules)"
        )
        print(f"   → Dense params removed (virtual): {int(dense_removed):,}")

        # Compute revert info and planned ranks
        planned_ranks = {key: states[key].current_rank() for key in stage_counts}
        reverted_keys, stage_revert_log, planned_ranks = compute_revert_info(
            states, stage_counts, planned_ranks
        )

        # Apply truncations
        with _time_block("truncation", stage=stage_idx, args=args):
            apply_truncations(
                args, model, tokenizer, states, planned_ranks, stage_counts, reverted_keys,
                L_cache, current_svd_cache, teacher_weights, teacher_biases, dev,
                substituted_keys=SUBSTITUTED_MODULE_KEYS,
            )

        # Optional gradient direction correction after truncation
        with _time_block("gd_correction", stage=stage_idx, args=args):
            run_gd_correction_after_truncation(
                args, model, tokenizer, dev,
                stage_idx=stage_idx,
                inner_idx=1,
            )

        # Free SVD cache
        if current_svd_cache is not None:
            del current_svd_cache
            current_svd_cache = None
            svd_cache = None
            torch.cuda.empty_cache()

        current_dense = compute_dense_params(args.model, model)
        print(
            f"\n   → Dense params after stage {stage_idx}: "
            f"{current_dense:,} "
            f"(removed {total_dense_baseline - current_dense:,} from baseline)"
        )

        # Build snapshot and stage data
        stage_rank_snapshot = build_stage_rank_snapshot(states)
        stage_rank_data = {
            "stage": stage_idx,
            "modules": stage_rank_snapshot,
            "reverts": stage_revert_log if 'stage_revert_log' in dir() else [],
        }

        # Evaluate perplexity BEFORE pull_subspace (wikitext2 + c4 + ptb)
        if (not getattr(args, "time_analysis", False)) and args.eval_ppl_per_outer_stage:
            print(f"\n   📊 [BEFORE pull_subspace] Evaluating perplexity on wikitext2 + c4 + ptb (outer stage {stage_idx}/{args.num_stages})...")
            if getattr(args, "quantize_8_bit", False):
                quant_dequant_roundtrip_targets_(args, model, dev)
            target_eval_dtype = _intermediate_eval_dtype(args)
            model_dtype = next(model.parameters()).dtype
            if model_dtype != target_eval_dtype:
                model = model.to(dtype=target_eval_dtype)
            outer_stage_ppl = ppl_eval(
                model, tokenizer,
                datasets=['wikitext2', 'c4', 'ptb'],
                model_seq_len=args.model_seq_len,
                batch_size=4,
                device=args.DEV,
            )
            for ds_name, ppl_val in outer_stage_ppl.items():
                print(f"   ✓ [BEFORE pull_subspace] {ds_name} PPL: {ppl_val:.4f}")
            if model_dtype != target_eval_dtype:
                model = model.to(dtype=model_dtype)

        # Evaluate perplexity BEFORE pull_subspace when pull_subspace is enabled (no fp16 conversion)
        if (not getattr(args, "time_analysis", False)) and getattr(args, "pull_subspace", False) and not args.eval_ppl_per_outer_stage:
            print(f"\n   {'='*60}")
            print(f"   📊 [BEFORE pull_subspace] Evaluating PPL on wikitext2 + c4 + ptb (stage {stage_idx}/{args.num_stages})")
            print(f"   {'='*60}")
            if getattr(args, "quantize_8_bit", False):
                quant_dequant_roundtrip_targets_(args, model, dev)
            target_eval_dtype = _intermediate_eval_dtype(args)
            model_dtype = next(model.parameters()).dtype
            if model_dtype != target_eval_dtype:
                model = model.to(dtype=target_eval_dtype)
            ppl_before_pullsubspace = ppl_eval(
                model, tokenizer,
                datasets=['wikitext2', 'c4', 'ptb'],
                model_seq_len=args.model_seq_len,
                batch_size=4,
                device=args.DEV,
            )
            for ds_name, ppl_val in ppl_before_pullsubspace.items():
                print(f"   ✓ [BEFORE pull_subspace] {ds_name} PPL: {ppl_val:.4f}")
            if model_dtype != target_eval_dtype:
                model = model.to(dtype=model_dtype)

        # Save checkpoint after truncation, before between-stage corrections
        if (not getattr(args, "time_analysis", False)) and getattr(args, "save_after_truncation", False):
            model_dtype = next(model.parameters()).dtype
            if model_dtype != torch.float16:
                model = model.half()
            model = _save_stage_checkpoint(model, tokenizer, exp_dir, stage_idx)
            model = model.to(dev)

        # Between-stage corrections
        is_final_stage = (stage_idx == args.num_stages)
        with _time_block("between_stage_corrections", stage=stage_idx, args=args):
            run_between_stage_corrections(
                args, model, tokenizer, dev, teacher_weights, stage_idx, is_final_stage,
                substituted_keys=SUBSTITUTED_MODULE_KEYS,
            )

        # Evaluate perplexity AFTER pull_subspace (wikitext2 only)
        if (not getattr(args, "time_analysis", False)) and args.eval_ppl_per_outer_stage:
            print(f"\n   📊 [AFTER pull_subspace] Evaluating perplexity on wikitext2 (outer stage {stage_idx}/{args.num_stages})...")
            if getattr(args, "quantize_8_bit", False):
                quant_dequant_roundtrip_targets_(args, model, dev)
            target_eval_dtype = _intermediate_eval_dtype(args)
            model_dtype = next(model.parameters()).dtype
            if model_dtype != target_eval_dtype:
                model = model.to(dtype=target_eval_dtype)
            outer_stage_ppl_after = ppl_eval(
                model, tokenizer,
                datasets=['wikitext2'],
                model_seq_len=args.model_seq_len,
                batch_size=4,
                device=args.DEV,
            )
            for ds_name, ppl_val in outer_stage_ppl_after.items():
                print(f"   ✓ [AFTER pull_subspace] {ds_name} PPL: {ppl_val:.4f}")
            if model_dtype != target_eval_dtype:
                model = model.to(dtype=model_dtype)

        # Evaluate perplexity AFTER pull_subspace when pull_subspace is enabled (no fp16 conversion)
        if (not getattr(args, "time_analysis", False)) and getattr(args, "pull_subspace", False) and not args.eval_ppl_per_outer_stage:
            print(f"\n   {'='*60}")
            print(f"   📊 [AFTER pull_subspace] Evaluating PPL on wikitext2 + c4 + ptb (stage {stage_idx}/{args.num_stages})")
            print(f"   {'='*60}")
            if getattr(args, "quantize_8_bit", False):
                quant_dequant_roundtrip_targets_(args, model, dev)
            target_eval_dtype = _intermediate_eval_dtype(args)
            model_dtype = next(model.parameters()).dtype
            if model_dtype != target_eval_dtype:
                model = model.to(dtype=target_eval_dtype)
            ppl_after_pullsubspace = ppl_eval(
                model, tokenizer,
                datasets=['wikitext2', 'c4', 'ptb'],
                model_seq_len=args.model_seq_len,
                batch_size=4,
                device=args.DEV,
            )
            for ds_name, ppl_val in ppl_after_pullsubspace.items():
                print(f"   ✓ [AFTER pull_subspace] {ds_name} PPL: {ppl_val:.4f}")
            if model_dtype != target_eval_dtype:
                model = model.to(dtype=model_dtype)

        # Log module distribution
        log_module_distribution(args, model, states)

        # Save stage rank data to JSON
        if not getattr(args, "time_analysis", False):
            stage_rank_path = os.path.join(exp_dir, f"stage{stage_idx}_ranks.json")
            with open(stage_rank_path, "w", encoding="utf-8") as stage_rank_file:
                json.dump(stage_rank_data, stage_rank_file, indent=2)

    # Check if consolidation is needed
    needs_consolidation = float(args.global_prune_ratio) > 0.0

    if args.selection_mode in ("svdllm", "vanilla_svd"):
        needs_consolidation = False
        print(f"\n✅ Skipping final consolidation - {args.selection_mode} mode already has final low-rank modules")

    if needs_consolidation:
        with _time_block("final_consolidation", stage=args.num_stages + 1, args=args):
            #Finalize all target modules to match planned ranks
            print("\n🧩 Finalizing modules to low rank where eligible...")
            layers = get_layers(args.model, model)
            for i, layer in enumerate(tqdm(layers, desc="Final consolidation", total=len(layers))):
                subset = find_target_modules_in_layer(layer)
                for name, mod in subset.items():
                    # name is logical: 'self_attn.q_proj', 'mlp.gate_proj', etc
                    key = f"layer{i}.{name}"
                    if key not in states:
                        continue
                    st = states[key]
                    final_rank = st.current_rank()
                    full_rank = min(st.shape)

                    # if isinstance(mod, LowRankLinear) and mod.rank == final_rank:
                    #     continue

                    # Only compress if we did drop something and the final rank is not above r*
                    if final_rank < full_rank and final_rank <= st.r_star:
                        cached_L = L_cache.get(i, {}).get(name) if L_cache else None
                        if cached_L is None:
                            L = torch.eye(mod.in_features, device=dev, dtype=torch.float32)
                        else:
                            L = cached_L.to(dev, dtype=torch.float32)
                        # replace with LowRankLinear
                        low_rank = factorize_linear(
                            mod,
                            L,
                            final_rank,
                            dev,
                            use_triangular_solve=args.use_triangular_solve,
                            keep_mask=st.keep_mask,
                            mask_rank_only=st.mask_rank_only,
                            remap_fake_bnb=getattr(args, "remap", False),
                            quantize_8_bit=getattr(args, "quantize_8_bit", False),
                        )
                        replace_module(args.model, model, key, low_rank.to(dev))
                    else:
                        # ensure dense stays dense with original shape, nothing to do
                        pass

    # Recompute the actual dense parameter count after the consolidation
    final_dense_after_consolidation = compute_dense_params(args.model, model)
    if needs_consolidation:
        print(f"✅ Consolidation complete, dense params now: {final_dense_after_consolidation:,}")
    else:
        print(f"✅ Model already in final form, dense params: {final_dense_after_consolidation:,}")

    final_ppl_results: Dict[str, float] = {}
    final_cs_results = None

    do_final_fp32_eval = (not getattr(args, "time_analysis", False)) and bool(getattr(args, "eval_ppl", False) or getattr(args, "evaluate_commonsense", False))
    if do_final_fp32_eval:
        prev_dtype = next(model.parameters()).dtype
        # Apply quant-dequant roundtrip to simulate int8 quantization error before final eval
        if getattr(args, "quantize_8_bit", False):
            quant_dequant_roundtrip_targets_(args, model, dev)
        model = model.to(dtype=evaluation_dtype)  # fp32

        if getattr(args, "eval_ppl", False):
            # Parse final evaluation datasets from comma-separated string
            final_eval_datasets = [ds.strip() for ds in args.final_eval_datasets.split(',') if ds.strip()]
            if not final_eval_datasets:
                final_eval_datasets = ['wikitext2']
            print(f"\n📈 Evaluating final perplexity on {final_eval_datasets} before saving...")
            ppl_results = ppl_eval(
                model, tokenizer, datasets=final_eval_datasets,
                model_seq_len=args.model_seq_len, batch_size=4, device=args.DEV
            )
            if isinstance(ppl_results, dict) and ppl_results:
                final_ppl_results = ppl_results
                for ds_name, ppl_val in ppl_results.items():
                    print(f"   ✅ Final {ds_name} PPL: {ppl_val:.4f}")

        if getattr(args, "evaluate_commonsense", False):
            tasks_csv = getattr(args, "commonsense_tasks", "arc_easy,arc_challenge,openbookqa,winogrande,hellaswag,piqa")
            print(f"\n🧠 Evaluating commonsense tasks: {tasks_csv}")
            final_cs_results = commonsense_eval(model, tokenizer, tasks_csv=tasks_csv, device=args.DEV)

            # Print a compact summary
            try:
                res = final_cs_results.get("results", {})
                print("\n🧠 Commonsense results summary")
                for task_name, metrics in res.items():
                    if isinstance(metrics, dict):
                        keys = [k for k in ["acc", "acc_norm", "exact_match", "mc1", "mc2"] if k in metrics]
                        if keys:
                            msg = ", ".join([f"{k}={metrics[k]:.4f}" for k in keys if isinstance(metrics[k], (int, float))])
                            print(f"   {task_name}: {msg}")
                        else:
                            print(f"   {task_name}: {metrics}")
                    else:
                        print(f"   {task_name}: {metrics}")
            except Exception:
                print("   Commonsense evaluation finished, results object printed below")
                print(final_cs_results)

        model = model.to(dtype=prev_dtype)

    if target_storage_dtype is not None and next(model.parameters()).dtype != target_storage_dtype:
        model = model.to(dtype=target_storage_dtype)
        print("\n🔄 Converted model to FP16 for storage.")

    if args.verbose and 'states' in locals() and states:
        print("\n📊 Final module ranks (kept / full rank) [shape]:")
        for key in sorted(states.keys()):
            state = states[key]
            full_rank = min(state.shape)
            current_rank = max(0, state.current_rank())
            print(f"   {key}: {current_rank}/{full_rank} (shape={state.shape[0]}x{state.shape[1]})")

    # Verify dense parameter reduction matches global target
    final_dense = compute_dense_params(args.model, model)
    dense_removed = total_dense_baseline - final_dense
    achieved_ratio = dense_removed / max(total_dense_baseline, 1)
    print("\n📏 Dense parameter verification")
    print(f"   Total dense params (baseline): {total_dense_baseline:,}")
    print(f"   Dense params after pruning:    {final_dense:,}")
    print(f"   Removed dense params:          {dense_removed:,}")
    print(f"   Achieved global drop ratio:    {achieved_ratio:.4f}")

    # Create final summary JSON
    final_summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_tag": exp_tag,
        "num_stages": args.num_stages,
        "global_prune_ratio": args.global_prune_ratio,
        "dense_baseline_params": total_dense_baseline,
        "dense_final_params": final_dense,
        "dense_params_removed": dense_removed,
        "achieved_prune_ratio": achieved_ratio,
        "final_perplexity": final_ppl_results,  # Dict with both wikitext2 and c4
        "commonsense": _extract_commonsense_metrics(final_cs_results),
        "initial_commonsense": _extract_commonsense_metrics(initial_cs_results),
        "model": args.model,
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "model_seq_len": args.model_seq_len,
        "seed": args.seed,
        "selection_mode": args.selection_mode,
        "truncation_mode": "fullprunestage",
    }

    if not getattr(args, "time_analysis", False):
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_summary_path = os.path.join(exp_dir, f"final_summary_{timestamp_str}.json")
        with open(final_summary_path, "w", encoding="utf-8") as f:
            json.dump(_sanitize_for_json(final_summary), f, indent=2)
        print(f"\n💾 Saved final summary to {final_summary_path}")

    if not getattr(args, "time_analysis", False):
        ppl_token = "pplNA"
        if final_ppl_results:
            # Use wikitext2 PPL for filename if available, otherwise first result
            final_ppl_value = final_ppl_results.get('wikitext2', next(iter(final_ppl_results.values())))
            ppl_token = f"ppl{final_ppl_value:.4f}".replace('.', 'p')

        # Build save blob
        save_blob = {
            "model": model,
            "tokenizer": tokenizer,
            "model_name": args.model,
        }

        # Apply 8-bit quantization if requested
        remap_tag = "_remap" if getattr(args, "remap", False) else ""
        if getattr(args, "quantize_8_bit", False):
            model = model.to(dev)
            mapping_info = quantize_target_modules_inplace(args, model, dev)
            save_blob["mapping_info"] = mapping_info
            save_blob["quantize_8_bit"] = True
            suffix = f"{remap_tag}_int8_compressed.pt"
        else:
            save_blob["quantize_8_bit"] = False
            suffix = f"{remap_tag}_fp16_compressed.pt"

        final_model_path = os.path.join(
            exp_dir,
            f"final_{ppl_token}{suffix}"
        )
        save_blob["model"] = model.cpu()
        torch.save(save_blob, final_model_path)
        print(f"\n💾 Saved compressed model snapshot to {final_model_path}")

    # Print timing summary if time_analysis mode is enabled
    if getattr(args, "time_analysis", False):
        print("\n" + "=" * 60)
        print("⏱️  Time analysis summary (seconds)")
        print("=" * 60)
        for k in sorted(TIMES.keys()):
            print(f"   {k:>26}: {TIMES[k]:.4f}")

        # Compute total time across all components
        total_time = sum(TIMES.values())
        print(f"   {'-'*26}--{'-'*8}")
        print(f"   {'TOTAL':>26}: {total_time:.4f}")

        print("\n⏱️  Per-stage breakdown (seconds)")
        total_all_stages = 0.0
        for s in sorted(STAGE_TIMES.keys()):
            row = STAGE_TIMES[s]
            stage_total = sum(row.values())
            total_all_stages += stage_total
            items = ", ".join([f"{k}={row[k]:.4f}" for k in sorted(row.keys())])
            print(f"   stage {s}: {items} | stage_total={stage_total:.4f}")

        print(f"\n   {'TOTAL across all stages':>30}: {total_all_stages:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
    print("\n🎉 All done!")

