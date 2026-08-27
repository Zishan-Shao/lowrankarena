"""
Argument parsing and validation for SVD-LLM compression.

Extracted from SVDLLM_sum0strategy_alllowrank_fullprune_eachstage_onescalaer_correction.py
"""

import argparse
import os
import torch


def define_compression_args() -> argparse.ArgumentParser:
    """Define all compression-related arguments."""
    parser = argparse.ArgumentParser(description="SVDLLM Sum-Zero Stage Strategy")

    # Model arguments
    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='./',
                        help='Directory to store logs/checkpoints')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Split to use when --dataset is a Hugging Face dataset name."
    )
    parser.add_argument(
        "--dataset_text_fields",
        type=str,
        default=None,
        help='Comma separated preferred fields to build text, example "question,context". If not set, code will guess.'
    )
    parser.add_argument('--nsamples', type=int, default=256)
    parser.add_argument(
        '--nsamples_gradient_subset',
        type=int,
        default=None,
        help='Number of calibration samples to use for gradient based importance; must not exceed nsamples'
    )
    parser.add_argument(
        '--importance_seq_len',
        type=int,
        default=None,
        help=(
            'Max sequence length used when computing gradients for importance '
            'recomputation; if set, sequences are truncated from the right'
        ),
    )
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--model_seq_len', type=int, default=2048)
    parser.add_argument('--DEV', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Strategy-specific knobs
    parser.add_argument('--global_prune_ratio', type=float, default=0.2,
                        help='Fraction of total parameters to drop globally (0-1)')
    parser.add_argument('--num_stages', type=int, default=4)
    parser.add_argument('--keep_rank_ratio', type=float, default=0.10,
                        help='Do not drop below keep_rank_ratio * r_star for any module '
                             '(ignored if --use_absolute_min_rank is set)')
    parser.add_argument('--use_absolute_min_rank', action='store_true',
                        help='Use a fixed minimum rank instead of keep_rank_ratio * r_star')
    parser.add_argument('--min_absolute_rank', type=int, default=0,
                        help='Absolute minimum rank to keep when --use_absolute_min_rank is set')
    parser.add_argument('--use_triangular_solve', action='store_true', default=False,
                        help='Use triangular solves instead of explicit inverses when handling whitening matrices')

    # Evaluation arguments
    parser.add_argument('--eval_ppl', action='store_true',
                        help='Evaluate Wikitext2 perplexity after compression')
    parser.add_argument('--save_after_truncation', action='store_true',
                        help='If set, save a model checkpoint after each outer stage truncation, before between-stage corrections')
    parser.add_argument('--eval_ppl_per_outer_stage', action='store_true',
                        help='Evaluate perplexity on wikitext2 and c4 after each outer stage')
    parser.add_argument('--final_eval_datasets', type=str, default='wikitext2,c4,ptb',
                        help='Comma-separated list of datasets for final PPL evaluation (default: wikitext2,c4,ptb)')

    # Commonsense evaluation
    parser.add_argument(
        '--evaluate_commonsense',
        action='store_true',
        help='If set, run commonsense benchmark evaluation at the end using lm_eval.'
    )
    parser.add_argument(
        '--commonsense_tasks',
        type=str,
        default='arc_easy,arc_challenge,openbookqa,winogrande,hellaswag,piqa,mathqa',
        help='Comma separated lm_eval task names.'
    )
    parser.add_argument(
        '--initial_evaluate_commonsense',
        action='store_true',
        help='If set, run commonsense benchmark evaluation on the uncompressed teacher model before compression.'
    )
    parser.add_argument(
        '--initial_eval_ppl',
        action='store_true',
        help='If set, evaluate perplexity on the uncompressed model before compression.'
    )

    # Remap and quantization
    parser.add_argument(
        '--remap',
        action='store_true',
        help='Enable remap retention stopping rule and fake bnb quant dequant noise on U Sigma and V transpose.'
    )
    parser.add_argument(
        '--quantize_8_bit',
        action='store_true',
        help='If set, quantize target linear modules to uint8 for storage and halve prune ratio targets.',
    )
    parser.add_argument(
        '--profile_independently',
        action='store_true',
        help='If set, profile a separate whitening matrix for each target module.',
    )

    # Selection mode
    parser.add_argument('--selection_mode', type=str, default='zero_sum',
                        choices=[
                            'zero_sum',
                            'vanilla_svd',
                            'smallest_mag',
                            'only_delta_ce',
                            'only_delta_ce_zerosum',
                            'sval_mag_het',
                            'svdllm',
                            'delta_in_subspace',
                            'negative_sum',
                            'negative_sum_no_order',
                        ],
                        help=(
                            'Pruning strategy: '
                            'zero_sum balances signs with smallest |delta|, '
                            'smallest_mag drops smallest |grad_sigma * sigma|, '
                            'only_delta_ce uses |grad_sigma * sigma| and ignores singular value magnitude ordering, '
                            'only_delta_ce_zerosum uses |grad_sigma * sigma| with zero sum sign balancing, '
                            'sval_mag_het uses only |sigma| magnitude, smallest singular values dropped first, '
                            'vanilla_svd truncates each target module by plain SVD on W (no profiling, no whitening), '
                            'with rank r = (1-pruning_ratio)*m*n/(m+n) per module, '
                            'svdllm skips the greedy stage planner and does a direct SVD truncation per module '
                            'in the whitened space W L with target rank from the global prune ratio, '
                            'delta_in_subspace scores each singular value by the norm of the induced change in W '
                            'projected onto the average gradient direction for that module, '
                            'negative_sum always picks the most negative delta globally, '
                            'then smallest positive if none negative remain, '
                            'negative_sum_no_order does not reorder SVs per-module; instead pushes ALL SVs '
                            'into a global pool and picks most negative delta first'
                        ))

    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output including detailed module ranks and densification info')

    parser.add_argument('--print_target_layers', action='store_true',
                        help='Print the names of all target layers before compression begins')

    parser.add_argument('--print_all_linear_layers', action='store_true',
                        help='Print all linear layers (target and non-target) before compression begins')

    parser.add_argument(
        '--sub_with_teacher_module',
        action='store_true',
        help=(
            'If set, whenever a planned target rank for a module exceeds r_star, '
            'substitute that module with the cached teacher weights instead of '
            'leaving its current truncated weights'
        ),
    )

    # Pull subspace correction
    parser.add_argument(
        '--pull_subspace',
        action='store_true',
        help=(
            'If set, perform a subspace based pull projection between stages. '
            'For each target module, build a gradient subspace from '
            'nsamples_subspace_proj calibration batches and project the '
            'teacher delta onto that subspace.'
        ),
    )
    parser.add_argument(
        '--nsamples_subspace_proj',
        type=int,
        default=None,
        help=(
            'Number of calibration batches to use for gradient subspace '
            'projection per module. If not set, uses the full calibration set.'
        ),
    )
    parser.add_argument(
        '--subspace_proj_seq_len',
        type=int,
        default=2048,
        help='Max sequence length used during pull_subspace gradient computation. '
             'Shorter sequences reduce memory. Default: 1024.'
    )
    parser.add_argument(
        '--random_samples_for_pullsubspace',
        action='store_true',
        help=(
            'If set, randomly select nsamples_subspace_proj batches from the '
            'base calibration set each time pull_subspace is run, instead of '
            'always using the first nsamples_subspace_proj batches.'
        ),
    )

    # Project to delta correction
    parser.add_argument(
        '--project_to_delta',
        action='store_true',
        help=(
            'If set, perform module-wise correction between stages by projecting '
            'the average gradient direction onto the delta direction to the '
            'pretrained teacher weights, then snapping weights by adding that '
            'projected component (no fixed step size).'
        ),
    )
    parser.add_argument(
        '--n_samples_proj_to_delta',
        type=int,
        default=None,
        help=(
            'Number of calibration batches from the original base calibration '
            'set to use for computing average gradients for project_to_delta. '
            'If not set, uses the full base calibration set.'
        ),
    )

    # Alpha blend correction
    parser.add_argument(
        '--alpha_blend_correction',
        action='store_true',
        help=(
            'If set, after densifying between stages, blend current truncated '
            'weights toward the cached teacher weights using '
            'W <- W + alpha * (W_T - W).'
        ),
    )
    parser.add_argument(
        '--blend_factor',
        type=float,
        default=0.0,
        help='Alpha used for alpha_blend_correction. Typical range is 0 to 1.',
    )

    parser.add_argument(
        '--efficient_importance',
        action='store_true',
        help=(
            'If set, compute importance one module at a time and accumulate only grad_sigma '
            '(length r) on CPU, instead of storing full weight sized average gradients.'
        ),
    )

    # Gradient direction correction
    parser.add_argument(
        '--gd_correction_mode',
        action='store_true',
        help=(
            'If set, after each truncation step, compute an average gradient '
            'over a subset of the original calibration samples and take one '
            'manual gradient step along that average direction.'
        ),
    )
    parser.add_argument(
        '--nsamples_gd_correction',
        type=int,
        default=None,
        help=(
            'Number of calibration samples to use for the gradient direction '
            'correction step. Samples are taken from the original base '
            'calibration set used for profiling.'
        ),
    )
    parser.add_argument(
        '--gd_correction_lr',
        type=float,
        default=1e-4,
        help='Learning rate for the one step gradient direction correction.',
    )
    # Efficient accumulation for profiling (reduces GPU memory)
    parser.add_argument(
        '--efficient_accumulate',
        action='store_true',
        help='Use chunked gram accumulation and offload covariance to CPU during profiling.'
    )
    parser.add_argument(
        '--efficient_accumulate_chunk_tokens',
        type=int,
        default=256,
        help='Token-chunk size used by --efficient_accumulate (smaller = lower GPU peak, slower).'
    )
    parser.add_argument(
        '--efficient_accumulate_dtype',
        type=str,
        default='bf16',
        choices=['bf16', 'fp16', 'fp32'],
        help='Dtype used on GPU for chunked gram matmul in --efficient_accumulate.'
    )

    # Time analysis mode
    parser.add_argument(
        '--time_analysis',
        action='store_true',
        help='Wall-clock time analysis for core method only (no load/save/eval).'
    )

    return parser


def validate_args(args) -> None:
    """Validate argument combinations and adjust values as needed."""

    # Validate bitsandbytes when --remap is enabled
    if getattr(args, "remap", False):
        try:
            import bitsandbytes as _bnb_check
        except Exception:
            raise RuntimeError("--remap requires bitsandbytes to be installed")
        if not torch.cuda.is_available():
            raise RuntimeError("--remap requires CUDA for bitsandbytes dynamic quantization")

    # Keep original ratio for logging
    args._global_prune_ratio_original = float(args.global_prune_ratio)

    # Keep the float conversion
    args.global_prune_ratio = float(args.global_prune_ratio)

    # When going 16->8 bits, apply maintenance doubling rule
    if getattr(args, "quantize_8_bit", False):
        p = args.global_prune_ratio
        p = 1.0 - 2.0 * (1.0 - p)
        args.global_prune_ratio = max(0.0, min(1.0, p))
        print(f"🔢 8-bit quantization enabled: adjusted prune ratio from {args._global_prune_ratio_original:.3f} to {args.global_prune_ratio:.3f}")

    # Validate and clamp alpha_blend_correction factor
    if getattr(args, "alpha_blend_correction", False):
        try:
            args.blend_factor = float(args.blend_factor)
        except Exception:
            args.blend_factor = 0.0
        # keep it sane, but do not hard fail
        if args.blend_factor < 0.0:
            args.blend_factor = 0.0
        if args.blend_factor > 1.0:
            args.blend_factor = 1.0


def print_args_summary(args) -> None:
    """Print algorithm arguments summary."""
    print("\n" + "=" * 60)
    print("⚙️  Algorithm Arguments Summary")
    print("=" * 60)
    print(f"   model:                {args.model}")
    print(f"   selection_mode:       {args.selection_mode}")
    if args.use_absolute_min_rank:
        print(f"   min_absolute_rank:    {max(1, args.min_absolute_rank)}")
    else:
        print(f"   keep_rank_ratio:      {args.keep_rank_ratio}")
    print(f"   num_stages:           {args.num_stages}")
    print(f"   global_prune_ratio:   {args.global_prune_ratio}")
    grad_subset_early = args.nsamples_gradient_subset if args.nsamples_gradient_subset is not None else args.nsamples
    print(f"   nsamples profiling:   {args.nsamples}")
    print(f"   nsamples gradients:   {grad_subset_early}")
    print(f"   use_triangular_solve: {args.use_triangular_solve}")
    print(f"   pull_subspace:        {getattr(args, 'pull_subspace', False)}")
    if getattr(args, "nsamples_subspace_proj", None) is not None:
        print(f"   nsamples_subspace_proj: {args.nsamples_subspace_proj}")
    print(f"   gd_correction_mode:   {getattr(args, 'gd_correction_mode', False)}")
    if getattr(args, "gd_correction_mode", False):
        print(f"   gd_correction_lr:     {args.gd_correction_lr}")
        ns_gd = args.nsamples_gd_correction if args.nsamples_gd_correction is not None else "all"
        print(f"   nsamples_gd_correction: {ns_gd}")
    print("=" * 60 + "\n")


def build_experiment_tag(args) -> str:
    """Build experiment directory tag from arguments."""
    # Extract model name and sanitize for filesystem
    model_name = getattr(args, "model", "unknown_model")
    # Take last component (after final /) and replace remaining / with __
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    model_tag = model_short.replace("/", "__")

    rank_descriptor = f"krr{args.keep_rank_ratio:g}"
    if args.use_absolute_min_rank:
        rank_descriptor = f"absmin{max(1, args.min_absolute_rank)}"
    rank_descriptor_tag = rank_descriptor.replace('.', 'p')
    selection_tag = f"sel{args.selection_mode}"

    # Add truncation mode tag (fullprunestage means densify after each stage)
    truncation_mode = "fullprunestage"

    # Add quantization and remap tags if enabled
    quant_tag = "_int8" if getattr(args, "quantize_8_bit", False) else ""
    remap_tag = "_remap" if getattr(args, "remap", False) else ""

    # Add correction mode tags if enabled
    alphablend_tag = "_alphablend" if getattr(args, "alpha_blend_correction", False) else ""
    gdcorr_tag = "_gdcorr" if getattr(args, "gd_correction_mode", False) else ""
    proj2delta_tag = "_proj2delta" if getattr(args, "project_to_delta", False) else ""

    gpr_for_name = getattr(args, "_global_prune_ratio_original", args.global_prune_ratio)
    exp_tag = f"{model_tag}_S{args.num_stages}_gpr{gpr_for_name:g}".replace('.', 'p')
    exp_tag = f"{exp_tag}_{rank_descriptor_tag}_{selection_tag}_{truncation_mode}{remap_tag}{quant_tag}{alphablend_tag}{gdcorr_tag}{proj2delta_tag}"
    return exp_tag
