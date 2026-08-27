# compression/ - Modular compression utilities for SVD-LLM
#
# This package contains extracted modules from the main compression script:
# - args.py: Argument parsing and validation
# - losses.py: Loss computation (CE)
# - batch_utils.py: Batch processing helpers
# - profiling.py: Whitening/profiling matrices
# - importance.py: Gradient importance computation
# - gradient.py: Gradient accumulation and importance orchestration
# - planning.py: Stage planning and module state management
# - corrections.py: Between-stage correction methods
# - evaluation.py: Evaluation helpers (commonsense, seeds)
# - truncation.py: SVD truncation application

from .args import (
    define_compression_args,
    validate_args,
    print_args_summary,
    build_experiment_tag,
)

from .losses import (
    compute_ce_loss,
    compute_loss,
    compute_per_sample_ce_loss,
)

from .batch_utils import (
    prepare_batch,
    cleanup_gpu,
    build_model_inputs,
    get_batch_size,
    get_seq_len,
)

from .profiling import (
    svd_whitened_sorted,
    prepare_calib_loader,
    compute_profiling_matrices,
    profle_svdllm_shared,
    build_L_cache,
)

from .importance import (
    _importance_autocast_dtype,
    right_solve_Rt,
    compute_importance_for_layer_fast,
    compute_grad_sigma_efficient_for_module,
)

from .gradient import (
    compute_average_gradients,
    recompute_importance,
    get_or_recompute_importance,
)

from .planning import (
    build_module_states,
    stage_plan,
    build_stage_rank_snapshot,
    compute_revert_info,
    log_module_distribution,
)

from .corrections import (
    run_pull_subspace_between_stages,
    run_project_to_delta_between_stages,
    run_between_stage_corrections,
)

from .evaluation import (
    commonsense_eval,
    set_deterministic_seeds,
)

from .truncation import (
    apply_truncations,
    run_vanilla_svd_stage,
)
