"""
Stage planning and module state management for SVD-LLM compression.

Contains functions for building module pruning states and planning
which singular values to drop at each compression stage.
"""

import heapq
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import torch
from tqdm import tqdm

from utils.model_utils import LowRankLinear
from utils.correction_utils import (
    _is_target_module,
    break_even_rank,
    ModulePruneState,
    StageSelection,
    get_module_from_key,
    get_module_current_rank,
    build_priority_queues,
)


def build_module_states(
    importance: Dict[str, Dict],
    args,
    model,
    model_name: str,
    current_ranks: Optional[Dict[str, int]] = None,
) -> Dict[str, ModulePruneState]:
    """
    Build per module prune states from current importance.

    Completely local per stage:
      * No prior_dropped_svs
      * Caps are defined by current rank and a minimum rank that can be
        controlled by one of:
          - absolute minimum rank
          - keep_rank_ratio * r_star

    Args:
        importance: Dict mapping module keys to importance payloads
        args: Compression arguments
        model: The language model
        model_name: Name of the model (for module lookup)
        current_ranks: Optional dict of persistent rank states

    Returns:
        Dict mapping module keys to ModulePruneState objects
    """
    states: Dict[str, ModulePruneState] = {}
    keep_rank_ratio = min(max(args.keep_rank_ratio, 0.0), 1.0)

    for key, payload in importance.items():
        if not _is_target_module(key):
            continue

        # Original SVD singular values (sorted by magnitude) for this module
        sigma_full = payload["sigma"].float()
        grad_sigma_full = payload["grad_sigma"].float()
        m, n = payload["shape"]
        full_rank = min(m, n)

        # Look at the live model and optional rank state
        try:
            module, _, _ = get_module_from_key(model_name, model, key)
        except KeyError:
            continue

        # If we have a persistent rank for this module, use it
        if current_ranks is not None and key in current_ranks:
            actual_rank = min(current_ranks[key], full_rank)
            is_currently_lowrank = actual_rank < full_rank
        else:
            actual_rank = get_module_current_rank(module)
            is_currently_lowrank = actual_rank is not None
            if actual_rank is None:
                actual_rank = full_rank

        # Truncate to the current rank if needed for both sigma and gradients
        if actual_rank < sigma_full.numel():
            sigma_full = sigma_full[:actual_rank]
            grad_sigma_full = grad_sigma_full[:actual_rank]

        # This copy will be possibly reordered depending on selection_mode
        sigma = sigma_full.clone()
        grad_sigma = grad_sigma_full.clone()

        # Helper for vectorized delta computation matching ModulePruneState.peek_delta()
        def _delta_tensor(sig: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
            beta_mixing = getattr(args, "beta_mixing", None)
            if beta_mixing is None:
                if getattr(args, "delta_order", "first_order") == "second_order":
                    return -sig * grad + 0.5 * (sig ** 2) * (grad ** 2)
                return -sig * grad

            beta = float(beta_mixing)
            eps = 1e-12
            sig_abs = torch.clamp(sig.abs(), min=eps)
            grad_abs = torch.clamp(grad.abs(), min=eps)
            sign = torch.where(grad >= 0.0, 1.0, -1.0)
            mixed = (sig_abs ** beta) * (grad_abs ** (1.0 - beta))
            return -sign * mixed

        # Optional reordering or override for selection modes
        sel_mode = getattr(args, "selection_mode", None)

        if sel_mode in (
            "only_delta_ce",
            "only_delta_ce_zerosum",
        ):
            # First order: delta = -sigma * grad
            delta = -sigma * grad_sigma

            order = torch.argsort(delta.abs(), descending=True)
            sigma = sigma[order]
            grad_sigma = grad_sigma[order]
            permutation = order.cpu().tolist()
        elif sel_mode == "sval_mag_het":
            # Ignore gradient information, order by |sigma| alone
            order = torch.argsort(sigma.abs(), descending=True)
            sigma = sigma[order]
            grad_sigma = torch.ones_like(sigma)
            permutation = order.cpu().tolist()
        elif sel_mode == "negative_sum":
            # Sort delta DESCENDING so most negative ends at tail (tail_index picks from end)
            delta = _delta_tensor(sigma, grad_sigma)
            order = torch.argsort(delta, descending=True)
            sigma = sigma[order]
            grad_sigma = grad_sigma[order]
            permutation = order.cpu().tolist()
        else:
            permutation = list(range(sigma.numel()))

        # r_star defined from original shape
        full_rank = min(m, n)
        max_dim = max(m, n)

        # r_star definition
        if getattr(args, "remap", False):
            r_star_full = max_dim
        else:
            r_star_full = break_even_rank(m, n)

        # Minimum rank constraint
        if args.use_absolute_min_rank:
            # Absolute minimum rank takes precedence
            min_rank_full = max(1, args.min_absolute_rank)
        else:
            if getattr(args, "remap", False):
                min_rank_full = max(1, int(math.ceil(keep_rank_ratio * full_rank)))
            else:
                min_rank_full = max(1, int(math.ceil(keep_rank_ratio * r_star_full)))

        # Local cap: from current state you can drop at most
        # actual_rank - min_rank_full, never below zero
        allowed_svs_now = max(0, actual_rank - min_rank_full)
        if allowed_svs_now <= 0:
            continue

        if getattr(args, "remap", False):
            params = actual_rank * max_dim
            cost = max_dim
        else:
            cost = (m + n)
            if is_currently_lowrank:
                params = actual_rank * (m + n)
            else:
                params = m * n

        state = ModulePruneState(
            key=key,
            layer_idx=int(key.split(".", 1)[0].replace("layer", "")),
            module_name=key.split(".", 1)[1],
            shape=(m, n),
            sigma=sigma,
            grad_sigma=grad_sigma,
            params=params,
            cost=cost,
            cap_params=params,
            max_svs=allowed_svs_now,
            r_star=(r_star_full if getattr(args, "remap", False) else min(r_star_full, actual_rank)),
            current_actual_rank=actual_rank,
            is_currently_lowrank=is_currently_lowrank,
            permutation=permutation,
            beta_mixing=getattr(args, "beta_mixing", None),
            delta_order=getattr(args, "delta_order", "first_order"),
        )

        # All singular values start as kept
        state.keep_mask = [True] * sigma.numel()
        state.mask_rank_only = (
            sel_mode == "smallest_mag"
            or sel_mode == "sval_mag_het"
        )


        # No preload of previous consumption
        state.consumed = 0

        states[key] = state

    return states


def stage_plan(
    states: Dict[str, ModulePruneState],
    stage_quota: int,
    remaining_param_budget: int,
    stage_idx: int = 0,
    selection_mode: str = "zero_sum",
    remap: bool = False,
    target_retained: Optional[int] = None,
) -> Tuple[List[StageSelection], float, int, Dict[str, int]]:
    """
    Plan which singular values to drop in this stage.

    Uses a greedy algorithm that selects singular values based on their
    importance scores while respecting the stage budget.

    Args:
        states: Dict of ModulePruneState objects
        stage_quota: Parameter budget for this stage
        remaining_param_budget: Total remaining parameter budget
        stage_idx: Current stage index (for logging)
        selection_mode: Strategy for selecting singular values
        remap: Whether remap mode is enabled
        target_retained: Target retained parameters (required if remap=True)

    Returns:
        Tuple of (selections, running_sum, dense_removed, dense_effects)
    """
    stage_quota = int(stage_quota)
    remaining_param_budget = int(remaining_param_budget)

    if not remap:
        if stage_quota <= 0 or remaining_param_budget <= 0:
            return [], 0.0, 0, {}
    else:
        if target_retained is None:
            raise ValueError("remap=True requires target_retained")

    if remap:
        retained_total = 0
        for st in states.values():
            m, n = st.shape
            retained_total += st.current_rank() * max(m, n)
        initial_retained = retained_total
        stage_budget = max(0, int(initial_retained - int(target_retained)))
    else:
        stage_budget = stage_quota

    queues = None
    smallest_heap = []
    heap_counter = 0

    def push_smallest(state: ModulePruneState):
        nonlocal heap_counter
        delta_val = state.peek_delta()
        if delta_val is None:
            return
        delta_val = float(delta_val)

        if selection_mode == "negative_sum":
            # Heap pops smallest tuple.
            # (False, delta) < (True, delta), so negatives come first.
            # Among negatives: more negative (smaller delta) pops first.
            # Among positives: smaller positive pops first.
            entry = (delta_val >= 0.0, delta_val, heap_counter, state.key, delta_val, state.version)
        else:
            # Existing behavior: smallest |delta| first
            entry = (abs(delta_val), heap_counter, state.key, delta_val, state.version)

        heap_counter += 1
        heapq.heappush(smallest_heap, entry)

    def delta_at_index(st: ModulePruneState, idx: int) -> float:
        """Compute delta for a specific SV index."""
        sigma = float(st.sigma[idx].item())
        grad_sigma = float(st.grad_sigma[idx].item())

        if st.beta_mixing is None:
            if st.delta_order == "second_order":
                return -sigma * grad_sigma + 0.5 * (sigma ** 2) * (grad_sigma ** 2)
            return -sigma * grad_sigma

        beta = float(st.beta_mixing)
        if grad_sigma == 0.0 and sigma == 0.0:
            return 0.0

        eps = 1e-12
        sig_abs = max(abs(sigma), eps)
        grad_abs = max(abs(grad_sigma), eps)
        sign = 1.0 if grad_sigma >= 0.0 else -1.0
        mixed = (sig_abs ** beta) * (grad_abs ** (1.0 - beta))
        return -sign * mixed

    if selection_mode in ("zero_sum", "only_delta_ce_zerosum"):
        # Match legacy zero sum behavior, select smallest |delta| while balancing signs
        queues = build_priority_queues(states, use_min_heap=True)
    elif selection_mode in ("smallest_mag", "only_delta_ce", "sval_mag_het", "delta_in_subspace", "negative_sum", "negative_sum_no_order"):
        # Use a global heap over candidates, ordered by |delta| (or signed for negative_sum*)
        # For sval_mag_het, delta is proportional to sigma, so this becomes |sigma|
        # For delta_in_subspace, delta is the gradient subspace projection score
        # For negative_sum, most negative delta is picked first, then smallest positive
        # For negative_sum_no_order, ALL SVs are pushed at once (no per-module sorting)
        for state in states.values():
            if selection_mode == "negative_sum_no_order":
                # Push ALL SVs into global pool at once
                for idx in range(state.sigma.numel()):
                    d = delta_at_index(state, idx)
                    # negatives first (False < True), then ascending delta within each group
                    entry = (d >= 0.0, d, heap_counter, state.key, idx, state.version)
                    heap_counter += 1
                    heapq.heappush(smallest_heap, entry)
            else:
                push_smallest(state)
    running_sum = 0.0
    dense_removed_stage = 0
    selected: List[StageSelection] = []
    dense_effects = defaultdict(int)

    unit_name = "kept_params" if remap else "params"
    stage_pbar = tqdm(total=max(1, stage_budget), desc=f"Stage {stage_idx} pruning", leave=False, unit=unit_name)

    def pop_one(prefer_positive: bool):
        """Returns (state, delta, idx) where idx is only used for negative_sum_no_order."""
        while True:
            if selection_mode in ("zero_sum", "only_delta_ce_zerosum"):
                entry = queues.pop_prefer_then_any(prefer_positive)
                if entry is None:
                    return None, None, None
                _, key, delta, version = entry
                st = states.get(key)
                if st is None or st.version != version or not st.has_candidate():
                    continue
                return st, float(delta), None

            # Heap-based modes
            while smallest_heap:
                entry = heapq.heappop(smallest_heap)

                if selection_mode == "negative_sum_no_order":
                    _, _, _, key, idx, version = entry
                    st = states.get(key)
                    if st is None:
                        continue
                    # Skip if cap reached
                    if st.consumed >= st.max_svs:
                        continue
                    # Skip if already dropped
                    if st.keep_mask is not None:
                        logical_idx = idx
                        if st.permutation is not None and 0 <= idx < len(st.permutation):
                            logical_idx = st.permutation[idx]
                        if 0 <= logical_idx < len(st.keep_mask) and not st.keep_mask[logical_idx]:
                            continue
                    delta = delta_at_index(st, idx)
                    return st, float(delta), int(idx)

                # Existing heap modes
                if selection_mode == "negative_sum":
                    _, _, _, key, delta, version = entry
                else:
                    _, _, key, delta, version = entry

                st = states.get(key)
                if st is None or st.version != version or not st.has_candidate():
                    continue
                return st, float(delta), None

            return None, None, None

    def has_candidates() -> bool:
        if selection_mode in ("zero_sum", "only_delta_ce_zerosum"):
            return not queues.empty()
        return bool(smallest_heap)

    while True:
        if remap:
            if retained_total <= int(target_retained):
                break
        else:
            if dense_removed_stage >= stage_budget or remaining_param_budget <= 0:
                break

        if not has_candidates():
            break

        prefer_positive = running_sum <= 0.0
        st, delta, idx = pop_one(prefer_positive)
        if st is None:
            break

        prev_dense = dense_removed_stage

        if selection_mode == "negative_sum_no_order":
            st.accept_at(idx)
        else:
            st.accept()

        running_sum += delta

        if remap:
            drop = int(st.cost)  # st.cost is max(m,n) in remap mode
            retained_total -= drop
            dense_removed_stage = int(initial_retained - retained_total)
            dense_effects[st.key] += drop
            dense_delta = drop
            cost = drop
        else:
            dense_delta = st.effective_dense_delta_if_drop()
            dense_removed_stage += dense_delta
            dense_effects[st.key] += dense_delta
            cost = max(1, int(st.dense_per_sv()))

        selected.append(StageSelection(name=st.key, delta=delta, cost=st.cost, dense_cost=cost))

        if remap:
            remaining_to_goal = max(0, int(initial_retained - int(target_retained) - prev_dense))
            upd = min(dense_delta, remaining_to_goal)
            stage_pbar.update(max(0, int(upd)))
            stage_pbar.set_postfix_str(f"{min(1.0, dense_removed_stage / max(1, stage_budget)):.2%}")
        else:
            stage_pbar.update(max(0, min(dense_delta, stage_budget - prev_dense)))
            stage_pbar.set_postfix_str(f"{min(1.0, dense_removed_stage / max(1, stage_budget)):.2%}")

        if st.has_candidate():
            if selection_mode in ("zero_sum", "only_delta_ce_zerosum"):
                # continue to use CandidateQueues in both zero_sum and only_delta_ce_zerosum
                new_delta = st.peek_delta()
                if new_delta is not None:
                    queues.push(st, new_delta)
            elif selection_mode != "negative_sum_no_order":
                # negative_sum_no_order already has all SVs in heap from start
                push_smallest(st)

    stage_pbar.close()
    return selected, running_sum, int(dense_removed_stage), dense_effects


def build_stage_rank_snapshot(states: Dict[str, ModulePruneState]) -> List[Dict]:
    """
    Build list of rank info dicts for logging.

    Args:
        states: Dict of ModulePruneState objects

    Returns:
        List of dicts with module rank information
    """
    snapshot = []
    for key in sorted(states.keys()):
        state = states[key]
        full_rank = min(state.shape)
        current_rank = max(0, state.current_rank())
        snapshot.append({
            "module": key,
            "current_rank": current_rank,
            "full_rank": full_rank,
            "shape": [state.shape[0], state.shape[1]],
        })
    return snapshot


def compute_revert_info(
    states: Dict[str, ModulePruneState],
    stage_counts: Dict[str, int],
    planned_ranks: Dict[str, int],
) -> Tuple[Set[str], List[Dict], Dict[str, int]]:
    """
    Check planned ranks against r_star and revert if needed.

    If a planned target rank exceeds r_star for a module, reset that
    module to full rank to avoid over-compression.

    Args:
        states: Dict of ModulePruneState objects
        stage_counts: Dict of counts per module for this stage
        planned_ranks: Dict of planned target ranks

    Returns:
        Tuple of (reverted_keys, stage_revert_log, updated_planned_ranks)
    """
    reverted_keys: Set[str] = set()
    stage_revert_log = []

    for key in stage_counts:
        state = states[key]
        full_rank = min(state.shape)
        target_rank = planned_ranks[key]
        if target_rank > state.r_star:
            revert_entry = {
                "module": key,
                "planned_target_rank": target_rank,
                "r_star": state.r_star,
                "full_rank": full_rank,
                "message": f"{key}: planned target {target_rank} is above r* {state.r_star}, resetting to full rank {full_rank}"
            }
            stage_revert_log.append(revert_entry)
            planned_ranks[key] = full_rank
            reverted_keys.add(key)
            state.consumed = 0
            state.keep_mask = [True] * state.sigma.numel()
            states[key] = state

    return reverted_keys, stage_revert_log, planned_ranks


def log_module_distribution(args, model, states: Dict[str, ModulePruneState]) -> None:
    """
    Log count of low-rank vs dense modules.

    Args:
        args: Compression arguments
        model: The language model
        states: Dict of ModulePruneState objects
    """
    lowrank_count = 0
    dense_count = 0
    for key in states.keys():
        try:
            mod, _, _ = get_module_from_key(args.model, model, key)
            if isinstance(mod, LowRankLinear):
                lowrank_count += 1
            else:
                dense_count += 1
        except Exception:
            pass
    print(f"   📊 Module distribution: {lowrank_count} low-rank, {dense_count} dense")
