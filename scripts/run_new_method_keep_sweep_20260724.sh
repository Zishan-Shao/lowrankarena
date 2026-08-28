#!/usr/bin/env bash
set -euo pipefail

repo_root="${LOWRANKARENA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${LOWRANKARENA_PYTHON:-python}"
output_root="${LOWRANKARENA_OUTPUT_ROOT:-/tmp/lowrankarena_new_methods}"
llama7_model="${LOWRANKARENA_LLAMA7_MODEL:-jeffwan/llama-7b-hf}"
llama31_model="${LOWRANKARENA_LLAMA31_MODEL:-meta-llama/Llama-3.1-8B}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

model_path() {
    case "$1" in
        llama7) printf '%s\n' "$llama7_model" ;;
        llama31_8b) printf '%s\n' "$llama31_model" ;;
        *) echo "unknown model tag: $1" >&2; return 2 ;;
    esac
}

keep_label() {
    case "$1" in
        0.7) printf '70\n' ;;
        0.6) printf '60\n' ;;
        0.5) printf '50\n' ;;
        0.4) printf '40\n' ;;
        *) echo "unsupported keep ratio: $1" >&2; return 2 ;;
    esac
}

prune_ratio() {
    case "$1" in
        0.7) printf '0.3\n' ;;
        0.6) printf '0.4\n' ;;
        0.5) printf '0.5\n' ;;
        0.4) printf '0.6\n' ;;
        *) echo "unsupported keep ratio: $1" >&2; return 2 ;;
    esac
}

run_zssvd() {
    local tag="$1"
    local keep="$2"
    local gpu="$3"
    local label prune base run_dir final_ckpt
    label="$(keep_label "$keep")"
    prune="$(prune_ratio "$keep")"
    base="$(model_path "$tag")"
    run_dir="$output_root/$tag/zssvd_wiki_keep$label"
    mkdir -p "$run_dir"

    if [[ ! -f "$run_dir/checkpoint_hf/lowrankarena_method.json" ]]; then
        if ! final_ckpt="$(find "$run_dir" -type f -name 'final_ppl*_fp16_compressed.pt' -print -quit)" || [[ -z "$final_ckpt" ]]; then
            echo "[$(date --iso-8601=seconds)] ZS-SVD start model=$tag keep=$keep gpu=$gpu" | tee -a "$run_dir/worker.log"
            (
                cd "$repo_root/compress/svd/Zero-Sum-SVD"
                CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u main_zero_sum.py \
                    --model "$base" \
                    --save_path "$run_dir" \
                    --dataset wikitext2 \
                    --global_prune_ratio "$prune" \
                    --keep_rank_ratio 0 \
                    --num_stages 1 \
                    --nsamples 256 \
                    --nsamples_gradient_subset 1 \
                    --selection_mode zero_sum \
                    --importance_seq_len 2048 \
                    --sub_with_teacher_module \
                    --eval_ppl \
                    --final_eval_datasets wikitext2 \
                    --save_after_truncation \
                    --seed 3 \
                    --model_seq_len 2048 \
                    --DEV cuda 2>&1 | tee -a "$run_dir/run.log"
            )
            final_ckpt="$(find "$run_dir" -type f -name 'final_ppl*_fp16_compressed.pt' -print -quit)"
        fi
        test -n "$final_ckpt"
        local overwrite=()
        if [[ -d "$run_dir/checkpoint_hf" ]] && [[ -n "$(find "$run_dir/checkpoint_hf" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            overwrite=(--unsafe-overwrite)
        fi
        (
            cd "$repo_root/compress/svd/Zero-Sum-SVD"
            "$python_bin" -u export_lowrank_hf.py \
                --checkpoint "$final_ckpt" \
                --output-dir "$run_dir/checkpoint_hf" \
                --keep-ratio "$keep" \
                --target-dtype float16 \
                --max-shard-size 5GB \
                "${overwrite[@]}" 2>&1 | tee "$run_dir/export.log"
        )
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$repo_root/scripts/validate_lowrank_artifact.py" \
        --artifact "$run_dir/checkpoint_hf" \
        --expected-keep "$keep" \
        --target-modules 224 \
        --keep-tolerance 0.001 2>&1 | tee "$run_dir/validate_forward.log"
    touch "$run_dir/SUCCESS"
    echo "[$(date --iso-8601=seconds)] ZS-SVD success model=$tag keep=$keep gpu=$gpu" | tee -a "$run_dir/worker.log"
}

swift_svd_file() {
    case "$1" in
        llama7) printf '%s\n' "$repo_root/compress/svd/Swift-SVD/svd_list/WikiText2/82eb0e6908390680598ca3ec1d77adfc5e1b24aa_WikiText2_svd_list_256_2048_s42.pk" ;;
        llama31_8b) printf '%s\n' "$repo_root/compress/svd/Swift-SVD/svd_list/WikiText2/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b_WikiText2_svd_list_256_2048_s42.pk" ;;
        *) echo "unknown model tag: $1" >&2; return 2 ;;
    esac
}

export_swiftsvd() {
    local tag="$1"
    local keep="$2"
    local label base svd_file run_dir allocation
    label="$(keep_label "$keep")"
    base="$(model_path "$tag")"
    svd_file="$(swift_svd_file "$tag")"
    run_dir="$output_root/$tag/swiftsvd_wiki_keep$label"
    allocation="$run_dir/rank_allocation_uniform_keep$label.pk"
    mkdir -p "$run_dir"
    test -f "$svd_file"

    if [[ ! -f "$allocation" ]]; then
        (
            cd "$repo_root/compress/svd/Swift-SVD"
            "$python_bin" -u uniform_rank_allocation.py \
                --local_model_path "$base" \
                --svd_file "$svd_file" \
                --compression_ratio "$keep" \
                --output_file "$allocation" 2>&1 | tee "$run_dir/rank_allocation.log"
        )
    fi
    if [[ ! -f "$run_dir/checkpoint_hf/lowrankarena_method.json" ]]; then
        local overwrite=()
        if [[ -d "$run_dir/checkpoint_hf" ]] && [[ -n "$(find "$run_dir/checkpoint_hf" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            overwrite=(--unsafe-overwrite)
        fi
        (
            cd "$repo_root/compress/svd/Swift-SVD"
            "$python_bin" -u export_lowrank_hf.py \
                --base-model "$base" \
                --svd-file "$svd_file" \
                --rank-allocation "$allocation" \
                --output-dir "$run_dir/checkpoint_hf" \
                --keep-ratio "$keep" \
                --target-dtype float16 \
                --max-shard-size 5GB \
                "${overwrite[@]}" 2>&1 | tee "$run_dir/export.log"
        )
    fi
    echo "[$(date --iso-8601=seconds)] Swift-SVD export success model=$tag keep=$keep" | tee -a "$run_dir/worker.log"
}

validate_swiftsvd() {
    local tag="$1"
    local keep="$2"
    local gpu="$3"
    local label run_dir
    label="$(keep_label "$keep")"
    run_dir="$output_root/$tag/swiftsvd_wiki_keep$label"
    test -f "$run_dir/checkpoint_hf/lowrankarena_method.json"

    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$repo_root/scripts/validate_lowrank_artifact.py" \
        --artifact "$run_dir/checkpoint_hf" \
        --expected-keep "$keep" \
        --target-modules 224 \
        --keep-tolerance 0.001 \
        --wiki-ppl \
        --ppl-batch-size 16 \
        --output-json "$run_dir/validation_and_wiki_ppl.json" \
        2>&1 | tee "$run_dir/validation_and_wiki_ppl.log"
    touch "$run_dir/SUCCESS"
    echo "[$(date --iso-8601=seconds)] Swift-SVD validation success model=$tag keep=$keep gpu=$gpu" | tee -a "$run_dir/worker.log"
}

case "${1:-}" in
    zssvd)
        run_zssvd "$2" "$3" "$4"
        ;;
    zssvd-worker)
        gpu="$2"
        shift 2
        for spec in "$@"; do
            IFS=: read -r tag keep <<<"$spec"
            run_zssvd "$tag" "$keep" "$gpu"
        done
        ;;
    swiftsvd)
        export_swiftsvd "$2" "$3"
        ;;
    swiftsvd-worker)
        shift
        for spec in "$@"; do
            IFS=: read -r tag keep <<<"$spec"
            export_swiftsvd "$tag" "$keep"
        done
        ;;
    swiftsvd-validate)
        validate_swiftsvd "$2" "$3" "$4"
        ;;
    swiftsvd-validate-worker)
        gpu="$2"
        shift 2
        for spec in "$@"; do
            IFS=: read -r tag keep <<<"$spec"
            validate_swiftsvd "$tag" "$keep" "$gpu"
        done
        ;;
    *)
        echo "usage: $0 {zssvd MODEL KEEP GPU | zssvd-worker GPU MODEL:KEEP... | swiftsvd MODEL KEEP | swiftsvd-worker MODEL:KEEP... | swiftsvd-validate MODEL KEEP GPU | swiftsvd-validate-worker GPU MODEL:KEEP...}" >&2
        exit 2
        ;;
esac
