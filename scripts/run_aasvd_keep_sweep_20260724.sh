#!/usr/bin/env bash
set -euo pipefail

repo_root="${LOWRANKARENA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
aa_root="$repo_root/compress/svd/AA-SVD"
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

model_config() {
    case "$1" in
        llama7) printf 'llama-7B\n' ;;
        llama31_8b) printf 'llama3-8B\n' ;;
        *) echo "unknown model tag: $1" >&2; return 2 ;;
    esac
}

keep_label() {
    case "$1" in
        0.8) printf '80\n' ;;
        0.7) printf '70\n' ;;
        0.6) printf '60\n' ;;
        0.5) printf '50\n' ;;
        0.4) printf '40\n' ;;
        *) echo "unsupported keep ratio: $1" >&2; return 2 ;;
    esac
}

run_dir_for() {
    local tag="$1"
    local keep="$2"
    local label
    label="$(keep_label "$keep")"
    if [[ "$tag" == llama7 && "$keep" == 0.8 ]]; then
        printf '%s\n' "$output_root/llama7/aasvd_keep80"
    else
        printf '%s\n' "$output_root/$tag/aasvd_wiki_keep$label"
    fi
}

run_aasvd() {
    local tag="$1"
    local keep="$2"
    local gpu="$3"
    local base hydra_model run_dir
    base="$(model_path "$tag")"
    hydra_model="$(model_config "$tag")"
    run_dir="$(run_dir_for "$tag" "$keep")"
    mkdir -p "$run_dir"

    if [[ ! -f "$run_dir/native_hf/model.safetensors.index.json" ]]; then
        echo "[$(date --iso-8601=seconds)] AA-SVD compression start model=$tag keep=$keep gpu=$gpu" | tee -a "$run_dir/worker.log"
        (
            cd "$aa_root"
            CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u main.py \
                "model=$hydra_model" \
                "model.name=$base" \
                model.dtype=bfloat16 \
                "compression.target_param_ratio=$keep" \
                compression.sub_method=obj2 \
                compression.dobi_remapping=false \
                compression.finetune.enabled=true \
                "compression.save_path=$run_dir/native_modules" \
                wandb.use=false \
                evaluate=null \
                "+save={dir:$run_dir,name:native_hf}" \
                "paths.output_dir=$run_dir/hydra_root" \
                "hydra.run.dir=$run_dir/hydra" \
                hydra.job.chdir=false \
                2>&1 | tee -a "$run_dir/run.log"
        )
    fi

    if [[ ! -f "$run_dir/checkpoint_hf/lowrankarena_method.json" ]]; then
        local overwrite=()
        if [[ -d "$run_dir/checkpoint_hf" ]] && [[ -n "$(find "$run_dir/checkpoint_hf" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            overwrite=(--unsafe-overwrite)
        fi
        (
            cd "$aa_root"
            "$python_bin" -u export_lowrank_hf.py \
                --native-checkpoint "$run_dir/native_hf" \
                --base-model "$base" \
                --output-dir "$run_dir/checkpoint_hf" \
                --keep-ratio "$keep" \
                --target-dtype float16 \
                --max-shard-size 5GB \
                "${overwrite[@]}" \
                2>&1 | tee "$run_dir/export.log"
        )
    fi

    jq -e \
        '.method == "AA-SVD" and .remapping == false and .quantization == false and .external_recovery == false' \
        "$run_dir/checkpoint_hf/lowrankarena_method.json" >/dev/null

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
    echo "[$(date --iso-8601=seconds)] AA-SVD success model=$tag keep=$keep gpu=$gpu" | tee -a "$run_dir/worker.log"
}

case "${1:-}" in
    run)
        run_aasvd "$2" "$3" "$4"
        ;;
    worker)
        gpu="$2"
        shift 2
        for spec in "$@"; do
            IFS=: read -r tag keep <<<"$spec"
            run_aasvd "$tag" "$keep" "$gpu"
        done
        ;;
    *)
        echo "usage: $0 {run MODEL KEEP GPU | worker GPU MODEL:KEEP...}" >&2
        exit 2
        ;;
esac
