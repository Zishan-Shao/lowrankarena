#!/bin/bash
################################################################################
# One-Click GLUE Benchmark Pipeline
# One-click GLUE evaluation script — can be run directly on any device
#
# Features:
#   1. Environment check
#   2. Model compression
#   3. Multi-task fine-tuning
#   4. Result aggregation
#
# Usage:
#   bash benchmark/one_click_glue.sh
#
# Configuration: modify the variables below, or override via environment variables:
#   METHOD=fwsvd RANK=256 bash benchmark/one_click_glue.sh
################################################################################

set -e  # Exit on error

# ═════════════════════════════════════════════════════════════════════════════
# Path detection - auto-detect script location
# ═════════════════════════════════════════════════════════════════════════════

# Detect the directory containing this script and set the working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If inside benchmark/, switch to the parent directory (eval_encoder/)
if [[ "$SCRIPT_DIR" == */benchmark ]]; then
    cd "$(dirname "$SCRIPT_DIR")"
fi
# ═════════════════════════════════════════════════════════════════════════════
# Configuration section - modify as needed
# ═════════════════════════════════════════════════════════════════════════════

# Compression configuration
METHOD="${METHOD:-fwsvd}"              # Compression method: svd, fwsvd, drone, adasvd
RANK="${RANK:-}"                       # SVD rank (mutually exclusive with RETENTION; leave empty to use RETENTION)
                                       # Used as global default when RANK_ATTN/RANK_FFN/RANK_WO are not set
RANK_ATTN="${RANK_ATTN:-}"             # Attention-layer-specific rank (Q/K/V), overrides RANK
RANK_FFN="${RANK_FFN:-}"               # FFN-layer-specific rank (Wi/Wo), overrides RANK
RANK_WO="${RANK_WO:-}"                 # Attention output projection rank, overrides RANK
RETENTION="${RETENTION:-}"             # Retention ratio (0.0-1.0), e.g. 0.5 retains 50% of dimensions
                                       # BERT-base: 0.5→rank=384, 0.3→rank=230
                                       # If both RANK and RETENTION are unset, defaults to rank=300
QKV_MODE="${QKV_MODE:-per_head}"       # QKV decomposition mode: per_head (per-head SVD), full (full-matrix SVD)
                                       # per_head: separate SVD per attention head (rank limited to head_dim=64)
                                       # full: SVD on the full 768x768 matrix (paper style, rank can reach 256+)
                                       # Note: FlashSVD backend only supports per_head mode
CALIB_BATCHES="${CALIB_BATCHES:-4}"    # Number of calibration batches (fwsvd/drone/adasvd)
CALIB_TASK="${CALIB_TASK:-}"           # Override calibration task (e.g. mnli for hans/anli which have no train split)
                                       # Recommended: 4-16 (fast) or 16-32 (better Fisher estimation)
BUDGET="${BUDGET:-0.6}"                # AdaSVD budget (only used when METHOD=adasvd)
                                       # Recommended: 0.5 (retain 50% params) or 0.6 (retain 60% params)
                                       # Note: 0.3 is too aggressive and significantly degrades accuracy
ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES:-4000}"  # ARS calibration samples (paper: ~4000)
ADASVD_STEPS="${ADASVD_STEPS:-800}"                   # ARS hypernetwork training steps (paper: 800)
BACKEND="${BACKEND:-naive}"            # Backend: naive, flashsvd, flashsvd15
                                       # flashsvd15 backend is recommended with bf16 for real speedup
DTYPE="${DTYPE:-fp32}"                 # Model dtype: fp32, fp16, bf16
                                       # --dtype bf16 + --backend flashsvd15 = zero cast overhead
NON_INTERACTIVE="${NON_INTERACTIVE:-true}"   # Non-interactive mode (default true, skips all prompts)
AUTO_FIGURES="${AUTO_FIGURES:-false}"  # Automatically collect results and regenerate figures after experiments

# Model configuration
MODEL_ID="${MODEL_ID:-bert-base-uncased}"
USE_TASK_MODELS="${USE_TASK_MODELS:-true}"  # Use task-specific pre-trained models by default
TASK_MODEL_PREFIX="${TASK_MODEL_PREFIX:-textattack}"  # Task model prefix (textattack or howey)
LOCAL_PRETRAINED_DIR="${LOCAL_PRETRAINED_DIR:-}"      # Local pre-trained model directory (takes priority over HuggingFace)
                                                      # Format: {dir}/{task}/pretrained_base/
                                                      # Example: LOCAL_PRETRAINED_DIR=compressed_models/bert

# Task configuration (choose based on available time)
# Option 1: Quick test (~1.5 hours)
# TASKS="cola sst2 mrpc rte"

# Option 2: Standard evaluation (~6 hours)
# TASKS="cola sst2 mrpc qnli rte stsb"

# Option 3: Full GLUE (~16 hours)
TASKS="${TASKS:-cola sst2 mrpc qqp mnli qnli rte stsb}"

# Training configuration
SKIP_FINETUNING="${SKIP_FINETUNING:-false}"   # Fine-tune by default (set to true to skip fine-tuning and evaluate only)
REUSE_CHECKPOINT="${REUSE_CHECKPOINT:-true}"  # Automatically reuse existing checkpoints (required in Docker environments)
PRETRAIN_BEFORE_COMPRESS="${PRETRAIN_BEFORE_COMPRESS:-false}"  # Fine-tune base model before compression, then fine-tune again
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"                # Batch size (32 recommended for higher throughput)
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SEQ_LEN="${SEQ_LEN:-512}"
SEED="${SEED:-42}"

# Output configuration — namespace by model slug so BERT and ModernBERT don't collide
_MODEL_SLUG_OCG="${MODEL_ID##*/}"
_MODEL_SLUG_OCG="${_MODEL_SLUG_OCG,,}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/glue/${_MODEL_SLUG_OCG}}"
OUT_CSV="${OUT_CSV:-experiments/encoder_runs.csv}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Build rank label for log filename:
# - If component ranks specified, use ra/rf/rw format
# - Else if global RANK set, use rRANK
# - Else if RETENTION set, use retRETENTION
# - Else "auto"
if [ -n "$RANK_ATTN" ] || [ -n "$RANK_FFN" ] || [ -n "$RANK_WO" ]; then
    _RA="${RANK_ATTN:-${RANK:-auto}}"
    _RF="${RANK_FFN:-${RANK:-auto}}"
    _RW="${RANK_WO:-${RANK:-auto}}"
    _RANK_LABEL="ra${_RA}_rf${_RF}_rw${_RW}"
else
    _RANK_LABEL="${RANK:-${RETENTION:+ret${RETENTION}}}"
fi
RUN_NAME="${METHOD}_${_RANK_LABEL:-rauto}_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/logs/${RUN_NAME}.log"

# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║ $1"
    printf "║ %-66s ║\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_section() {
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  $1"
    echo "────────────────────────────────────────────────────────────────────"
    echo ""
}

print_error() {
    echo ""
    echo "❌ ERROR: $1"
    echo ""
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "Required command '$1' not found. Please install it first."
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# Step 0: Environment Check
# ═════════════════════════════════════════════════════════════════════════════

check_environment() {
    print_section "Step 0: Environment Check"

    echo "Checking required commands..."
    check_command python
    check_command nvidia-smi

    echo "✓ Python found: $(python --version)"
    echo "✓ NVIDIA GPU found"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

    echo ""
    echo "Checking Python packages..."
    python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || print_error "PyTorch not installed"
    python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" || print_error "Transformers not installed"
    python -c "import datasets; print(f'✓ Datasets: {datasets.__version__}')" || print_error "Datasets not installed"

    echo ""
    echo "Checking CUDA availability..."
    python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" || print_error "CUDA not available"
    python -c "import torch; print(f'✓ CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

    echo ""
    echo "✅ Environment check passed!"
}

# ═════════════════════════════════════════════════════════════════════════════
# Main flow
# ═════════════════════════════════════════════════════════════════════════════

main() {
    print_header "One-Click GLUE Benchmark Pipeline"

    # Create output directory
    mkdir -p "${OUTPUT_DIR}/logs"


    # Print configuration
    cat << EOF
Configuration:
──────────────────────────────────────
  Method:        $METHOD
  Rank (global): ${RANK:-N/A}
  Rank Attn:     ${RANK_ATTN:-${RANK:-N/A}}
  Rank FFN:      ${RANK_FFN:-${RANK:-N/A}}
  Rank Wo:       ${RANK_WO:-${RANK:-N/A}}
  Retention:     ${RETENTION:-N/A}
  QKV Mode:      $QKV_MODE
  Calib Batches: $CALIB_BATCHES
  Backend:       $BACKEND
  Dtype:         $DTYPE
  Model:         $MODEL_ID
  Use Task Models: $USE_TASK_MODELS
  Tasks:         $TASKS
  Skip Finetuning: $SKIP_FINETUNING
  Epochs:        $NUM_EPOCHS
  Batch Size:    $BATCH_SIZE
  Learning Rate: $LEARNING_RATE
  Output:        $OUTPUT_DIR
  Log:           $LOG_FILE
──────────────────────────────────────

EOF

    # Ask for confirmation (skipped in non-interactive mode)
    if [ "$NON_INTERACTIVE" != "true" ]; then
        read -p "Continue with this configuration? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted by user."
            exit 0
        fi
    else
        echo "[non-interactive] Auto-continuing with configuration..."
    fi

    # Start logging
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1

    echo "Starting pipeline at $(date)"
    echo "All output will be logged to: $LOG_FILE"

    # Environment check (skipped when called from compare_all_methods.sh to avoid redundant checks)
    if [[ "${SKIP_ENV_CHECK:-false}" != "true" ]]; then
        check_environment
    fi

    # Time estimation
    print_section "Time Estimation"
    TASK_COUNT=$(echo $TASKS | wc -w)
    echo "Tasks to run: $TASK_COUNT"
    if [ "$SKIP_FINETUNING" = "true" ]; then
        echo "Mode: Compression + Evaluation only (no fine-tuning)"
        echo "Estimated time per task: 5-10 minutes"
        echo "Total estimated time: ~$((TASK_COUNT * 10)) minutes"
    else
        echo "Mode: Compression + Fine-tuning + Evaluation"
        echo "Estimated time per task: 1-2 hours (depends on dataset size)"
        echo "Total estimated time: $((TASK_COUNT * 1))-$((TASK_COUNT * 2)) hours"
    fi
    echo ""
    if [ "$NON_INTERACTIVE" != "true" ]; then
        read -p "Press Enter to start..."
    else
        echo "[non-interactive] Starting pipeline..."
    fi

    # Run pipeline
    print_section "Starting GLUE Pipeline"

    # Build command with conditional rank/retention
    CMD="python src/encoders/glue_pipeline.py \
        --method \"$METHOD\" \
        --backend \"$BACKEND\" \
        --dtype \"$DTYPE\" \
        --model_id \"$MODEL_ID\" \
        --tasks $TASKS \
        --num_epochs \"$NUM_EPOCHS\" \
        --batch_size \"$BATCH_SIZE\" \
        --learning_rate \"$LEARNING_RATE\" \
        --seq_len \"$SEQ_LEN\" \
        --seed \"$SEED\" \
        --output_dir \"$OUTPUT_DIR\" \
        --out_csv \"$OUT_CSV\""

    # Add component-specific ranks if specified
    if [ -n "$RANK_ATTN" ]; then
        CMD="$CMD --rank_attn $RANK_ATTN"
    fi
    if [ -n "$RANK_FFN" ]; then
        CMD="$CMD --rank_ffn $RANK_FFN"
    fi
    if [ -n "$RANK_WO" ]; then
        CMD="$CMD --rank_wo $RANK_WO"
    fi

    # Add base rank or retention (mutually exclusive)
    # Only add if component-specific ranks are not all specified
    if [ -z "$RANK_ATTN" ] || [ -z "$RANK_FFN" ] || [ -z "$RANK_WO" ]; then
        if [ -n "$RANK" ]; then
            CMD="$CMD --rank $RANK"
        elif [ -n "$RETENTION" ]; then
            CMD="$CMD --retention $RETENTION"
        fi
    fi

    # Add QKV mode and calibration batches
    CMD="$CMD --qkv_mode $QKV_MODE"
    CMD="$CMD --calib_batches $CALIB_BATCHES"
    [[ -n "$CALIB_TASK" ]] && CMD="$CMD --calib_task $CALIB_TASK"

    # Add budget and ARS params for AdaSVD (calib_batches is ignored for adasvd_origin)
    if [ "$METHOD" = "adasvd" ]; then
        CMD="$CMD --budget $BUDGET"
        CMD="$CMD --adasvd_calib_samples $ADASVD_CALIB_SAMPLES"
        CMD="$CMD --adasvd_steps $ADASVD_STEPS"
    fi

    # Add task-specific model flag
    # When LOCAL_PRETRAINED_DIR is set, also pass --use_task_models so run_pipeline()
    # uses per-task compression (not shared model). get_task_model_id() will then
    # prefer the local path over HuggingFace.
    if [ -n "$LOCAL_PRETRAINED_DIR" ]; then
        CMD="$CMD --local_pretrained_dir $LOCAL_PRETRAINED_DIR --use_task_models"
    elif [ "$USE_TASK_MODELS" = "true" ]; then
        CMD="$CMD --use_task_models --task_model_prefix $TASK_MODEL_PREFIX"
    fi

    # Add pipeline control flags
    if [ "$SKIP_FINETUNING" = "true" ]; then
        CMD="$CMD --skip_finetuning"
    fi

    if [ "$REUSE_CHECKPOINT" = "true" ]; then
        CMD="$CMD --reuse_checkpoint"
    fi

    if [ "$PRETRAIN_BEFORE_COMPRESS" = "true" ]; then
        CMD="$CMD --pretrain_before_compress"
    fi

    eval $CMD

    # Done
    print_header "Pipeline Complete!"

    echo "Results saved to:"
    echo "  - JSON: ${OUTPUT_DIR}/glue_results_${METHOD}_${BACKEND}_*.json"
    echo "  - Log:  ${LOG_FILE}"
    echo ""
    echo "To view results:"
    echo "  cat ${OUTPUT_DIR}/glue_results_${METHOD}_${BACKEND}_*.json | python -m json.tool"
    echo ""

    # Auto-generate figures
    if [ "$AUTO_FIGURES" = "true" ]; then
        print_section "Auto-generating figures"
        FIGURES_DIR="benchmark"
        python "benchmark/analysis/collect_glue_results.py" && \
        python "benchmark/figures/gen_figures.py" 2>/dev/null && \
        echo "✅ Figures updated: ${FIGURES_DIR}/figures/" || \
        echo "⚠️  Figure generation failed (results still saved)"
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# Signal handling
# ═════════════════════════════════════════════════════════════════════════════

trap 'echo ""; echo "WARNING: Pipeline interrupted by user"; exit 130' INT TERM

# ═════════════════════════════════════════════════════════════════════════════
# Execute
# ═════════════════════════════════════════════════════════════════════════════

main "$@"
