#!/bin/bash
################################################################################
# One-Click GLUE Benchmark Pipeline
# 一键式 GLUE 评测脚本 - 可直接在其他设备运行
#
# 功能：
#   1. 环境检查
#   2. 模型压缩
#   3. 多任务微调
#   4. 结果汇总
#
# 使用方法：
#   bash eval_encoder/scripts/one_click_glue.sh
#
# 配置方式：修改下面的配置变量，或通过环境变量覆盖：
#   METHOD=fwsvd RANK=256 bash eval_encoder/scripts/one_click_glue.sh
################################################################################

set -e  # Exit on error

# ═════════════════════════════════════════════════════════════════════════════
# 路径检测 - 自动适配脚本位置
# ═════════════════════════════════════════════════════════════════════════════

# 检测脚本所在目录并设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 如果在 eval_encoder/scripts/ 下，则切换到 eval_encoder/ 父目录
if [[ "$SCRIPT_DIR" == */eval_encoder/scripts ]]; then
    cd "$SCRIPT_DIR/../.."  # 切到 lowrankarena/
    EVAL_ENCODER_PATH="eval_encoder"
elif [[ "$SCRIPT_DIR" == */eval_encoder ]]; then
    cd "$SCRIPT_DIR/.."  # 切到 lowrankarena/
    EVAL_ENCODER_PATH="eval_encoder"
else
    # 假设在 lowrankarena/ 根目录
    EVAL_ENCODER_PATH="eval_encoder"
fi

# ═════════════════════════════════════════════════════════════════════════════
# 配置区域 - 根据需要修改
# ═════════════════════════════════════════════════════════════════════════════

# 压缩配置
METHOD="${METHOD:-fwsvd}"              # 压缩方法: svd, fwsvd, drone, adasvd
RANK="${RANK:-}"                       # SVD 秩 (与 RETENTION 互斥，留空使用 RETENTION)
                                       # 当 RANK_ATTN/RANK_FFN/RANK_WO 未设置时，作为全局默认值
RANK_ATTN="${RANK_ATTN:-}"             # Attention 层专用秩 (Q/K/V)，覆盖 RANK
RANK_FFN="${RANK_FFN:-}"               # FFN 层专用秩 (Wi/Wo)，覆盖 RANK
RANK_WO="${RANK_WO:-}"                 # Attention output projection 专用秩，覆盖 RANK
RETENTION="${RETENTION:-}"             # 保有率 (0.0-1.0)，例如 0.5 表示保留 50% 的维度
                                       # BERT-base: 0.5→rank=384, 0.3→rank=230
                                       # 如果 RANK 和 RETENTION 都未设置，默认 rank=300
QKV_MODE="${QKV_MODE:-per_head}"       # QKV 分解模式: per_head (每头分解), full (全矩阵分解)
                                       # per_head: 对每个 attention head 单独 SVD (rank 限制到 head_dim=64)
                                       # full: 对整个 768x768 矩阵 SVD (论文风格, rank 可达 256+)
                                       # 注意: FlashSVD 后端仅支持 per_head 模式
CALIB_BATCHES="${CALIB_BATCHES:-4}"    # 校准批次数 (fwsvd/drone/adasvd)
                                       # 建议: 4-16 (快速) 或 16-32 (更好的 Fisher 估计)
BUDGET="${BUDGET:-0.6}"                # AdaSVD 预算 (仅当 METHOD=adasvd 时使用)
                                       # 推荐: 0.5 (保留50%参数) 或 0.6 (保留60%参数)
                                       # 注意: 0.3太激进，准确率会显著下降
ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES:-4000}"  # ARS 校准样本数 (paper: ~4000)
ADASVD_STEPS="${ADASVD_STEPS:-800}"                   # ARS 超网络训练步数 (paper: 800)
BACKEND="${BACKEND:-naive}"            # 后端: naive, flashsvd, flashsvd15
                                       # flashsvd15 后端建议使用 bf16 获得真实加速效果
DTYPE="${DTYPE:-fp32}"                 # 模型精度: fp32, fp16, bf16
                                       # --dtype bf16 + --backend flashsvd15 = 零 cast overhead
NON_INTERACTIVE="${NON_INTERACTIVE:-true}"   # 非交互模式（默认true，跳过所有提示）
AUTO_FIGURES="${AUTO_FIGURES:-false}"  # 实验完成后自动收集结果并重新生成图表

# 模型配置
MODEL_ID="${MODEL_ID:-bert-base-uncased}"
USE_TASK_MODELS="${USE_TASK_MODELS:-true}"  # 默认使用任务特定的预训练模型
TASK_MODEL_PREFIX="${TASK_MODEL_PREFIX:-textattack}"  # 任务模型前缀（textattack 或 howey）
LOCAL_PRETRAINED_DIR="${LOCAL_PRETRAINED_DIR:-}"      # 本地预训练模型目录（优先于 HuggingFace）
                                                      # 格式: {dir}/{task}/pretrained_base/
                                                      # 示例: LOCAL_PRETRAINED_DIR=eval_encoder/models

# 任务配置 (根据可用时间选择)
# 选项 1: 快速测试 (约 1.5 小时)
# TASKS="cola sst2 mrpc rte"

# 选项 2: 标准评测 (约 6 小时)
# TASKS="cola sst2 mrpc qnli rte stsb"

# 选项 3: 完整 GLUE (约 16 小时)
TASKS="${TASKS:-cola sst2 mrpc qqp mnli qnli rte stsb}"

# 训练配置
SKIP_FINETUNING="${SKIP_FINETUNING:-false}"   # 默认进行微调 (设为 true 可跳过微调,仅评估)
REUSE_CHECKPOINT="${REUSE_CHECKPOINT:-true}"  # 自动重用现有 checkpoint (docker 环境必需)
PRETRAIN_BEFORE_COMPRESS="${PRETRAIN_BEFORE_COMPRESS:-false}"  # 先微调base model再压缩再微调
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"                # Batch size (建议 32 以提高吞吐量)
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SEQ_LEN="${SEQ_LEN:-512}"
SEED="${SEED:-42}"

# 输出配置
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_ENCODER_PATH}/glue_results}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/encoder_runs.csv}"
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
# 辅助函数
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
# 步骤 0: 环境检查
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
# 主流程
# ═════════════════════════════════════════════════════════════════════════════

main() {
    print_header "One-Click GLUE Benchmark Pipeline"

    # 创建输出目录
    mkdir -p "${OUTPUT_DIR}/logs"
    mkdir -p "${EVAL_ENCODER_PATH}/models"

    # 打印配置
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

    # 询问确认（非交互模式跳过）
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

    # 开始记录日志
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1

    echo "Starting pipeline at $(date)"
    echo "All output will be logged to: $LOG_FILE"

    # 环境检查（被 compare_all_methods.sh 调用时跳过，避免重复 check）
    if [[ "${SKIP_ENV_CHECK:-false}" != "true" ]]; then
        check_environment
    fi

    # 估算时间
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

    # 运行 pipeline
    print_section "Starting GLUE Pipeline"

    # Build command with conditional rank/retention
    CMD="python ${EVAL_ENCODER_PATH}/glue_pipeline.py \
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

    # 完成
    print_header "Pipeline Complete!"

    echo "Results saved to:"
    echo "  - JSON: ${OUTPUT_DIR}/glue_results_${METHOD}_${BACKEND}_*.json"
    echo "  - Log:  ${LOG_FILE}"
    echo ""
    echo "To view results:"
    echo "  cat ${OUTPUT_DIR}/glue_results_${METHOD}_${BACKEND}_*.json | python -m json.tool"
    echo ""

    # 自动出图
    if [ "$AUTO_FIGURES" = "true" ]; then
        print_section "Auto-generating figures"
        FIGURES_DIR="${EVAL_ENCODER_PATH}/eval_results"
        python "${FIGURES_DIR}/collect_glue_results.py" && \
        python "${FIGURES_DIR}/gen_figures.py" && \
        echo "✅ Figures updated: ${FIGURES_DIR}/figures/" || \
        echo "⚠️  Figure generation failed (results still saved)"
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# 信号处理
# ═════════════════════════════════════════════════════════════════════════════

trap 'echo ""; echo "⚠️  Pipeline interrupted by user"; exit 130' INT TERM

# ═════════════════════════════════════════════════════════════════════════════
# 执行
# ═════════════════════════════════════════════════════════════════════════════

main "$@"
