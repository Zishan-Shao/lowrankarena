#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <llama1_7b|llama31_8b> [targets_csv]" >&2
  exit 1
fi

MODEL_KEY="$1"
TARGETS_CSV="${2:-0.8,0.7,0.6,0.5,0.4}"
ROOT="/deac/csc/yangGrp/cuij/LLM/BlockPruner"
CONDA_SH="/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh"

case "$MODEL_KEY" in
  llama1_7b)
    MODEL_PATH="/deac/csc/yangGrp/cuij/LLM/models/hf_models/huggyllama__llama-7b"
    MODEL_BASENAME="huggyllama__llama-7b"
    ;;
  llama31_8b)
    MODEL_PATH="/deac/csc/yangGrp/cuij/LLM/models/hf_models/meta-llama__Llama-3.1-8B"
    MODEL_BASENAME="meta-llama__Llama-3.1-8B"
    ;;
  *)
    echo "unsupported model key: $MODEL_KEY" >&2
    exit 1
    ;;
esac

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/formal_${MODEL_KEY}_${RUN_STAMP}"
RESULT_ROOT="$ROOT/results/formal/$MODEL_KEY"
SEARCH_DIR="$RESULT_ROOT/search"
CALIB_DIR="$RESULT_ROOT/calibration"
REPO_DIR="$RESULT_ROOT/repo_native"
STD_DIR="$RESULT_ROOT/standardized"
mkdir -p "$LOG_DIR" "$SEARCH_DIR" "$CALIB_DIR" "$REPO_DIR" "$STD_DIR"

SEARCH_FILE="$SEARCH_DIR/${MODEL_BASENAME}_mix_alpaca_ns_256_del_order_list.json"
CALIB_JSON="$CALIB_DIR/${MODEL_KEY}_keep_ratio_calibration.json"

source "$CONDA_SH"
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

cd "$ROOT"

echo "[pipeline] model_key=$MODEL_KEY"
echo "[pipeline] model_path=$MODEL_PATH"
echo "[pipeline] log_dir=$LOG_DIR"
echo "[pipeline] result_root=$RESULT_ROOT"
nvidia-smi || true

if [[ -f "$SEARCH_FILE" ]]; then
  echo "[search] skip existing $SEARCH_FILE"
else
  echo "[search] start $(date)"
  conda activate dobisvd
  python block_search.py \
    --model-path "$MODEL_PATH" \
    --block-type mix \
    --cal-nsamples 256 \
    --del-block-num 64 \
    --cal-dataset alpaca \
    --ppl-search-path "$SEARCH_DIR" \
    --ppl-eval-batch-size 2 \
    --compute-dtype bf16 \
    --device cuda \
    2>&1 | tee "$LOG_DIR/search.log"
  echo "[search] done $(date)"
fi

conda activate dobisvd
python scripts/calibrate_keep_ratio.py \
  --model-path "$MODEL_PATH" \
  --ppl-search-file "$SEARCH_FILE" \
  --targets "$TARGETS_CSV" \
  --output-json "$CALIB_JSON" \
  2>&1 | tee "$LOG_DIR/calibration.log"

IFS=',' read -r -a TARGET_ARRAY <<< "$TARGETS_CSV"
for TARGET in "${TARGET_ARRAY[@]}"; do
  TARGET="$(echo "$TARGET" | xargs)"
  DEL_BLOCK_NUM="$(python - <<PY
import json
payload=json.load(open('$CALIB_JSON'))
needle=float('$TARGET')
for item in payload['target_mapping']:
    if abs(float(item['target_keep_ratio'])-needle) < 1e-9:
        print(int(item['selected_del_block_num']))
        break
PY
)"
  ACH_KEEP="$(python - <<PY
import json
payload=json.load(open('$CALIB_JSON'))
needle=float('$TARGET')
for item in payload['target_mapping']:
    if abs(float(item['target_keep_ratio'])-needle) < 1e-9:
        print(float(item['achieved_keep_ratio']))
        break
PY
)"
  ACH_PRUNE="$(python - <<PY
import json
payload=json.load(open('$CALIB_JSON'))
needle=float('$TARGET')
for item in payload['target_mapping']:
    if abs(float(item['target_keep_ratio'])-needle) < 1e-9:
        print(float(item['achieved_prune_ratio']))
        break
PY
)"

  echo "[target] requested_keep=$TARGET del_block_num=$DEL_BLOCK_NUM achieved_keep=$ACH_KEEP achieved_prune=$ACH_PRUNE"

  REPO_JSON="$REPO_DIR/${MODEL_KEY}_keep_${TARGET}_repo_native.json"
  if [[ -f "$REPO_JSON" ]]; then
    echo "[repo-native] skip existing $REPO_JSON"
  else
    conda activate dobisvd
    python eval.py \
      --do-eval \
      --model-path "$MODEL_PATH" \
      --del-block-num "$DEL_BLOCK_NUM" \
      --cal-dataset wikitext2 \
      --ppl-search-file "$SEARCH_FILE" \
      --ppl-eval-batch-size 1 \
      --batch-size 1 \
      --tasks piqa winogrande hellaswag arc_easy arc_challenge \
      --device cuda \
      --compute-dtype bf16 \
      --label "${MODEL_KEY}_keep_${TARGET}_repo_native" \
      --output-json "$REPO_JSON" \
      2>&1 | tee "$LOG_DIR/repo_native_keep_${TARGET}.log"
  fi

  PPL_JSON="$STD_DIR/${MODEL_KEY}_keep_${TARGET}_ppl.json"
  MCQ_JSON="$STD_DIR/${MODEL_KEY}_keep_${TARGET}_7task.json"
  if [[ -f "$PPL_JSON" && -f "$MCQ_JSON" ]]; then
    echo "[standardized] skip existing keep=$TARGET"
  else
    conda activate lowrankarena
    python scripts/eval_standardized.py \
      --model-path "$MODEL_PATH" \
      --ppl-search-file "$SEARCH_FILE" \
      --del-block-num "$DEL_BLOCK_NUM" \
      --requested-keep-ratio "$TARGET" \
      --achieved-keep-ratio "$ACH_KEEP" \
      --achieved-prune-ratio "$ACH_PRUNE" \
      --ppl-output-json "$PPL_JSON" \
      --mcq-output-json "$MCQ_JSON" \
      --device cuda:0 \
      --dtype bfloat16 \
      --ppl-batch-size 1 \
      --lm-eval-batch-size 1 \
      --c4-max-eval-tokens 262144 \
      2>&1 | tee "$LOG_DIR/standardized_keep_${TARGET}.log"
  fi

  conda activate lowrankarena
  python scripts/update_summary.py > "$LOG_DIR/update_summary_keep_${TARGET}.log" 2>&1 || true
done

conda activate lowrankarena
python scripts/update_summary.py 2>&1 | tee "$LOG_DIR/update_summary_final.log"
echo "[pipeline] done $(date)"
