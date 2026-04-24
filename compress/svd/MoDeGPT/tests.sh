MODEL_NAME="Qwen/Qwen3-32B"
TESTS_OUTPUT_DIR="$TMPDIR/compressed_output/$MODEL_NAME/"

export HF_HUB_ENABLE_HF_TRANSFER=0

# echo "Test 1"
# python -m src.run_modegpt \
#     --model "$MODEL_NAME" \
#     --device 0 \
#     --compression_ratio 0.30 \
#     --calib_size 128 \
#     --calibs_batch_size 16 \
#     --output_dir "$TESTS_OUTPUT_DIR" \
#     --note "qwen3 14-b" \
#     --order "mlp,qk,vo" \
#     --max_sparsity 0.95 \
#     --ridge_vo 1e-5 \
#     --ridge_qk 1e-4 \
#     --sparsity_smoothing 0.07629 \
#     --nystrom_ridge 1e-4 \
#     --dataset alpaca

# lm_eval --model hf \
#     --model_args pretrained="$TESTS_OUTPUT_DIR/model,dtype=bfloat16,trust_remote_code=True" \
#     --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \
#     --batch_size 128 \
#     --output_path ./metrics/lm_eval/ \
#     --num_fewshot 0 \
#     --metadata '{"note": "Qwen3-32B--0.3 sparsity"}'

# MODEL_NAME="Qwen/Qwen3-14B"
# TESTS_OUTPUT_DIR="$TMPDIR/compressed_output/$MODEL_NAME/"
# echo "Test 1"
# python -m src.run_modegpt \
#     --model "$MODEL_NAME" \
#     --device 0 \
#     --compression_ratio 0.40 \
#     --calib_size 128 \
#     --calibs_batch_size 16 \
#     --output_dir "$TESTS_OUTPUT_DIR" \
#     --note "qwen3 14-b" \
#     --order "mlp,qk,vo" \
#     --max_sparsity 0.95 \
#     --ridge_vo 9e-5 \
#     --ridge_qk 1e-5 \
#     --sparsity_smoothing 0.06672 \
#     --nystrom_ridge 1e-4 \
#     --dataset alpaca

# lm_eval --model hf \
#     --model_args pretrained="$TESTS_OUTPUT_DIR/model,dtype=bfloat16,trust_remote_code=True" \
#     --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \
#     --batch_size 128 \
#     --output_path ./metrics/lm_eval/ \
#     --num_fewshot 0 \
#     --metadata '{"note": "Qwen3-14B--0.4 sparsity"}'


# MODEL_NAME="Qwen/Qwen3-14B"
# TESTS_OUTPUT_DIR="$TMPDIR/compressed_output/$MODEL_NAME/"
# echo "Test 1"
# python -m src.run_modegpt \
#     --model "$MODEL_NAME" \
#     --device 0 \
#     --compression_ratio 0.30 \
#     --calib_size 128 \
#     --calibs_batch_size 16 \
#     --output_dir "$TESTS_OUTPUT_DIR" \
#     --note "qwen3 14-b" \
#     --order "mlp,qk,vo" \
#     --max_sparsity 0.95 \
#     --ridge_vo 9e-5 \
#     --ridge_qk 1e-5 \
#     --sparsity_smoothing 0.06672 \
#     --nystrom_ridge 1e-4 \
#     --dataset alpaca

# lm_eval --model hf \
#     --model_args pretrained="$TESTS_OUTPUT_DIR/model,dtype=bfloat16,trust_remote_code=True" \
#     --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \
#     --batch_size 128 \
#     --output_path ./metrics/lm_eval/ \
#     --num_fewshot 0 \
#     --metadata '{"note": "Qwen3-14B--0.3 sparsity"}'


MODEL_NAME="Qwen/Qwen3-8B"
TESTS_OUTPUT_DIR="$TMPDIR/compressed_output/$MODEL_NAME/"
echo "Test 1"
python -m src.run_modegpt \
    --model "$MODEL_NAME" \
    --device 0 \
    --compression_ratio 0.40 \
    --calib_size 128 \
    --calibs_batch_size 16 \
    --output_dir "$TESTS_OUTPUT_DIR" \
    --temp_storage_dir "$TMPDIR/layers/" \
    --note "qwen3 14-b" \
    --order "mlp,qk,vo" \
    --max_sparsity 0.95 \
    --ridge_vo 1e-5 \
    --ridge_qk 1e-2 \
    --sparsity_smoothing 0.04948 \
    --nystrom_ridge 1e-4 \
    --dataset alpaca

lm_eval --model hf \
    --model_args pretrained="$TESTS_OUTPUT_DIR/model,dtype=bfloat16,trust_remote_code=True" \
    --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \
    --batch_size 128 \
    --output_path ./metrics/lm_eval/ \
    --num_fewshot 0 \
    --metadata '{"note": "Qwen3-8B--0.4 sparsity"}'


MODEL_NAME="Qwen/Qwen3-8B"
TESTS_OUTPUT_DIR="$TMPDIR/compressed_output/$MODEL_NAME/"
echo "Test 1"
python -m src.run_modegpt \
    --model "$MODEL_NAME" \
    --device 0 \
    --compression_ratio 0.30 \
    --calib_size 128 \
    --calibs_batch_size 16 \
    --output_dir "$TESTS_OUTPUT_DIR" \
    --note "qwen3 14-b" \
    --order "mlp,qk,vo" \
    --max_sparsity 0.95 \
    --ridge_vo 1e-5 \
    --ridge_qk 1e-2 \
    --sparsity_smoothing 0.04948 \
    --nystrom_ridge 1e-4 \
    --dataset alpaca

lm_eval --model hf \
    --model_args pretrained="$TESTS_OUTPUT_DIR/model,dtype=bfloat16,trust_remote_code=True" \
    --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \
    --batch_size 128 \
    --output_path ./metrics/lm_eval/ \
    --num_fewshot 0 \
    --metadata '{"note": "Qwen3-8B--0.3 sparsity"}'








# MODEL_NAME="Qwen/Qwen3-8B"
# TESTS_OUTPUT_DIR="$TMPDIR/compressed_output/$MODEL_NAME/"
# echo "Test 2"
# python -m src.run_modegpt \
#     --model "$MODEL_NAME" \
#     --device 0 \
#     --compression_ratio 0.50 \
#     --calib_size 128 \
#     --calibs_batch_size 16 \
#     --output_dir "$TESTS_OUTPUT_DIR" \
#     --note "qwen3 8-b" \
#     --order "mlp,qk,vo" \
#     --max_sparsity 0.95 \
#     --ridge_vo 1e-5 \
#     --ridge_qk 1e-2 \
#     --sparsity_smoothing 0.0495 \
#     --nystrom_ridge 1e-4 \
#     --dataset alpaca

# lm_eval --model hf \
#     --model_args pretrained="$TESTS_OUTPUT_DIR/model,dtype=bfloat16,trust_remote_code=True" \
#     --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \
#     --batch_size 128 \
#     --output_path ./metrics/lm_eval/ \
#     --num_fewshot 0 \
#     --metadata '{"note": "Qwen3-8B--FINAL"}'