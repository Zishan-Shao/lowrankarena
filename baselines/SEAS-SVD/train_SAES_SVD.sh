


# train with llama-7b with 40% params  jeffwan's version
CUDA_VISIBLE_DEVICES=2 python saes_svd.py \
  --model_id jeffwan/llama-7b-hf \
  --output_dir jeffwan_llama7b_saes_r0.4 \
  --compression_ratio 0.4 \
  --seq_len 2048 \
  --calib_sequences 128 \
  --batch_size 1 \
  --max_tokens_total 262144 \
  --beta_mode aces \
  --device cuda \
  --teacher_device cuda \
  --dtype bfloat16 \
  --teacher_dtype bfloat16 \
  --factor_dtype bfloat16

# train with llama-7b with 40% params huggyllama version
CUDA_VISIBLE_DEVICES=2 python saes_svd.py \
  --model_id huggyllama/llama-7b \
  --output_dir huggyllama_llama7b_saes_r0.4 \
  --compression_ratio 0.4 \
  --seq_len 2048 \
  --calib_sequences 128 --batch_size 1 \
  --max_tokens_total 262144 \
  --beta_mode aces \
  --device cuda \
  --teacher_device cuda \
  --dtype bfloat16 \
  --teacher_dtype bfloat16 \
  --factor_dtype bfloat16


# # how to run 40% params with llama-2-7b
# CUDA_VISIBLE_DEVICES=3 python saes_svd.py \
#   --model_id meta-llama/Llama-2-7b-hf \
#   --output_dir robust/llama2_saes_r0.4 \
#   --compression_ratio 0.4 \
#   --seq_len 2048 \
#   --calib_sequences 128 \
#   --batch_size 1 \
#   --max_tokens_total 262144 \
#   --beta_mode aces \
#   --device cuda \
#   --teacher_device cuda \
#   --dtype bfloat16 \
#   --teacher_dtype bfloat16 \
#   --factor_dtype bfloat16