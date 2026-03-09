# train llama-7b with 40% params jeffwan's version
CUDA_VISIBLE_DEVICES=0 python df_svd.py \
  --model_id jeffwan/llama-7b-hf \
  --output_dir ./jeffwan_llama7b_dfsvd_0.4 \
  --compression_ratio 0.4 \
  --base_update_rank 8 \
  --seq_len 2048 \
  --calib_sequences 256 \
  --cov_max_tokens_total 524288 \
  --batch_size 1 \
  --whitening_cache_dir ./cache/dfsvd_whitening_llama7b_seq2048_n256_tok524288 \
  --whitening_cache_dtype float32 \
  --factor_dtype float32 \
  --dtype float16 \
  --do_train \
  --train_steps 200 \
  --train_lr 5e-4


# huggyllama version 
CUDA_VISIBLE_DEVICES=0 python df_svd.py \
  --model_id huggyllama/llama-7b \
  --output_dir ./huggyllama_llama7b_dfsvd_0.4 \
  --compression_ratio 0.4 \
  --base_update_rank 8 \
  --seq_len 2048 \
  --calib_sequences 256 \
  --cov_max_tokens_total 524288 \
  --batch_size 1 \
  --whitening_cache_dir ./cache/dfsvd_whitening_llama7b_seq2048_n256_tok524288 \
  --whitening_cache_dtype float32 \
  --factor_dtype float32 \
  --dtype float16 \
  --do_train \
  --train_steps 200 \
  --train_lr 5e-4


# # train llama-2-7b with 40% params
# CUDA_VISIBLE_DEVICES=0 python df_svd.py \
#   --model_id meta-llama/Llama-2-7b-hf \
#   --output_dir ./llama2_dfsvd_debug \
#   --compression_ratio 0.4 \
#   --base_update_rank 8 \
#   --seq_len 2048 \
#   --calib_sequences 256 \
#   --cov_max_tokens_total 524288 \
#   --batch_size 1 \
#   --whitening_cache_dir ./cache/dfsvd_whitening_llama2_7b_seq2048_n256_tok524288 \
#   --whitening_cache_dtype float32 \
#   --factor_dtype float32 \
#   --dtype float16 \
#   --do_train \
#   --train_steps 200 \
#   --train_lr 5e-4
