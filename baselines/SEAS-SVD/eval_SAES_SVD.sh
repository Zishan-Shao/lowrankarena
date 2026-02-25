# # how to run 40% params
# CUDA_VISIBLE_DEVICES=3 python -m baselines.SEAS-SVD.saes_svd \
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

# code to eval SAES-SVD
# benchmark
CUDA_VISIBLE_DEVICES=3 python -m eval_results.eval_benchmarks \
  --model baselines/SEAS-SVD/robust/llama2_saes_r0.4 \
  --saes_svd \
  --saes_base_model meta-llama/Llama-2-7b-hf \
  --use_lm_eval \
  --lm_eval_tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa \
  --device cuda \
  --batch_size 1 \
  --dtype bfloat16 \
  --lm_eval_num_fewshot 0 \
  --output_json outputs/saes_benchmark.json
# ppl

CUDA_VISIBLE_DEVICES=3 python -m eval_results.eval_general_ppl \
  --saes_model baselines/SEAS-SVD/robust/llama2_saes_r0.4 \
  --saes_base_model meta-llama/Llama-2-7b-hf \
  --datasets wikitext2,ptb,c4 \
  --c4_stream --c4_docs 2000 \
  --seqlen 2048 \
  --batch_size 1 \
  --device cuda \
  --dtype bfloat16 \
  --metrics token \
  --output_json outputs/saes_ppl.json
# linguistic tasks
CUDA_VISIBLE_DEVICES=3 python -m eval_results.eval_linguistic_tasks \
  --saes_model baselines/SEAS-SVD/robust/llama2_saes_r0.4 \
  --saes_base_model meta-llama/Llama-2-7b-hf \
  --tasks blimp \
  --num_fewshot 0 \
  --batch_size 1 \
  --device cuda \
  --dtype bfloat16 \
  --output_json outputs/saes_linguistic.json
