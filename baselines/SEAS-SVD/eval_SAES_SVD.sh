# llama1
CUDA_VISIBLE_DEVICES=6 python -m eval_results.eval_benchmarks \
  --model baselines/SEAS-SVD/robust/jeffwan_llama7b_saes_r0.4 \
  --saes_svd \
  --saes_base_model jeffwan/llama-7b-hf \
  --use_lm_eval \
  --lm_eval_tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa \
  --device cuda \
  --batch_size 1 \
  --dtype bfloat16 \
  --lm_eval_num_fewshot 0 \
  --output_json outputs/saes_benchmark_jeffwan.json
# ppl

CUDA_VISIBLE_DEVICES=6 python -m eval_results.eval_general_ppl \
  --saes_model baselines/SEAS-SVD/robust/jeffwan_llama7b_saes_r0.4 \
  --saes_base_model jeffwan/llama-7b-hf \
  --datasets wikitext2,ptb,c4 \
  --c4_stream --c4_docs 2000 \
  --seqlen 2048 \
  --batch_size 1 \
  --device cuda \
  --dtype bfloat16 \
  --metrics token \
  --output_json outputs/saes_ppl_jeffwan.json
# linguistic tasks
CUDA_VISIBLE_DEVICES=6 python -m eval_results.eval_linguistic_tasks \
  --saes_model baselines/SEAS-SVD/robust/jeffwan_llama7b_saes_r0.4 \
  --saes_base_model jeffwan/llama-7b-hf \
  --tasks blimp \
  --num_fewshot 0 \
  --batch_size 1 \
  --device cuda \
  --dtype bfloat16 \
  --output_json outputs/saes_linguistic_jeffwan.json


# code to eval SAES-SVD
# benchmark
# CUDA_VISIBLE_DEVICES=3 python -m eval_results.eval_benchmarks \
#   --model baselines/SEAS-SVD/robust/llama2_saes_r0.4 \
#   --saes_svd \
#   --saes_base_model meta-llama/Llama-2-7b-hf \
#   --use_lm_eval \
#   --lm_eval_tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa \
#   --device cuda \
#   --batch_size 1 \
#   --dtype bfloat16 \
#   --lm_eval_num_fewshot 0 \
#   --output_json outputs/saes_benchmark.json
# # ppl

# CUDA_VISIBLE_DEVICES=3 python -m eval_results.eval_general_ppl \
#   --saes_model baselines/SEAS-SVD/robust/llama2_saes_r0.4 \
#   --saes_base_model meta-llama/Llama-2-7b-hf \
#   --datasets wikitext2,ptb,c4 \
#   --c4_stream --c4_docs 2000 \
#   --seqlen 2048 \
#   --batch_size 1 \
#   --device cuda \
#   --dtype bfloat16 \
#   --metrics token \
#   --output_json outputs/saes_ppl.json
# # linguistic tasks
# CUDA_VISIBLE_DEVICES=3 python -m eval_results.eval_linguistic_tasks \
#   --saes_model baselines/SEAS-SVD/robust/llama2_saes_r0.4 \
#   --saes_base_model meta-llama/Llama-2-7b-hf \
#   --tasks blimp \
#   --num_fewshot 0 \
#   --batch_size 1 \
#   --device cuda \
#   --dtype bfloat16 \
#   --output_json outputs/saes_linguistic.json
