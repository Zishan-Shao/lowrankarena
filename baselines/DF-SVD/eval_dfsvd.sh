CUDA_VISIBLE_DEVICES=0 python -m eval_results.eval_benchmarks \
  --model ./baselines/DF-SVD/llama7b_dfsvd_0.4 \
  --dfsvd_base_model huggyllama/llama-7b \
  --device cuda --batch_size 8 --dtype bfloat16 \
  --use_lm_eval --lm_eval_tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa \
    --output_json outputs/dfsvd_benchmarks_llama.json


# ppl
CUDA_VISIBLE_DEVICES=0 python -m eval_results.eval_general_ppl \
  --dfsvd_model ./baselines/DF-SVD/llama7b_dfsvd_0.4 \
  --dfsvd_base_model huggyllama/llama-7b \
  --datasets wikitext2,ptb,c4 \
  --device cuda --seqlen 2048 --batch_size 4 --dtype bfloat16 \
    --metrics token \
    --output_json outputs/dfsvd_ppl_llama.json
# lingsuistic eval
CUDA_VISIBLE_DEVICES=0 python -m eval_results.eval_linguistic_tasks \
  --dfsvd_model ./baselines/DF-SVD/llama7b_dfsvd_0.4 \
  --dfsvd_base_model huggyllama/llama-7b \
  --tasks blimp \
  --device cuda --batch_size 8 --dtype bfloat16 \
    --output_json outputs/dfsvd_linguistic_llama.json


# llama 2-7b with 40% params
#common sense
# CUDA_VISIBLE_DEVICES=0 python -m eval_results.eval_benchmarks \
#   --model ./baselines/DF-SVD/llama2_dfsvd_debug \
#   --dfsvd_base_model meta-llama/Llama-2-7b-hf \
#   --device cuda --batch_size 8 --dtype bfloat16 \
#   --use_lm_eval --lm_eval_tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa \
#     --output_json outputs/dfsvd_benchmarks.json


# # ppl
# CUDA_VISIBLE_DEVICES=0 python -m eval_results.eval_general_ppl \
#   --dfsvd_model ./baselines/DF-SVD/llama2_dfsvd_debug \
#   --dfsvd_base_model meta-llama/Llama-2-7b-hf \
#   --datasets wikitext2,ptb,c4 \
#   --device cuda --seqlen 2048 --batch_size 4 --dtype bfloat16 \
#     --metrics token \
#     --output_json outputs/dfsvd_ppl.json
# # lingsuistic eval
# CUDA_VISIBLE_DEVICES=0 python -m eval_results.eval_linguistic_tasks \
#   --dfsvd_model ./baselines/DF-SVD/llama2_dfsvd_debug \
#   --dfsvd_base_model meta-llama/Llama-2-7b-hf \
#   --tasks blimp \
#   --device cuda --batch_size 8 --dtype bfloat16 \
#     --output_json outputs/dfsvd_linguistic.json