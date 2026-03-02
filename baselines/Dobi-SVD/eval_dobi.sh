# these are for llama-1
# commonsense eval with lm_eval
CUDA_VISIBLE_DEVICES=6 python -m  eval_results.eval_benchmarks \
  --model Qinsi1/DobiSVD-Llama-7b-hf-0.4 \
  --device cuda \
  --batch_size 1 \
  --use_lm_eval \
  --dtype bfloat16 \
  --lm_eval_tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa \
  --output_json outputs/dobi_benchmarks_llama.json

# ppl eval 
CUDA_VISIBLE_DEVICES=6 python -m eval_results.eval_general_ppl \
  --dobi_model Qinsi1/DobiSVD-Llama-7b-hf-0.4 \
  --datasets wikitext2,ptb,c4 \
  --metrics token \
  --device cuda --dtype bfloat16 \
  --output_json outputs/dobi_ppl_llama.json

# linguistic eval 
CUDA_VISIBLE_DEVICES=6 python  -m eval_results.eval_linguistic_tasks \
  --dobi_model Qinsi1/DobiSVD-Llama-7b-hf-0.4 \
  --tasks blimp \
  --device cuda --dtype bf16 \
  --output_json outputs/dobi_linguistic_llama.json



# # llama 2
# CUDA_VISIBLE_DEVICES=6 python -m  eval_results.eval_benchmarks \
#   --model Qinsi1/DobiSVD-Llama-2-7b-hf-0.4 \
#   --device cuda \
#   --batch_size 1 \
#   --use_lm_eval \
#   --dtype bfloat16 \
#   --lm_eval_tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa \
#   --output_json outputs/dobi_benchmarks.json

# # ppl eval 
# CUDA_VISIBLE_DEVICES=6 python -m eval_results.eval_general_ppl \
#   --dobi_model Qinsi1/DobiSVD-Llama-2-7b-hf-0.4 \
#   --datasets wikitext2,ptb,c4 \
#   --metrics token \
#   --device cuda --dtype bfloat16 \
#   --output_json outputs/dobi_ppl.json

# # linguistic eval 
# CUDA_VISIBLE_DEVICES=6 python  -m eval_results.eval_linguistic_tasks \
#   --dobi_model Qinsi1/DobiSVD-Llama-2-7b-hf-0.4 \
#   --tasks blimp \
#   --device cuda --dtype bf16 \
#   --output_json outputs/dobi_linguistic.json
