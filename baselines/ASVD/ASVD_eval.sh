



# llama-7b-asvd40
ASVD_MODEL="/home/lz299/lowrankarena/baselines/ASVD/huggingface_repos/llama-7b-asvd40"
RUN_ID="$(basename "$ASVD_MODEL")_$(date +%Y%m%d_%H%M%S)"
# benchmark (lm-eval)
CUDA_VISIBLE_DEVICES=1 python -m eval_results.eval_benchmarks \
  --model baselines/ASVD/huggingface_repos/llama-7b-asvd40 \
  --use_lm_eval \
  --lm_eval_tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa,truthfulqa_mc1 \
  --lm_eval_num_fewshot 0 \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 1

# ppl
CUDA_VISIBLE_DEVICES=0 python -m baselines.ASVD.eval_ASVD_ppl_with_json \
  --checkpoint "$ASVD_MODEL" \
  --datasets wikitext2,ptb,c4 \
  --device cuda \
  --seqlen 2048 \
  --batch_size 1 \
  --dtype bfloat16 \
  --output_dir "outputs" \
  --run_name "$RUN_ID" 

# linguistic
CUDA_VISIBLE_DEVICES=0 python -m baselines.ASVD.eval_ASVD_linguistic_tasks_with_json \
  --model "$ASVD_MODEL" \
  --trust_remote_code \
  --tasks blimp \
  --batch_size 1 \
  --device cuda \
  --dtype fp16 \
  --output_dir "outputs" \
  --run_name "$RUN_ID"




