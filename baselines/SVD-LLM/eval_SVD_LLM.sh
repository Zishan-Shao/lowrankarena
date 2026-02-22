# run SVD-LLM (whitening + SVD). 40% params
#mkdir -p checkpoints

# CUDA_VISIBLE_DEVICES=1 \
# python baselines/SVD-LLM/SVDLLM.py \
#   --model meta-llama/Llama-2-7b-hf \
#   --step 1 \
#   --ratio 0.6 \
#   --dataset wikitext2 \
#   --whitening_nsamples 256 \
#   --model_seq_len 2048 \
#   --save_path checkpoints \





CUDA_VISIBLE_DEVICES=1 python -m baselines.SVD-LLM.eval_SVDLLM_benchmark \
  --model checkpoints/meta_llama_Llama_2_7b_hf_whitening_only_0.4.pt \
  --device cuda \
  --batch_size 1 \
  --dtype bfloat16 \
  --use_lm_eval \
  --lm_eval_tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa \
  --lm_eval_num_fewshot 0 \
  --lm_eval_max_length 2048 \
  --output_json outputs/benchmark.json

# eval ppl on wikitext2, ptb, c4
CUDA_VISIBLE_DEVICES=1 python -m baselines.SVD-LLM.eval_SVDLLM_ppl \
  --checkpoint checkpoints/meta_llama_Llama_2_7b_hf_whitening_only_0.4.pt \
  --datasets wikitext2,ptb,c4 \
  --max_batches 50 \
  --device cuda \
  --seqlen 2048 \
  --batch_size 1 \
  --dtype bfloat16 \
  --output_json outputs/ppl.json

# eval linguistic tasks
CUDA_VISIBLE_DEVICES=1 python -m baselines.SVD-LLM.eval_SVDLLM_linguistic \
  --checkpoint checkpoints/meta_llama_Llama_2_7b_hf_whitening_only_0.4.pt \
  --device cuda \
  --batch_size 1 \
  --dtype bfloat16 \
  --tasks blimp \
  --output_json outputs/linguistic.json


