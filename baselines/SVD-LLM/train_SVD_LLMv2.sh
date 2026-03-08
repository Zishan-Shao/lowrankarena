# physical GPU 2 becomes cuda:0 inside the process
CUDA_VISIBLE_DEVICES=2 python -u SVDLLM_v2.py \
  --model jeffwan/llama-7b-hf \
  --ratio 0.6 \
  --dataset wikitext2 \
  --nsamples 256 --seq_len 2048 --batch_size 1 \
  --device cuda:0 \
  --load_dtype float16 \
  --stats_dtype float32 \
  --sqrt_dtype float32 --store_act_dtype float16 --sqrt_on_gpu \
  --save_path ./checkpoints/llama2_svdllmv2_red0.6.pt \
  --timing_file svdllmv2_timing.json \
  --tqdm auto

# CUDA_VISIBLE_DEVICES=1 python SVDLLM_v2.py \
#   --model meta-llama/Llama-2-7b-hf \
#   --step 1 \
#   --ratio 0.6 \
#   --whitening_nsamples 256 \
#   --dataset wikitext2 \
#   --seed 3 \
#   --model_seq_len 2048 \
#   --save_path ./checkpoints