CUDA_VISIBLE_DEVICES=1 \
python baselines/SVD-LLM/SVDLLM.py \
  --model huggyllama/llama-7b \
  --step 1 \
  --ratio 0.6 \
  --dataset wikitext2 \
  --whitening_nsamples 256 \
  --model_seq_len 2048 \
  --save_path checkpoints \