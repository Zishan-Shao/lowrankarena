# llama-7b
CUDA_VISIBLE_DEVICES=6 \
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.6 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path ./checkpoints

# llama2-7b
CUDA_VISIBLE_DEVICES=6 \
python SVDLLM.py --model meta-llama/Llama-2-7b-hf --step 1 --ratio 0.4 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path ./checkpoints

