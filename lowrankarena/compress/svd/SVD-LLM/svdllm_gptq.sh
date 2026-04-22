#!/bin/bash

# run data whitening with keep_ratio=0.8 (20% parameter reduction)
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.8 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .

# further compress the model with GPTQ-4bit
python quant_llama.py --model_path whitening/jeffwan_llama_7b_hf_whitening_0.8.pt --dataset c4 --wbits 4 --true-sequential --act-order --new-eval  --save svdllm_gptq_4.pt
