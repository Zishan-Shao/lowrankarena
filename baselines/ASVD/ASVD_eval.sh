# codes to run asvd but wo saving the model just checking runnability
CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="meta-llama/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.4 --use_cache

# code to actually build the asvd model and save to HF repo 40% params
CUDA_VISIBLE_DEVICES=0 python huggingface_repos/build_asvd_repo.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --act_aware \
  --alpha 0.5 \
  --calib_dataset wikitext2 \
  --scaling_method abs_mean \
  --sensitivity_metric ppl \
  --param_ratio_target 0.4 \
  --use_cache

# code to run all tasks on eval_benchmarks 
# 40% params 
export ASVD_MODEL="baselines/ASVD/huggingface_repos/Llama-2-7b-hf-asvd40"
CUDA_VISIBLE_DEVICES=0 python eval_results/eval_benchmarks.py   --model "$ASVD_MODEL" --device cuda --batch_size 1 --dtype bfloat16 --limit 50

# or just use lm_eval
export ASVD_MODEL="/home/lz299/lowrankarena/baselines/ASVD/huggingface_repos/Llama-2-7b-hf-asvd40"
CUDA_VISIBLE_DEVICES=0 python /home/lz299/lowrankarena/baselines/ASVD/eval_ASVD_benchmark.py \
  --model "$ASVD_MODEL" \
  --device cuda \
  --batch_size 1 \
  --dtype bfloat16 \
  --use_lm_eval \
  --limit 50 \
  --lm_eval_tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa\



# run linguistic tasks 
CUDA_VISIBLE_DEVICES=0 python eval_ASVD_linguistic_tasks.py \
  --model ./huggingface_repos/Llama-2-7b-hf-asvd90 \
  --trust_remote_code \
  --tasks blimp,cola \
  --batch_size 1 \
  --device cuda \
  --dtype fp16



