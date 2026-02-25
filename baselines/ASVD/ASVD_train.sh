# code to actually build the asvd model and save to HF repo 40% params 
CUDA_VISIBLE_DEVICES=5 python huggingface_repos/build_asvd_repo.py \
  --model_id huggyllama/llama-7b \
  --act_aware \
  --alpha 0.5 \
  --calib_dataset wikitext2 \
  --scaling_method abs_mean \
  --sensitivity_metric ppl \
  --param_ratio_target 0.4 \
  --use_cache