# code to actually build the asvd model and save to HF repo 40% params 
CUDA_VISIBLE_DEVICES=0 python huggingface_repos/build_asvd_repo.py \
  --model_id jeffwan/llama-7b-hf \
  --act_aware \
  --alpha 0.5 \
  --calib_dataset wikitext2 \
  --scaling_method abs_mean \
  --sensitivity_metric ppl \
  --param_ratio_target 0.4



# llama2-7b 
CUDA_VISIBLE_DEVICES=0 python huggingface_repos/build_asvd_repo.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --act_aware \
  --alpha 0.5 \
  --calib_dataset wikitext2 \
  --scaling_method abs_mean \
  --sensitivity_metric ppl \
  --param_ratio_target 0.4