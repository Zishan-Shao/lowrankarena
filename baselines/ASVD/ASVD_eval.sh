# code to run avd with 90% param ratio


CUDA_VISIBLE_DEVICES=0 python eval_results/eval_general_ppl.py --model "$ASVD_MODEL" --datasets wikitext2 --max_batches 10 --device cuda --dtype bfloat16
CUDA_VISIBLE_DEVICES=0 python eval_results/eval_benchmarks.py   --model "$ASVD_MODEL" --limit 20 --skip_gsm8k --skip_truthfulqa --device cuda --batch_size 4 --dtype bfloat16
CUDA_VISIBLE_DEVICES=0 python eval_results/eval_linguistic_tasks.py --model "$ASVD_MODEL" --tasks blimp,cola --limit 20 --device cuda --batch_size 4 --dtype bfloat16
CUDA_VISIBLE_DEVICES=0 python eval_results/eval_template_stability.py --model "$ASVD_MODEL" --tasks arc_easy --templates plain,qa --limit 20 --device cuda --batch_size 4 --dtype bfloat16


# codes to run asvd test
CUDA_VISIBLE_DEVICES='2' python asvd.py --model_id="meta-llama/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.4 --use_cache

# code to put asvd into HF repo
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
CUDA_VISIBLE_DEVICES=0 python eval_results/eval_benchmarks.py   --model "$ASVD_MODEL" --limit 20 --device cuda --batch_size 4 --dtype bfloat16



