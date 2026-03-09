
# llama7b with no remapping
CUDA_VISIBLE_DEVICES=6 python svd_trainer.py \
  --model_id jeffwan/llama-7b-hf \
  --target_ratio 0.4 \
  --seq_len 2048 \
  --seed 0 \
  --training_dataset wikitext2 \
  --n_train_epochs 20 \
  --n_train_samples 256 \
  --profile_train \
  --profile_log_every 1 \
  --profile_warmup_steps 2