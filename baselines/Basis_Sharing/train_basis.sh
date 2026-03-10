python - <<'PY'
from pathlib import Path
import yaml

src = Path("tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml")
dst = Path("tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_60_train.yaml")

cfg = yaml.safe_load(src.read_text())

cfg["model_args"]["compression_ratio"] = 60
cfg["calibration_args"]["dataset_name"] = "wikitext"

cfg["model_saving"]["updated_model_path"] = "./updated_model/share_llama-7b_60/wikitext"
cfg["model_saving"]["untrained_model_path"] = "./untrained_model/share_llama-7b_60/wikitext"
cfg["model_saving"]["trained_model_path"] = "./trained_model/share_llama-7b_60/wikitext"

if "lora_args" in cfg:
    cfg["lora_args"]["lora_output_dir"] = "./lora/share_llama-7b_60/wikitext"
    cfg["lora_args"]["lora_run_name"] = "share_llama-7b_60"

dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote {dst}")
PY

CUDA_VISIBLE_DEVICES=2 python train.py \
  --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_60_train.yaml \
  --dataset_name wikitext