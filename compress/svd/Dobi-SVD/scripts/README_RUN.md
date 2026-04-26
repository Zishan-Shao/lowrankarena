# Run Guide (Dobi-SVD Matrix)

## 1) Activate environment
```bash
source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
```

## 2) Smoke on one model/ratio
```bash
python scripts/run_dobi_matrix.py \
  --repo-root /deac/csc/yangGrp/cuij/LLM/Dobi-SVD \
  --path-head-folder /deac/csc/yangGrp/cuij/LLM \
  --path-head-folder-output /deac/csc/yangGrp/cuij/LLM/Dobi-SVD/results \
  --models /deac/csc/yangGrp/cuij/LLM/models/hf_models/Qwen__Qwen3-8B-Base \
  --ratios 0.8 \
  --seq-len 512 \
  --n-train-epochs 1 \
  --n-train-samples 8 \
  --n-eval-samples 2 \
  --gpu-tries 3 \
  --max-gamma 128
```

## 3) Full matrix (3 models x 5 ratios)
```bash
python scripts/run_dobi_matrix.py \
  --repo-root /deac/csc/yangGrp/cuij/LLM/Dobi-SVD \
  --path-head-folder /deac/csc/yangGrp/cuij/LLM \
  --path-head-folder-output /deac/csc/yangGrp/cuij/LLM/Dobi-SVD/results \
  --models \
    /deac/csc/yangGrp/cuij/LLM/models/hf_models/huggyllama__llama-7b \
    /deac/csc/yangGrp/cuij/LLM/models/hf_models/Qwen__Qwen3-8B-Base \
    /deac/csc/yangGrp/cuij/LLM/models/hf_models/Qwen__Qwen3-8B \
  --ratios 0.4 0.5 0.6 0.7 0.8 \
  --seq-len 2048 \
  --n-train-epochs 20 \
  --n-train-samples 256 \
  --n-eval-samples 256 \
  --gpu-tries 0 0,1 0,1,2 0,1,2,3
```

## Output
- Logs and metrics: `logs/matrix_<timestamp>/`
- Summary CSV: `logs/matrix_<timestamp>/summary.csv`
- Per-stage peak memory is in metrics JSON and summary CSV.
