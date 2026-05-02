# Serving Runbook

This is the quick operator guide for running LowRankArena serving checks on the
Codex branch. It covers pruning checkpoints, MoDeGPT serving, and the ASVD /
SVD-LLM / DoBi / Basis Sharing serving summary jobs.

## Setup

Start from the branch that contains the vLLM wrapper fixes:

```bash
cd /home/shaoz/lowrankarena
git fetch origin
git checkout codex/standardize-benchmark-flow-vllm-eval
source /home/shaoz/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
```

For Llama checkpoints, make sure the machine has Hugging Face access to the
gated Meta repos before launching a job.

## What To Run

Use these index files:

- Qwen3 dense, ASVD, SVD-LLM v1/v2, DoBi, Basis Sharing, and MoDeGPT:
  `benchmark/speed/serving_summary_qwen_index.csv`
- Llama-3.1 dense, ASVD, SVD-LLM v1/v2, DoBi, Basis Sharing, and MoDeGPT:
  `benchmark/speed/serving_summary_llama31_index.csv`
- SliceGPT pruning checkpoints:
  `benchmark/speed/pruning_index.csv`

Use these suites:

- Online OpenAI-compatible server/client serving:
  `speed/serve_e2e`
- Offline vLLM pruning serving:
  `speed/pruning_serve`

`status.tsv` is the first file to check. A row with status `0` means that the
checkpoint loaded, served, and completed the benchmark request set.

## Qwen Serving Summary

Smoke the full Qwen set before a long run:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT="/home/shaoz/lowrankarena/results/leaderboard/serving_summary/qwen_tiny_smoke_${TS}"
sbatch --parsable \
  --partition=gpu \
  --constraint=a100_40 \
  --time=06:00:00 \
  --job-name=qwen_smoke \
  --export=ALL,SERVING_NUM_PROMPTS=1,SERVING_MAX_CONCURRENCY=1,SERVING_REQUEST_RATE=1.0,SERVING_READY_TIMEOUT=240,SERVING_PROMPT_LENGTH=16,SERVING_GENERATION_LENGTH=4,SERVING_MAX_MODEL_LEN=128,SERVING_GPU_MEMORY_UTILIZATION=0.75,SERVING_ENFORCE_EAGER=1,SERVING_OUTPUT_DIR="$OUT",SERVING_RUN_LABEL=qwen_tiny_smoke,SERVING_PORT=8012 \
  scripts/slurm/serving_summary_qwen.sbatch
```

Run the paper-facing Qwen serving summary:

```bash
sbatch --parsable scripts/slurm/serving_summary_qwen.sbatch
```

Run only MoDeGPT rows:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT="/home/shaoz/lowrankarena/results/leaderboard/serving_summary/qwen_modegpt_${TS}"
sbatch --parsable \
  --partition=gpu \
  --constraint=a100_40 \
  --time=04:00:00 \
  --job-name=qwen_modegpt \
  --export=ALL,SERVING_CHECKPOINTS="qwen3-8b-modegpt-0.6-serving qwen3-8b-base-modegpt-0.6-serving",SERVING_OUTPUT_DIR="$OUT",SERVING_PORT=8013,SERVING_ENFORCE_EAGER=1 \
  scripts/slurm/serving_summary_qwen.sbatch
```

## Llama-3.1 SVD And MoDeGPT Serving

Smoke the Llama-3.1 serving summary:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT="/home/shaoz/lowrankarena/results/leaderboard/serving_summary/llama31_tiny_smoke_${TS}"
sbatch --parsable \
  --partition=gpu \
  --constraint=a100_40 \
  --time=06:00:00 \
  --job-name=l31_smoke \
  --export=ALL,SERVING_NUM_PROMPTS=1,SERVING_MAX_CONCURRENCY=1,SERVING_REQUEST_RATE=1.0,SERVING_READY_TIMEOUT=240,SERVING_PROMPT_LENGTH=16,SERVING_GENERATION_LENGTH=4,SERVING_MAX_MODEL_LEN=128,SERVING_GPU_MEMORY_UTILIZATION=0.75,SERVING_ENFORCE_EAGER=1,SERVING_OUTPUT_DIR="$OUT",SERVING_RUN_LABEL=llama31_tiny_smoke,SERVING_PORT=8014 \
  scripts/slurm/serving_summary_llama31.sbatch
```

Run the full Llama-3.1 ASVD / SVD-LLM / DoBi / Basis / MoDeGPT summary:

```bash
sbatch --parsable scripts/slurm/serving_summary_llama31.sbatch
```

Run one checkpoint directly without the Slurm wrapper:

```bash
python scripts/run_speed.py \
  llama31-8b-svdllm-v2-0.6-serving \
  --index benchmark/speed/serving_summary_llama31_index.csv \
  --suite speed/serve_e2e \
  --output-dir results/leaderboard/serving_summary/manual_l31_svdllm_v2 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --request-rate 1.0 \
  --prompt-length 16 \
  --generation-length 4 \
  --max-model-len 128 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --strict-validation
```

## Pruning Serving

The pruning path uses `speed/pruning_serve`, which is an offline vLLM generate
suite. It is still the serving runner, but it does not start the OpenAI server.

Smoke one pruning checkpoint:

```bash
python scripts/run_speed.py \
  llama31-8b-slicegpt-prune-only-0.6 \
  --index benchmark/speed/pruning_index.csv \
  --suite speed/pruning_serve \
  --output-dir results/leaderboard/speed/pruning_smoke \
  --batch-size 1 \
  --prompt-length 128 \
  --generation-length 16 \
  --repeat 1 \
  --warmup 0 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.5 \
  --strict-validation
```

Run the default pruning suite for one checkpoint:

```bash
python scripts/run_speed.py \
  llama31-8b-slicegpt-prune-only-0.6 \
  --index benchmark/speed/pruning_index.csv \
  --suite speed/pruning_serve \
  --output-dir results/leaderboard/speed/pruning_serve
```

Submit the same smoke through Slurm:

```bash
sbatch --parsable \
  --partition=gpu \
  --constraint=a100_40 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=02:00:00 \
  --job-name=prune_smoke \
  --wrap='cd /home/shaoz/lowrankarena && source /home/shaoz/miniconda3/etc/profile.d/conda.sh && conda activate lowrankarena && python scripts/run_speed.py llama31-8b-slicegpt-prune-only-0.6 --index benchmark/speed/pruning_index.csv --suite speed/pruning_serve --output-dir results/leaderboard/speed/pruning_smoke --batch-size 1 --prompt-length 128 --generation-length 16 --repeat 1 --warmup 0 --max-model-len 512 --gpu-memory-utilization 0.5 --strict-validation'
```

## Outputs And Debugging

For Slurm summary jobs, inspect:

- `$SERVING_OUTPUT_DIR/status.tsv`
- `$SERVING_OUTPUT_DIR/serving_summary.tsv`
- `$SERVING_OUTPUT_DIR/logs/<checkpoint>.log`
- `$SERVING_OUTPUT_DIR/bench_serve_artifacts/speed_serve_e2e/<checkpoint>/server.stderr.log`

Useful checks:

```bash
squeue -j <jobid> -o '%i %j %T %M %R'
sacct -j <jobid> --format=JobID,JobName%24,State,ExitCode,Elapsed,NodeList -P
cat <output-dir>/status.tsv
```

If a server fails before any requests are sent, read `server.stderr.log` first.
If the client starts and then fails, read the matching profile stderr under the
same `bench_serve_artifacts` directory.

## Notes

- MoDeGPT runs through the Transformers backend wrapper. Qwen MoDeGPT also uses
  the eager-attention fallback to avoid compressed-head attention shape issues
  in native vLLM kernels.
- ASVD, SVD-LLM, DoBi, and Basis Sharing are served through the vLLM adapter
  wrappers in `src/vllm/vllm_adapter.py`.
- Use a unique `SERVING_PORT` for each concurrent online serving job.
- Tiny smoke numbers are only for load/request validation. Use the full defaults
  for reporting throughput and latency.
