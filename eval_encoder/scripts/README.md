# eval_encoder/scripts — 实验入口说明

## Canonical Entrypoints（论文四大实验）

| 脚本 | 实验 | 输出 |
|------|------|------|
| `expA.sh` | 质量实验（压缩精度）| `glue_results/*.json`, `eval_results/expA.csv` |
| `expB.sh` | 后端性能微基准 | `eval_results/expB.csv` |
| `expC.sh` | Scaling（seq_len / batch_size）| `eval_results/expC_seqlen.csv`, `eval_results/expC_batch.csv` |
| `expD.sh` | Kernel 分析（nsys）| `eval_results/expD.csv`, figures |

### Canonical config（所有实验共用）

```
QKV_MODE  = per_head
RANK_ATTN = 48   RANK_FFN = 256   RANK_WO = 208
BUDGET    = 0.527          # AdaSVD，参数量与上面 rank 等价
DTYPE     = bf16
SEQ_LEN   = 512   BATCH_SIZE = 32
```

---

## expA.sh — 质量实验

```bash
cd lowrankarena/

# 全量（GLUE 8 tasks + SuperGLUE/HANS/ANLI 7 tasks）
bash eval_encoder/scripts/expA.sh

# 只跑 GLUE
PHASES=glue bash eval_encoder/scripts/expA.sh

# 只跑 SuperGLUE/HANS/ANLI
PHASES=superglue bash eval_encoder/scripts/expA.sh

# 方法子集，只跑 stage1（不微调，看原始压缩精度）
PHASES=glue METHODS="svd fwsvd" TWO_STAGE=false \
  bash eval_encoder/scripts/expA.sh
```

**依赖**：`compare_all_methods.sh` + `run_superglue_benchmark.sh`（expA 是薄包装，固定 canonical config）

---

## expB.sh — 后端性能微基准

```bash
cd lowrankarena/

# 全量（8 GLUE tasks × 4 methods × 4 backends）
bash eval_encoder/scripts/expB.sh

# 只跑两个代表任务，验证 flashsvd 收益
TASKS="mnli stsb" bash eval_encoder/scripts/expB.sh

# 指定 backend 子集
BACKENDS="naive flashsvd15" bash eval_encoder/scripts/expB.sh
```

**前提**：`eval_encoder/models/{task}/{method}_ra48_rf256_rw208_per_head_naive/` 已存在

> ⚠️ **expB 只消费 expA 产物，不负责生成 checkpoint。**
> 缺少 checkpoint 时该组合会打印 `[skip]` 并继续，不会自动压缩。
> 如果 skip 大量出现，先跑 `expA.sh` 补全 checkpoint。

**输出**：`eval_encoder/eval_results/expB.csv`（canonical 路径，固定不变）

> ⚠️ **三张图（plot_backend_sweep / plot_combined_figure / plot_flops_breakdown）默认都读这个路径。**
> 不要用 `--out_csv` 把数据写到别处，否则画图时需要手动传参。
> 列集合固定（16 列），若 `analyze_compute.py` 更新了输出列，需先删除旧 CSV 再重新生成。

（含 latency_ms / throughput_sps / peak_mem_mb / FLOPs / MFU / arithmetic_intensity）

**画图**：

```bash
# 后端延迟/吞吐/显存对比
python eval_encoder/scripts/plot_backend_sweep.py \
  --tasks mnli stsb --dtype bf16 --seq_len 512

# FLOPs breakdown
python eval_encoder/scripts/plot_flops_breakdown.py \
  --task mnli --dtype bf16 --seq_len 512

# 综合大图（4 metrics × 2 tasks）
python eval_encoder/scripts/plot_combined_figure.py \
  --tasks mnli stsb --dtype bf16 --seq_len 512
```

---

## expC.sh — Scaling 实验

```bash
cd lowrankarena/

# 全量（seq_len sweep + batch_size sweep）
bash eval_encoder/scripts/expC.sh

# 只跑 seq_len sweep
PHASES=seqlen bash eval_encoder/scripts/expC.sh

# 只跑 batch_size sweep
PHASES=batch bash eval_encoder/scripts/expC.sh

# 自定义扫描点
SEQ_LENS="128 256 512" METHODS="svd" bash eval_encoder/scripts/expC.sh
```

**前提**：同 expB，checkpoint 由 expA 生成；缺 checkpoint 时 [skip]

**扫描参数**：

| Phase | 固定 | 扫描 |
|-------|------|------|
| seqlen | batch=32 | seq_len ∈ {128, 256, 384, 512} |
| batch  | seq_len=512 | batch ∈ {8, 16, 32, 64} |

> ⚠️ **seq_len=768 不支持**：BERT-base max_position_embeddings=512，超出报错。

**输出**：
- `eval_encoder/eval_results/expC_seqlen.csv`
- `eval_encoder/eval_results/expC_batch.csv`

---

## expD.sh — Kernel 分析（nsys）

```bash
cd lowrankarena/

# 全量（4 个代表点：svd/adasvd × naive/flashsvd15）
bash eval_encoder/scripts/expD.sh

# 只跑特定 task（默认 mnli）
TASK=mnli bash eval_encoder/scripts/expD.sh
```

**前提**：

1. `nsys` 已安装（`which nsys` 有输出；随 CUDA toolkit 附带）
2. expA 已生成 mnli 的 svd + adasvd checkpoint

**4 个 profiling 点**（与 `plot_nsys_kernel.py` 的 `POINT_META` 一致）：

| Tag | Method | Backend |
|-----|--------|---------|
| `mnli_svd_naive` | SVD | naive |
| `mnli_svd_flashsvd15` | SVD | flashsvd15 |
| `mnli_adasvd_naive` | AdaSVD | naive |
| `mnli_adasvd_flashsvd15` | AdaSVD | flashsvd15 |

**输出**：
- `eval_encoder/eval_results/nsys/*.nsys-rep` — 原始 profile 文件
- `eval_encoder/eval_results/nsys/nsys_summary.txt` — cuda_gpu_kern_sum 汇总
- `eval_encoder/eval_results/expD.csv` — 结构化指标
- `eval_encoder/eval_results/figures/nsys_kernel_analysis_mnli_bf16_seq512.png`

**手动调 parse + plot（已有 .nsys-rep 时）**：

```bash
python eval_encoder/scripts/parse_nsys_summary.py \
  --input   eval_encoder/eval_results/nsys/nsys_summary.txt \
  --out_csv eval_encoder/eval_results/expD.csv

python eval_encoder/scripts/plot_nsys_kernel.py \
  --csv    eval_encoder/eval_results/expD.csv \
  --outdir eval_encoder/eval_results/figures
```

---

## 底层脚本（被 expA 调用，不直接运行）

| 脚本 | 职责 |
|------|------|
| `compare_all_methods.sh` | GLUE 8 tasks × 多方法 × 多 stage × 多 backend 主循环 |
| `one_click_glue.sh` | 单方法 GLUE pipeline（compress → finetune → eval） |
| `run_superglue_benchmark.sh` | SuperGLUE + HANS + ANLI benchmark |

---

## 分析 / 绘图工具

| 脚本 | 用途 |
|------|------|
| `analyze_compute.py` | 计算 FLOPs / MFU / arithmetic intensity，写入 expB.csv |
| `plot_backend_sweep.py` | 后端对比图（latency / speedup / memory）|
| `plot_combined_figure.py` | 综合大图（4 metrics × 2 tasks）|
| `plot_flops_breakdown.py` | FLOPs breakdown stacked bar（compute quality + seq padding）|
| `plot_nsys_kernel.py` | nsys kernel 时间 breakdown 图（expD）|
| `parse_nsys_summary.py` | 解析 nsys 输出摘要（expD）|
| `analyze_results.py` | GLUE JSON 结果分析 |
| `generate_comparison_table.py` | 输出 Markdown / LaTeX / CSV 对比表 |

---

## CSV 文件说明

| 文件 | 写入方式 | 来源 |
|------|----------|------|
| `expA.csv` | append | `run_encoder_benchmark.py --out_csv`（SuperGLUE 结果）|
| `expB.csv` | append（文件不存在时写 header）| `analyze_compute.py --out_csv` |
| `expC_seqlen.csv` | append | `analyze_compute.py --out_csv` |
| `expC_batch.csv` | append | `analyze_compute.py --out_csv` |
| `expD.csv` | 覆盖写 | `parse_nsys_summary.py --out_csv` |

> **注意**：若 `analyze_compute.py` 更新了输出列，需删除旧 `expB.csv` 重新生成，
> 否则新旧行列数不一致会导致图表读取错误。

---

## _retired/（归档，勿删）

| 脚本 | 归档原因 |
|------|----------|
| `run_expA_sdpa.sh` | 已被 `expB.sh` 取代；输出原 `expA_backend.csv`（现已改名 `expB.csv`）|
| `run_sdpa_ablation.sh` | 已被 `expB.sh` 取代；只跑 sdpa 后端且写到 `encoder_runs.csv` |
| `compare_backends.sh` | 旧版后端对比，调用完整 glue_pipeline，功能已被 `expB.sh` 覆盖 |
| `test_fwsvd_full_vs_perhead.sh` | 一次性验证脚本 |

---

## 环境工具

```bash
bash eval_encoder/scripts/test_setup.sh   # 环境 + GPU 检查
```

最后更新：2026-03-02
