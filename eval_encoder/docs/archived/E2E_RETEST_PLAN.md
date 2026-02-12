# E2E Memory Retest Plan

**Date**: 2026-02-11
**Reason**: 原有测试缺少 E2E memory 数据，需要补充

---

## 📊 测试状态总览

### ✅ 已有 E2E 数据（2月11日）
- `comprehensive_test_results.csv` - **9 个测试** ✅
  - 包含：dense, svd, fwsvd, drone, adasvd
  - 格式：`peak_mem_infer_mb` + `peak_mem_e2e_mb`

### ❌ 缺少 E2E 数据（2月9日）

| CSV 文件 | 测试数 | 测试内容 | 优先级 |
|---------|--------|----------|--------|
| `encoder_runs_sst2_adasvd_refactored_5budgets.csv` | 10 | AdaSVD 5个预算（0.1-0.7） | **P1 必须** |
| `encoder_runs_flashsvd_longseq_test.csv` | 12 | 长序列（256/512/1024） | **P1 必须** |
| `encoder_runs_flashsvd_comparison.csv` | 4 | FlashSVD vs Naive | P2 可选 |
| `encoder_runs_small_ranks_comparison.csv` | 3 | 小 Rank（16/32/64） | P2 可选 |
| `encoder_runs.csv` | 8 | 早期一般测试 | P3 跳过 |
| **总计** | **37** | - | - |

---

## 🎯 重测建议

### Priority 1: 核心数据（推荐）

**必须重测的配置**：
1. **AdaSVD 多预算对比** (10 个测试)
   - 5个预算：0.1, 0.2, 0.3, 0.5, 0.7
   - 2个后端：naive, flashsvd
   - **原因**: AdaSVD e2e 内存是推理的 **2.13-2.16x**，对比不同预算的行为很重要

2. **长序列测试** (12 个测试)
   - 3个序列长度：256, 512, 1024
   - 4个方法：svd, fwsvd, drone, adasvd
   - **原因**: 长序列的 calibration 内存开销可能显著增加

**预计时间**: ~40 分钟

**运行方法**:
```bash
cd /mnt/e/learning/SVD-Benchmark/lowrankarena/lowrankarena-main/lowrankarena-main/eval_encoder
bash retest_e2e_priority1.sh
```

**输出**: `eval_results/e2e_priority1_retest.csv`

---

### Priority 2: 补充数据（可选）

**可选重测的配置**：
1. **FlashSVD 对比** (4 个测试)
   - svd + fwsvd，naive vs flashsvd
   - **原因**: 验证 FlashSVD 不影响 e2e memory

2. **小 Rank 对比** (3 个测试)
   - rank=16, 32, 64
   - **原因**: 完整性，但使用较少

**预计时间**: ~15 分钟

---

### Priority 3: 可以跳过

**不需要重测**：
- `encoder_runs.csv` (8 个测试)
- **原因**: 已被 `comprehensive_test_results.csv` 覆盖

---

## 🔍 E2E Memory 数据的重要性

### 为什么需要 E2E 数据？

**之前的问题**（只有 `peak_mem_mb`）:
- 只记录了**推理阶段**的内存峰值
- **忽略了** calibration 阶段的巨大内存开销
- 用户会**低估** 2-5.5x 的实际内存需求

**现在的改进**（`peak_mem_infer_mb` + `peak_mem_e2e_mb`）:
```
E2E Peak = max(Compression Peak, Inference Peak)
```

### 实际影响（已验证）

| 方法 | 推理峰值 | E2E 峰值 | 隐藏内存 | 倍数 |
|------|---------|---------|---------|------|
| **Dense** | 291 MB | 291 MB | 0 MB | 1.00x |
| **SVD** | 269 MB | 275 MB | +6 MB | 1.02x |
| **FWSVD** | 276 MB | **1535 MB** | **+1259 MB** | **5.55x** ❗❗ |
| **DRONE** | 270 MB | **1323 MB** | **+1053 MB** | **4.89x** ❗❗ |
| **AdaSVD** | 1080 MB | **2299 MB** | **+1219 MB** | **2.13x** ❗ |

**结论**:
- SVD（无 calibration）: e2e ≈ 推理（差异 <10%）
- FWSVD/DRONE/AdaSVD（有 calibration）: e2e = 2-5.5x 推理 ⚠️

---

## 🚀 执行步骤

### Step 1: 运行 Priority 1 重测（推荐）

```bash
cd eval_encoder
bash retest_e2e_priority1.sh 2>&1 | tee retest_e2e_priority1.log
```

### Step 2: 查看结果

```bash
# 查看新结果
head -20 eval_results/e2e_priority1_retest.csv

# 对比旧结果
python scripts/utils/organize_csvs.py
```

### Step 3: 合并和分析

```bash
# 合并所有有 e2e 数据的 CSV
cat eval_results/comprehensive_test_results.csv \
    eval_results/e2e_priority1_retest.csv \
    > eval_results/all_tests_with_e2e.csv

# 生成分析报告
python analyze_e2e_memory.py
```

---

## 📝 CSV 格式对比

### 旧格式（2月9日）
```csv
...,latency_ms,throughput_sps,peak_mem_mb,param_ratio,...
...,129.06,247.9,1079.8,0.2962,...
```
- 只有 `peak_mem_mb`（含义不明确）

### 新格式（2月11日）
```csv
...,latency_ms,throughput_sps,peak_mem_infer_mb,peak_mem_e2e_mb,peak_mem_mb,param_ratio,...
...,134.3,238.3,1079.8,2299.3,2299.3,0.2962,...
```
- `peak_mem_infer_mb`: 推理阶段峰值（纯推理内存）
- `peak_mem_e2e_mb`: 端到端峰值（包括 calibration）
- `peak_mem_mb`: 向后兼容（= peak_mem_e2e_mb）

---

## 🎯 推荐行动

### 立即执行（重要）
✅ **运行 Priority 1 重测** (~40 分钟)
- 补充 AdaSVD 多预算的 e2e 数据
- 补充长序列测试的 e2e 数据

### 可选执行（补充）
⏸️ **运行 Priority 2 重测** (~15 分钟)
- FlashSVD 对比
- 小 Rank 对比

### 不需要执行
❌ **跳过 Priority 3**
- encoder_runs.csv 已过时

---

## 📄 相关文档

- `COMPREHENSIVE_TEST_RESULTS.md` - 最新测试结果（包含 e2e 分析）
- `docs/development/peak-memory-analysis.md` - E2E memory 实现细节
- `run_encoder_benchmark.py:1092` - E2E memory 计算逻辑

---

**建议**: 先运行 Priority 1 重测（~40分钟），获取最关键的 e2e 数据，然后再决定是否运行 Priority 2。
