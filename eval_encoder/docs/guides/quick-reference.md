# FlashSVD Quick Reference Card

## 🎯 最佳配置（一句话推荐）

**生产环境最佳：DRONE rank=32 + FlashSVD @ seq=512, batch=64**
- 72% 内存省 + 9% 更快 + 71% 准确率 = 完美！

---

## 📊 选择流程图

```
你的场景是什么？
│
├─ 内存非常紧张（边缘设备、移动端）
│  └─ ✅ 使用 FlashSVD（任何方法，rank=32-64）
│     收益：65-72% 内存节省
│
├─ 长序列推理（seq ≥ 512）
│  └─ ✅ 使用 FlashSVD（SVD/DRONE，rank=32-64）
│     收益：65-72% 内存省，不变慢或更快
│
├─ 大批次推理（batch ≥ 128）
│  ├─ 用 SVD？
│  │  └─ ✅ 使用 FlashSVD
│  │     收益：48-56% 内存省，快 32-48%！
│  └─ 用 DRONE？
│     └─ ⚠️ 使用 naive（FlashSVD 会慢 14-27%）
│
├─ 短序列 + 小批次 + 小 rank（r=32-64）
│  └─ ✅ 使用 FlashSVD
│     收益：30-40% 内存省，不变慢
│
├─ 短序列 + 小批次 + 大 rank（r=256+）
│  └─ ❌ 不用 FlashSVD
│     问题：仅省 27% 内存，慢 1.8-2x
│
└─ 使用 AdaSVD？
   └─ ❌ 永远不用 FlashSVD
      问题：仅省 6% 内存，慢 2x
```

---

## 🏆 Top 3 配置

### 1️⃣ 极限内存压力
```
Config:  seq=512, batch=64, rank=32, DRONE + FlashSVD
Memory:  1799 MB → 503 MB (72% ↓)
Speed:   415 ms → 379 ms (9% ↑)
Accuracy: 71.0%
Use:     边缘设备、移动端、嵌入式系统
```

### 2️⃣ 大批次快速推理
```
Config:  seq=128, batch=128, rank=64, SVD + FlashSVD
Memory:  673 MB → 347 MB (48% ↓)
Speed:   205 ms → 138 ms (48% ↑)  🔥
Accuracy: 52.3%
Use:     批量评分、离线推理
```

### 3️⃣ 长文本理解
```
Config:  seq=512, batch=32, rank=32, DRONE + FlashSVD
Memory:  942 MB → 294 MB (69% ↓)
Speed:   214 ms → 193 ms (10% ↑)
Accuracy: 66.7%
Use:     文档处理、长文QA、内容分析
```

---

## 📋 方法速查表

| 方法 | 何时用 | 何时不用 | FlashSVD推荐 |
|------|--------|---------|-------------|
| **SVD** | 快速基线、大批次 | 需要高准确率 | ✅ 推荐 |
| **DRONE** | 需要高准确率 | - | ✅ 推荐 |
| **FWSVD** | 短序列小批次 | 大批次（崩溃） | ⚠️ 有条件 |
| **AdaSVD** | - | 任何情况 | ❌ 永不 |

---

## 🚫 避雷指南

### ❌ 绝对不要：
1. **AdaSVD + FlashSVD**
   - 仅省 6% 内存，慢 2 倍
   - 预算控制失败（所有 budget → 66.5%）

2. **FWSVD + 大批次**
   - 慢 6-17 倍！
   - 即使用 FlashSVD 也慢 6 倍

3. **大 rank (256) + 短序列 + FlashSVD**
   - 仅省 27% 内存，慢 1.8-2 倍

### ⚠️ 注意：
- DRONE + FlashSVD 在大批次下慢 14-27%
  - 解决：用 SVD 代替，或用 naive backend

---

## 💾 内存节省对照表

| 场景 | seq | batch | rank | 内存节省 |
|------|-----|-------|------|---------|
| 极限 | 512 | 64 | 32 | **72%** 🥇 |
| 长序列 | 512 | 32 | 32 | **69%** 🥈 |
| 大批次 | 128 | 128 | 32 | **56%** 🥉 |
| 短序列 | 128 | 32 | 32 | 40% |
| 短序列 | 128 | 32 | 256 | 27% |
| AdaSVD | 任何 | 任何 | - | **6%** ❌ |

**规律：seq × batch 越大，内存节省越多！**

---

## ⚡ 速度影响对照表

| 配置 | rank | 方法 | 速度比 | 结果 |
|------|------|------|--------|------|
| seq=512, bs=64 | 32 | SVD/DRONE | **0.91x** | 快 9% ✅ |
| seq=512, bs=32 | 32 | SVD/DRONE | **0.88x** | 快 12% ✅ |
| seq=128, bs=128 | 64 | SVD | **0.68x** | 快 48% 🔥 |
| seq=128, bs=32 | 32 | 任何 | **0.87-1.03x** | 相近 ✅ |
| seq=128, bs=32 | 64 | 任何 | 1.04-1.10x | 稍慢 ⚠️ |
| seq=128, bs=32 | 128 | 任何 | 1.22-1.31x | 慢 25% ⚠️ |
| seq=128, bs=32 | 256 | 任何 | 1.82-1.98x | 慢 90% ❌ |
| 任何 | - | AdaSVD | **2.06x** | 慢 2 倍 ❌ |
| seq=512, bs=64 | 32 | FWSVD | **6.5x** | 慢 6 倍 ❌ |

**规律：rank 越小 + 内存压力越大 → FlashSVD 越快！**

---

## 🔧 实用命令

### 运行基准测试
```bash
# 短序列小 rank
bash scripts/core/test_small_ranks_complete.sh

# 长序列/大批次
bash scripts/core/test_longseq_lowrank.sh

# 极限场景
bash scripts/core/test_extreme_memory.sh
```

### 查看结果
```bash
# 查看所有结果
cat eval_results/final/small_ranks_complete_benchmark.csv

# 统计内存节省
awk -F',' 'NR>1 {print $9, $13, $25}' file.csv | sort
```

---

## 📚 相关文档

- **完整报告：** `FINAL_FLASHSVD_REPORT.md`
- **详细分析：** `SMALL_RANKS_ANALYSIS.md`
- **使用指南：** `BENCHMARK_GUIDE.md`
- **性能分析：** `FLASHSVD_PERFORMANCE_ANALYSIS.md`

---

## 🎓 一句话总结

**FlashSVD = Memory-Efficient（不是 Speed-Efficient）**
- 长序列/大批次 → 用！（省 65-72% 内存，还更快）
- 短序列小 rank → 用！（省 30-40% 内存，不变慢）
- AdaSVD → 永远不用！（仅省 6%，慢 2 倍）
- 大 rank (256+) + 短序列 → 别用！（省得少，慢 2 倍）

---

*快速参考卡 | 更新：2026-02-09*
