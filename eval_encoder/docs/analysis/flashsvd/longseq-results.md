# FlashSVD Long Sequence & Large Batch Test Results

**Date:** 2026-02-09
**Hypothesis:** FlashSVD should be faster with long sequences and large batches

## 🎯 Test Configuration

We tested 3 scenarios to progressively increase parallelism:

| Scenario | Seq Len | Batch Size | Description |
|----------|---------|------------|-------------|
| **Baseline** | 128 | 32 | Short sequence, small batch (known slow) |
| **Long-Small** | 512 | 32 | 4x longer sequence, small batch |
| **Long-Large** | 512 | 64 | 4x longer sequence, 2x larger batch |

Each scenario tested with:
- Methods: SVD, FWSVD
- Backends: naive, flashsvd

---

## 📊 Complete Results

### Baseline: seq=128, batch=32

| Method | Backend | Latency (ms) | Memory (MB) | Speedup | Memory Save |
|--------|---------|--------------|-------------|---------|-------------|
| SVD    | naive   | **42.32** | 360.2 | 1.00x | baseline |
| SVD    | flashsvd | 148.49 | **276.3** | **0.28x** ❌ | -23.3% |
| FWSVD  | naive   | **42.90** | 367.4 | 1.00x | baseline |
| FWSVD  | flashsvd | 146.11 | **283.1** | **0.29x** ❌ | -22.9% |

**Finding:** FlashSVD is **3.5x slower** (as expected from previous tests)

---

### Long-Small: seq=512, batch=32

| Method | Backend | Latency (ms) | Memory (MB) | Speedup | Memory Save |
|--------|---------|--------------|-------------|---------|-------------|
| SVD    | naive   | **271.32** | 1088.4 | 1.00x | baseline |
| SVD    | flashsvd | 687.17 | **440.5** | **0.39x** ❌ | -59.5% |
| FWSVD  | naive   | **275.95** | 1099.3 | 1.00x | baseline |
| FWSVD  | flashsvd | 693.21 | **451.4** | **0.40x** ❌ | -58.9% |

**Finding:** Even with **4x longer sequences**, FlashSVD is still **2.5x slower**!

**Positive:** Memory savings improved from 23% to **59%** with longer sequences

---

### Long-Large: seq=512, batch=64 (IDEAL FOR FLASHSVD)

| Method | Backend | Latency (ms) | Memory (MB) | Speedup | Memory Save |
|--------|---------|--------------|-------------|---------|-------------|
| SVD    | naive   | **629.11** | 1960.7 | 1.00x | baseline |
| SVD    | flashsvd | 1475.80 | **664.9** | **0.43x** ❌ | -66.1% |

**Finding:** Even in the BEST CASE scenario (long seq + large batch), FlashSVD is STILL **2.35x slower**!

**Positive:** Memory savings reached **66%** - very significant!

---

## 🔍 Analysis: Why FlashSVD Never Gets Faster

### Theory vs Reality

**Theory (FlashSVD design goals):**
- Long sequences → more parallelism → better GPU utilization → faster
- Large batches → amortize kernel launch overhead → faster

**Reality:**
```
seq=128, batch=32: flashsvd is 3.51x SLOWER
seq=512, batch=32: flashsvd is 2.53x SLOWER  (improved but still bad)
seq=512, batch=64: flashsvd is 2.35x SLOWER  (improved but STILL bad)
```

**Conclusion:** Even under ideal conditions, FlashSVD never achieves parity with naive backend.

---

## 💡 Root Cause: Fundamental Overhead

### 1. **Kernel Launch Overhead is Too High**

Even with 4x longer sequences and 2x larger batches, the Triton kernel launch overhead is not sufficiently amortized. The overhead is **fundamental**, not just a tuning issue.

### 2. **PyTorch Native CUDA is Extremely Well-Optimized**

The naive backend uses:
- `torch.einsum` - highly optimized CUDA kernels
- `torch.matmul` - cuBLAS, one of the fastest libraries
- Automatic operator fusion

These are **extremely hard to beat** with custom Triton kernels for encoder-scale workloads.

### 3. **FlashSVD Optimized for Different Scale**

FlashSVD kernels are likely optimized for:
- **Decoder LLMs**: seq_len ≥ 2048, batch_size = 1-4
- **Autoregressive generation**: KV-cache reuse patterns
- **Very long sequences**: seq_len = 4096, 8192, 16384

**Not** for:
- **Encoder models**: seq_len = 128-512, batch_size = 32-64
- **Classification tasks**: fixed-length forward passes

---

## ✅ When FlashSVD IS Beneficial: Memory Savings

While FlashSVD is slower, it provides **significant memory savings** for long sequences:

| Scenario | Naive Mem | FlashSVD Mem | Savings |
|----------|-----------|--------------|---------|
| seq=128, batch=32 | 360 MB | 276 MB | **-23%** |
| seq=512, batch=32 | 1088 MB | 441 MB | **-59%** |
| seq=512, batch=64 | 1961 MB | 665 MB | **-66%** |

**Use Case:** If you're **memory-constrained** and can tolerate 2-3x slower inference, FlashSVD enables:
- Processing longer sequences that wouldn't fit in memory
- Larger batch sizes on the same GPU
- Running larger models on smaller GPUs

---

## 📋 Recommendations Updated

### ✅ Use Naive Backend (Default)
- All scenarios where **latency/throughput matters**
- Training and inference workloads
- Short-to-medium sequences (≤ 512)
- When memory is NOT a constraint

### ⚠️ Consider FlashSVD Backend ONLY IF:
1. **Memory is critically constrained**
2. **You can tolerate 2-3x slower inference**
3. **Sequences are long** (≥ 512) for better memory savings
4. Example: Processing very long documents on low-memory GPUs

### ❌ Don't Use FlashSVD For:
- Speed optimization (it's always slower)
- Short sequences (< 256)
- Standard encoder benchmarks
- Production inference where latency matters

---

## 🎯 Final Verdict

**FlashSVD is NOT a speed optimization for encoder models.**

**It is a MEMORY optimization that trades 2-3x speed for memory savings.**

### Speed Comparison Summary:

```
                 Naive vs FlashSVD Speedup
Baseline:        naive is 3.5x FASTER  ✅
Long seq:        naive is 2.5x FASTER  ✅
Long+Large:      naive is 2.4x FASTER  ✅

CONCLUSION: ALWAYS use naive backend for speed
```

### Memory Comparison Summary:

```
                 FlashSVD Memory Savings
Baseline:        -23%  (360→276 MB)      ⚠️ Not worth it
Long seq:        -59%  (1088→441 MB)     💾 Significant
Long+Large:      -66%  (1961→665 MB)     💾 Very significant

CONCLUSION: Use FlashSVD ONLY if memory is critical
```

---

## 🔬 Future Work

Given these results, FlashSVD should be:

1. **Documented** as a memory optimization, NOT a speed optimization
2. **Tested on decoder LLMs** where it was originally designed for
3. **Benchmarked on very long sequences** (seq_len ≥ 2048) to see if speedup ever materializes
4. **Compared with Flash Attention 2** which has better small-batch performance

For encoder benchmarks, **naive backend is the clear winner** for all performance metrics except memory.
