# Peak Memory Measurement Audit

**Date**: February 11, 2026
**Status**: 🔍 **AUDIT COMPLETE**

---

## 📋 Executive Summary

Audited peak memory measurement logic across three implementations to identify potential issues with `torch.cuda.reset_peak_memory_stats()` calls that may cause underreporting of true peak memory usage.

### Key Finding

**All three implementations reset peak memory stats before inference measurement**, which causes them to report only inference peak rather than the true peak across all phases (calibration + local update + inference).

---

## 🔍 Detailed Analysis

### 1. RoBERTaWhiten (v2) - ✅ FIXED

**File**: `src/encoders/RoBERTaWhiten/profile_svdllm_v2_simple_ffnwo.py`

**Problem (Before Fix)**:
- Line 543: Reset peak stats after local update
- Line 355: Reset peak stats in evaluate function
- **Result**: Only reported inference peak (786.4 MiB), missed local update peak (1746.7 MiB)

**Fix Applied**:
```python
# Before evaluate, capture peak (includes local update)
peak_before_eval = torch.cuda.max_memory_allocated() / 1024**2

acc, peak_lr, t = acc_peak_time(model, loader, device, task_name)

# Use true peak (max of all phases)
true_peak = max(peak_before_eval, peak_lr)
print(f"RoBERTa Whitening v2 | acc={acc:.4f} | peak ={true_peak:6.1f} MiB")
print(f"  (Peak before eval: {peak_before_eval:.1f} MiB, Peak during eval: {peak_lr:.1f} MiB)")
```

**Results After Fix**:
```
📊 Local update completed:
  • Peak memory during update: 1746.7 MiB

RoBERTa Whitening v2 | acc=0.8549 | peak =1746.7 MiB
  (Peak before eval: 1746.7 MiB, Peak during eval: 786.4 MiB)
```

**Status**: ✅ **FIXED** - Now correctly reports true peak (1746.7 MiB)

---

### 2. BERTWhiting (v2) - ⚠️ HAS SAME ISSUE

**File**: `src/encoders/BERTWhiting/profile_svdllm_v2_simple_ffnwo.py`

**Problem Locations**:
- **Line 363**: Reset peak stats in `acc_peak_time()` function
  ```python
  def acc_peak_time(mdl, loader, device, task_name: str):
      mdl.eval()
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()  # ← RESETS PEAK!
      # ... inference ...
      peak = torch.cuda.max_memory_allocated() / 1024**2
      return total / max(steps, 1), peak, ms_per_batch
  ```

- **Line 735**: Initial reset (this one is OK - at start of entire process)
  ```python
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()  # ← OK: at very start
  torch.cuda.synchronize()
  ```

**Flow**:
1. Line 735: Reset (start) ✅
2. Lines 738-754: Calibration + SVD decomposition
3. Lines 756-764: **Local update** (produces peak)
4. Line 798: Records `max_memory_allocated()` (should include local update peak)
5. Line 805: Calls `acc_peak_time()` which **resets at line 363** ❌
6. Line 806: Reports only inference peak, **missing local update peak**

**Impact**:
```python
# Line 798 - captures peak INCLUDING local update
with_act = torch.cuda.max_memory_allocated() / 1024**2
print(f"low-rank model storage with GPU redundancy: {with_act:.1f} MiB")  # Correct!

# Line 805-806 - but then evaluate resets and reports only inference peak
acc, peak_lr, t = acc_peak_time(model, loader, device, task_name)
print(f"Data-aware (Whiting) | acc={acc:.4f} | peak ={peak_lr:6.1f} MiB")  # Wrong!
                                                         # ↑ Only inference peak
```

**Status**: ⚠️ **NEEDS FIX** - Same issue as RoBERTaWhiten had

---

### 3. eval_encoder - 🟡 BY DESIGN (but could be clearer)

**File**: `eval_encoder/run_encoder_benchmark.py`

**Reset Locations**:
- **Line 686**: Reset in full validation mode
  ```python
  if is_cuda:
      torch.cuda.synchronize()
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()  # Before full dataset measurement
  ```

- **Line 730**: Reset in standard mode (per run)
  ```python
  if is_cuda:
      torch.cuda.synchronize()
      torch.cuda.reset_peak_memory_stats()  # Before each run
  ```

**Flow**:
1. Lines 999-1000: `compress_model()` - calibration happens here
2. Line 1017: `evaluate_task()` - accuracy measurement
3. Lines 1022-1065: **Optional** reload model to clean GPU state
4. Lines 1075-1078: `measure_performance()` - resets at line 686/730 ❌

**Design Intent**:
The code seems designed to measure **inference-only peak**, intentionally excluding calibration peak. This is supported by:
- `--reload-before-perf` option (line 1022) that explicitly cleans GPU state
- Comment at line 1021: "This ensures clean GPU state after calibration backward passes"

**Issue**:
Even without `--reload-before-perf`, the reset at line 686/730 still excludes calibration peak from the performance measurement.

**Impact**:
- Reported peak only reflects inference memory
- Calibration phase peak is not captured in the final CSV results
- Users may underestimate total memory requirements

**Recommendations**:
1. **Option A**: Add a flag to optionally report both peaks separately
   ```python
   # Report both calibration peak and inference peak
   calib_peak = torch.cuda.max_memory_allocated() / 1024**2  # After compression
   # ... then reset for inference ...
   infer_peak = measured in measure_performance()
   # Report: calib_peak, infer_peak, max(calib_peak, infer_peak)
   ```

2. **Option B**: Document clearly that reported peak is inference-only
   ```python
   # Add comment or column name clarification
   "peak_mem_inference_mb" instead of just "peak_mem_mb"
   ```

**Status**: 🟡 **BY DESIGN** - But could be clearer about what "peak" means

---

## 📊 Comparison Table

| Implementation | Reset Location | Phases Included | Reported Peak | True Peak | Status |
|----------------|----------------|-----------------|---------------|-----------|--------|
| **RoBERTaWhiten v2 (fixed)** | Line 355 (but captured before) | Calibration + Local Update + Inference | 1746.7 MiB | 1746.7 MiB | ✅ Correct |
| **BERTWhiting v2** | Line 363 (in evaluate) | Inference only | ~700-800 MiB | ~1700 MiB (estimated) | ⚠️ Under-reported |
| **eval_encoder** | Lines 686/730 | Inference only | Varies | Not captured | 🟡 By design |

---

## 🔧 Root Cause Analysis

### Why This Happens

All implementations follow this pattern:
```python
# Phase 1: Calibration/Training (produces peak)
calibrate_model(...)
peak_after_calib = torch.cuda.max_memory_allocated()  # Captures true peak

# Phase 2: Cleanup (optional)
del teacher
torch.cuda.empty_cache()

# Phase 3: Inference measurement
def evaluate():
    torch.cuda.reset_peak_memory_stats()  # ← RESETS THE PEAK!
    # ... inference ...
    peak = torch.cuda.max_memory_allocated()  # Only inference peak
    return peak
```

### Why It's Problematic

1. **User Expectation**: Users expect "peak memory" to mean "maximum memory used by the entire program"
2. **Deployment Planning**: Under-reported peak can lead to OOM errors in production
3. **Fair Comparison**: Different implementations may include/exclude different phases

### Why It Happens

- Historical reason: Inference measurement wants clean baseline
- PyTorch API: `reset_peak_memory_stats()` is global, can't scope it per-phase
- Documentation: Not always clear what "peak" refers to

---

## ✅ Recommended Fixes

### For BERTWhiting v2 (Immediate)

Apply the same fix as RoBERTaWhiten:

```python
# In profile_svdllm_v2_simple_ffnwo.py, around line 803-806

# ─── Evaluate ────────────────────────────────────────────────────────────
# Capture peak before evaluate (includes local update)
peak_before_eval = torch.cuda.max_memory_allocated() / 1024**2

metric_name = "pearson" if task_name == "stsb" else "acc"
acc, peak_lr, t = acc_peak_time(model, loader, device, task_name)

# Use the true peak (max of all phases)
true_peak = max(peak_before_eval, peak_lr)
print(f"Data-aware (Whiting) | {metric_name}={acc:.4f} | peak ={true_peak:6.1f} MiB | {t:6.1f} ms/b")
print(f"  (Peak before eval: {peak_before_eval:.1f} MiB, Peak during eval: {peak_lr:.1f} MiB)")
```

### For eval_encoder (Enhancement)

Add explicit tracking of both peaks:

```python
# After compress_model (around line 1000)
calib_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"[compress] Peak memory during compression: {calib_peak_mb:.1f} MB")

# After measure_performance (around line 1078)
print(f"[perf] Peak memory during inference: {peak_mem_mb:.1f} MB")
print(f"[perf] Overall peak memory: {max(calib_peak_mb, peak_mem_mb):.1f} MB")

# In CSV output, add both columns:
# "peak_calib_mb", "peak_infer_mb", "peak_overall_mb"
```

### General Best Practice

**Always report what "peak" means**:
```python
# Good ✅
print(f"Peak memory (inference only): {peak:.1f} MiB")
print(f"Peak memory (all phases): {peak:.1f} MiB")

# Bad ❌
print(f"Peak memory: {peak:.1f} MiB")  # Ambiguous!
```

---

## 📝 Testing Verification

### BERTWhiting v2 (Before Fix)

Expected behavior:
```bash
cd src/encoders/BERTWhiting
python profile_svdllm_v2_simple_ffnwo.py

# Check output:
# "low-rank model storage with GPU redundancy: XXXX MiB"  ← True peak (with local update)
# "Data-aware (Whiting) | ... | peak = YYYY MiB"         ← Only inference peak
# XXXX should be >> YYYY (like 1700 vs 800)
```

### After Applying Fix

Should show:
```
Data-aware (Whiting) | acc=0.XXXX | peak =1700.0 MiB
  (Peak before eval: 1700.0 MiB, Peak during eval: 800.0 MiB)
```

---

## 🎯 Impact Assessment

### Critical Impact (BERTWhiting v2)

- **Severity**: HIGH
- **User Impact**: Users may underestimate memory requirements by ~2x
- **Fix Difficulty**: LOW (5 lines of code, same as RoBERTaWhiten)
- **Recommendation**: **Fix immediately**

### Medium Impact (eval_encoder)

- **Severity**: MEDIUM
- **User Impact**: Inference peak is useful, but full peak would be more complete
- **Fix Difficulty**: MEDIUM (requires changes to CSV schema and reporting)
- **Recommendation**: **Enhance in next version** with separate columns

---

## 📚 Lessons Learned

1. **Always capture peak before reset**: If you need to reset for a specific measurement, capture the current peak first
2. **Report what you measure**: Be explicit about whether peak includes calibration/training
3. **Consider multiple peaks**: Report both "peak during inference" and "peak overall" when relevant
4. **Document assumptions**: Clearly state what phases are included in "peak memory"
5. **Test with verbose output**: Print intermediate peaks to verify logic

---

## 🎉 Conclusion

### Summary of Findings

- ✅ **RoBERTaWhiten v2**: Fixed, now correctly reports true peak
- ⚠️ **BERTWhiting v2**: Has same issue, needs fix (5 lines)
- 🟡 **eval_encoder**: By design, but could be enhanced

### Recommended Actions

1. **Immediate**: Apply fix to BERTWhiting v2 (high priority)
2. **Short-term**: Update documentation for eval_encoder to clarify "inference-only peak"
3. **Long-term**: Consider adding "--report-full-peak" option to eval_encoder

---

**Audit Completed**: February 11, 2026
**Auditor**: Claude Sonnet 4.5
**Status**: ✅ **COMPLETE**
