# eval_encoder Peak Memory Enhancement Plan

**Date**: February 11, 2026
**Status**: 📋 **DESIGN PROPOSAL**
**Target**: `eval_encoder/run_encoder_benchmark.py`

---

## 🎯 Goals

1. **Capture full peak memory** across all phases (compression + inference)
2. **Report both peaks separately** for transparency
3. **Maintain backward compatibility** with existing CSV schema
4. **Add optional flag** to enable enhanced tracking

---

## 📊 Current Behavior vs Desired Behavior

### Current Behavior ❌

```python
# Compression phase
compress_model(...)  # Peak: ~1700 MB (not captured)

# Performance measurement
torch.cuda.reset_peak_memory_stats()  # ← Loses compression peak!
measure_performance(...)  # Peak: ~750 MB (captured)

# Result: Only reports 750 MB
```

**Output**:
```csv
method,rank,peak_mem_mb
svd,0.5,750.0
```

### Desired Behavior ✅

```python
# Compression phase
compress_model(...)
compression_peak = torch.cuda.max_memory_allocated() / 1024**2  # Capture: ~1700 MB

# Performance measurement
infer_peak = measure_performance(...)  # ~750 MB

# Report both
overall_peak = max(compression_peak, infer_peak)  # 1700 MB
```

**Output**:
```csv
method,rank,peak_compress_mb,peak_infer_mb,peak_overall_mb
svd,0.5,1700.0,750.0,1700.0
```

---

## 🔧 Implementation Design

### Phase 1: Add Peak Tracking in Main Flow

**Location**: `run_encoder_benchmark.py`, main() function

#### Step 1.1: Track Compression Peak

After compression (around line 1000):

```python
# 3) compress
compression_peak_mb = 0.0
if args.method != "dense":
    # Capture peak before compression
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # Start fresh

    model = compress_model(model, args.method, args.rank, args.budget,
                           args.scope, loader, args.device, args.calib_batches,
                           calib_loader=calib_loader, backend=args.backend)

    # Capture compression peak (includes calibration, SVD, local update, etc.)
    if torch.cuda.is_available():
        compression_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[compress] Peak memory during compression: {compression_peak_mb:.1f} MB")
```

#### Step 1.2: Modify measure_performance Return

Currently returns: `(latency_ms, throughput_sps, peak_mem_mb)`

Keep this for backward compatibility, but add a flag to control whether to reset:

```python
def measure_performance(
    model, loader, device, warmup_steps, measure_steps, num_runs,
    full_validation=False,
    reset_peak_before_measure=True  # NEW parameter
):
    """
    ...
    Args:
        reset_peak_before_measure: If True, reset peak stats before measurement.
                                   If False, keep existing peak for cumulative tracking.
    """
    # ... existing code ...

    if full_validation:
        # ... warmup ...

        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if reset_peak_before_measure:  # Only reset if requested
                torch.cuda.reset_peak_memory_stats()

        # ... measure ...
    else:
        # ... standard mode ...

        if is_cuda:
            torch.cuda.synchronize()
            if reset_peak_before_measure:  # Only reset if requested
                torch.cuda.reset_peak_memory_stats()
```

#### Step 1.3: Capture Both Peaks

In main(), after performance measurement (around line 1075):

```python
# 6) measure performance
if args.full_validation:
    print(f"\n[perf] Full validation mode ...")
else:
    print(f"\n[perf] Warmup={args.warmup_steps}  Measure={args.measure_steps} steps ...")

# Option A: Reset and measure inference peak separately (current behavior)
# Option B: Don't reset, capture overall peak (new behavior)
reset_before_perf = not args.track_full_peak  # New flag

latency_ms, throughput_sps, infer_peak_mb = measure_performance(
    model, loader, args.device, args.warmup_steps, args.measure_steps, args.num_runs,
    full_validation=args.full_validation,
    reset_peak_before_measure=reset_before_perf,
)

# Calculate overall peak
if args.track_full_peak:
    overall_peak_mb = infer_peak_mb  # Already includes compression peak
    # Re-calculate inference-only peak for comparison
    # (Would need second measurement pass, or estimate from previous runs)
else:
    overall_peak_mb = max(compression_peak_mb, infer_peak_mb)

# Print detailed breakdown
print(f"\n[memory] Peak Memory Breakdown:")
print(f"  • Compression phase: {compression_peak_mb:.1f} MB")
print(f"  • Inference phase:   {infer_peak_mb:.1f} MB")
print(f"  • Overall peak:      {overall_peak_mb:.1f} MB")

if not args.full_validation:
    if args.num_runs > 1:
        print(f"[perf] MEDIAN: latency={latency_ms:.2f} ms/batch  "
              f"throughput={throughput_sps:.1f} samples/s  "
              f"peak_mem={overall_peak_mb:.1f} MB")  # Use overall peak
```

---

### Phase 2: Add Command-Line Flag

**Location**: `run_encoder_benchmark.py`, parse_args() function

```python
def parse_args():
    parser = argparse.ArgumentParser(...)

    # ... existing args ...

    # NEW: Peak memory tracking options
    perf_group = parser.add_argument_group("performance measurement")
    perf_group.add_argument(
        "--track-full-peak",
        action="store_true",
        default=False,
        help="Track peak memory across all phases (compression + inference). "
             "Default: False (separate peaks for each phase)."
    )
    perf_group.add_argument(
        "--report-peak-breakdown",
        action="store_true",
        default=False,
        help="Report detailed peak memory breakdown (compression, inference, overall). "
             "Adds extra columns to CSV output."
    )

    return parser.parse_args()
```

---

### Phase 3: Update CSV Output Schema

**Location**: `run_encoder_benchmark.py`, end of main() function

#### Current CSV Output (Backward Compatible)

```python
# Default behavior: keep existing schema
csv_data = {
    "model_id": args.model_id,
    "task": args.task,
    "method": args.method,
    "rank": args.rank if args.rank else args.budget,
    "budget": args.budget,
    "scope": args.scope,
    "backend": args.backend,
    "metric_name": metric_name,
    "metric_value": metric_value,
    "latency_ms": latency_ms,
    "throughput_sps": throughput_sps,
    "peak_mem_mb": overall_peak_mb,  # Use overall peak by default
}
```

#### Enhanced CSV Output (With Flag)

```python
# With --report-peak-breakdown
if args.report_peak_breakdown:
    csv_data.update({
        "peak_compress_mb": compression_peak_mb,
        "peak_infer_mb": infer_peak_mb,
        "peak_overall_mb": overall_peak_mb,
    })
else:
    # Backward compatible: single peak column
    csv_data["peak_mem_mb"] = overall_peak_mb
```

---

## 🎨 Alternative Design: Simpler Approach

If we want to avoid complexity, here's a **minimal enhancement**:

### Minimal Enhancement (Recommended)

Just capture and report both peaks, always:

```python
# After compression (line ~1000)
compression_peak_mb = 0.0
if args.method != "dense" and torch.cuda.is_available():
    compression_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

# After performance measurement (line ~1078)
overall_peak_mb = max(compression_peak_mb, infer_peak_mb)

print(f"\n[memory] Peak Memory:")
print(f"  • Compression: {compression_peak_mb:.1f} MB")
print(f"  • Inference:   {infer_peak_mb:.1f} MB")
print(f"  • Overall:     {overall_peak_mb:.1f} MB")

# In CSV, just update the meaning of peak_mem_mb
csv_data["peak_mem_mb"] = overall_peak_mb  # Now includes compression!

# Add comment in CSV header
# "peak_mem_mb: overall peak across compression and inference phases"
```

**Pros**:
- ✅ Simple: ~10 lines of code
- ✅ No new flags needed
- ✅ Backward compatible CSV schema
- ✅ Just changes the meaning of `peak_mem_mb` to be more accurate

**Cons**:
- ⚠️ Changes existing behavior (peak values will increase)
- ⚠️ Can't isolate inference-only peak for comparison

---

## 📋 Implementation Steps

### Step 1: Minimal Enhancement (Week 1)

**Priority**: HIGH
**Difficulty**: LOW
**Impact**: HIGH

```python
# In run_encoder_benchmark.py, main()

# After line 1000 (compress_model)
compression_peak_mb = 0.0
if args.method != "dense" and torch.cuda.is_available():
    compression_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[compress] Peak memory during compression: {compression_peak_mb:.1f} MB")

# After line 1078 (measure_performance)
# Calculate overall peak
overall_peak_mb = max(compression_peak_mb, peak_mem_mb)

# Enhanced reporting
print(f"\n{'='*60}")
print(f"Memory Usage Summary:")
print(f"  Compression phase: {compression_peak_mb:>8.1f} MB")
print(f"  Inference phase:   {peak_mem_mb:>8.1f} MB")
print(f"  Overall peak:      {overall_peak_mb:>8.1f} MB")
print(f"{'='*60}")

# Update CSV
# Replace: "peak_mem_mb": peak_mem_mb
# With:    "peak_mem_mb": overall_peak_mb
```

### Step 2: Add Peak Breakdown Columns (Week 2)

**Priority**: MEDIUM
**Difficulty**: MEDIUM
**Impact**: MEDIUM

```python
# Add to CSV output
csv_data.update({
    "peak_compress_mb": compression_peak_mb,
    "peak_infer_mb": peak_mem_mb,
    "peak_overall_mb": overall_peak_mb,
})

# Keep old column for compatibility
csv_data["peak_mem_mb"] = overall_peak_mb
```

### Step 3: Add Visualization Script (Week 3)

**Priority**: LOW
**Difficulty**: LOW
**Impact**: LOW

Create `eval_encoder/scripts/plot_peak_breakdown.py`:

```python
#!/usr/bin/env python3
"""
Plot peak memory breakdown from CSV results.
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Filter to methods with compression
df_compressed = df[df["method"] != "dense"]

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
methods = df_compressed["method"].unique()
x = range(len(methods))

compress_peaks = [df_compressed[df_compressed["method"]==m]["peak_compress_mb"].values[0]
                  for m in methods]
infer_peaks = [df_compressed[df_compressed["method"]==m]["peak_infer_mb"].values[0]
               for m in methods]

ax.bar(x, compress_peaks, label="Compression Peak")
ax.bar(x, infer_peaks, bottom=compress_peaks, label="Inference Peak")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Peak Memory (MB)")
ax.set_title("Peak Memory Breakdown by Method")
ax.legend()
plt.savefig("peak_breakdown.png")
```

---

## 🧪 Testing Plan

### Test 1: Verify Compression Peak Captured

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id=google-bert/bert-base-uncased \
  --task=sst2 \
  --method=svd \
  --rank=0.5 \
  --backend=naive

# Expected output:
# [compress] Peak memory during compression: 1234.5 MB
# [perf] Peak memory during inference: 678.9 MB
# Overall peak: 1234.5 MB
```

### Test 2: Verify Dense Model (No Compression)

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id=google-bert/bert-base-uncased \
  --task=sst2 \
  --method=dense

# Expected output:
# [compress] Peak memory during compression: 0.0 MB
# [perf] Peak memory during inference: 456.7 MB
# Overall peak: 456.7 MB
```

### Test 3: Verify CSV Output

```bash
# Check CSV has correct columns
head -1 results.csv
# Should show: ...,peak_compress_mb,peak_infer_mb,peak_overall_mb,...

# Check values make sense
tail -1 results.csv
# peak_overall_mb should be >= max(peak_compress_mb, peak_infer_mb)
```

### Test 4: Compare Before/After

```bash
# Before enhancement
python run_encoder_benchmark.py ...
# Old: peak_mem_mb = 678.9 MB (inference only)

# After enhancement
python run_encoder_benchmark.py ...
# New: peak_mem_mb = 1234.5 MB (overall)

# Verify increase makes sense
```

---

## 📊 Expected Results

### Example: BERT-base SVD (rank=0.5)

**Before Enhancement**:
```
method,rank,peak_mem_mb
svd,0.5,728.3
```

**After Enhancement (Minimal)**:
```
method,rank,peak_mem_mb
svd,0.5,1630.7
```

**After Enhancement (Full)**:
```
method,rank,peak_compress_mb,peak_infer_mb,peak_overall_mb
svd,0.5,1630.7,728.3,1630.7
```

### Impact on Existing Results

| Method | Before (MB) | After (MB) | Change |
|--------|-------------|------------|--------|
| dense | 456 | 456 | 0% |
| svd | 728 | 1631 | +124% |
| fwsvd | 750 | 1700 | +127% |
| drone | 740 | 1680 | +127% |
| adasvd | 800 | 1750 | +119% |

---

## ⚠️ Breaking Changes and Migration

### What Changes

1. **CSV column `peak_mem_mb` meaning changes**:
   - Before: Inference-only peak
   - After: Overall peak (max of compression + inference)

2. **Reported values increase**:
   - Compressed methods: ~2x increase
   - Dense method: No change

### Migration Guide

For users with existing scripts:

```python
# OLD CODE (Before enhancement)
df = pd.read_csv("results.csv")
infer_peak = df["peak_mem_mb"]  # This was inference-only

# NEW CODE (After enhancement, backward compatible)
df = pd.read_csv("results.csv")
overall_peak = df["peak_mem_mb"]  # Now includes compression

# If you want inference-only (with enhanced CSV):
if "peak_infer_mb" in df.columns:
    infer_peak = df["peak_infer_mb"]  # Separate column
else:
    infer_peak = df["peak_mem_mb"]  # Fallback for old CSVs
```

### Versioning

Add a version field to CSV to track schema changes:

```python
csv_data["schema_version"] = "2.0"  # Was "1.0" before
```

---

## 🎯 Recommended Approach

**Phase 1 (Immediate)**: Implement **Minimal Enhancement**
- 10 lines of code
- No new flags
- Clear improvement
- ~1 hour of work

**Phase 2 (Optional)**: Add **Peak Breakdown Columns**
- Separate columns for detailed analysis
- ~2 hours of work

**Phase 3 (Nice-to-have)**: Add **Visualization Tools**
- Plotting scripts
- ~2 hours of work

---

## 📝 Code Diff Preview

### Minimal Enhancement Diff

```python
# File: eval_encoder/run_encoder_benchmark.py

def main():
    # ... existing code ...

    # 3) compress
+   compression_peak_mb = 0.0
    if args.method != "dense":
        model = compress_model(...)
+       if torch.cuda.is_available():
+           compression_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
+           print(f"[compress] Peak memory during compression: {compression_peak_mb:.1f} MB")

    # ... existing code ...

    # 6) measure performance
    latency_ms, throughput_sps, peak_mem_mb = measure_performance(...)

+   # Calculate overall peak
+   overall_peak_mb = max(compression_peak_mb, peak_mem_mb)
+
+   print(f"\n{'='*60}")
+   print(f"Memory Usage Summary:")
+   print(f"  Compression phase: {compression_peak_mb:>8.1f} MB")
+   print(f"  Inference phase:   {peak_mem_mb:>8.1f} MB")
+   print(f"  Overall peak:      {overall_peak_mb:>8.1f} MB")
+   print(f"{'='*60}")

    # ... CSV output ...
    csv_data = {
        # ... other fields ...
-       "peak_mem_mb": peak_mem_mb,
+       "peak_mem_mb": overall_peak_mb,  # Changed: now includes compression
    }
```

**Lines changed**: ~15
**Files modified**: 1
**Estimated time**: 1 hour

---

## ✅ Acceptance Criteria

- [ ] Compression peak is captured after compress_model()
- [ ] Overall peak is calculated as max(compression, inference)
- [ ] Console output shows detailed breakdown
- [ ] CSV reports overall peak in peak_mem_mb column
- [ ] Dense method reports 0 for compression peak
- [ ] All existing tests pass
- [ ] New tests verify peak tracking
- [ ] Documentation updated

---

## 🎉 Benefits

1. **Accurate Memory Reporting**: Users see true peak memory requirements
2. **Better Planning**: Deployment planning based on real memory needs
3. **Fair Comparison**: All methods report memory consistently
4. **Transparency**: Users understand where memory is used
5. **Backward Compatible**: Existing scripts continue to work (with updated values)

---

**Plan Created**: February 11, 2026
**Author**: Claude Sonnet 4.5
**Status**: ✅ **READY FOR IMPLEMENTATION**
