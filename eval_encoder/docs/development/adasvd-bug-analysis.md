# AdaSVD Budget Control Failure Analysis

**Date:** 2026-02-09
**Issue:** AdaSVD budget control completely failed - all budgets (0.1, 0.2, 0.3, 0.5) converged to ~66.5%

**STATUS:** ✅ **FIXED** - See `ADASVD_FIXES_APPLIED.md` for complete fix documentation and verification

---

## 🐛 Root Cause: Two Critical Bugs

### Bug #1: Loss Coefficients Too Small (Primary Cause)

**Location:** `eval_encoder/adasvd_refactored/adasvd_wrapper.py:110`

```python
# CURRENT (WRONG):
loss = task_loss + 0.1 * budget_loss + 0.01 * align_loss
```

**Problem:**
- Budget loss coefficient: **0.1** (should be ~16.0)
- Alignment loss coefficient: **0.01** (should be ~10.0)

**Expected (from original adaptive_rank_selection.py:223):**
```python
# CORRECT:
loss = task_loss + 16.0 * budget_loss + 10.0 * align_loss
```

**Impact:**
The budget constraint is **160x weaker** than intended! The task loss completely dominates, so the optimizer ignores the budget constraint and just minimizes task loss.

### Bug #2: Budget Calculation Base is Wrong

**Location:** `eval_encoder/adasvd_refactored/adaptive_rank_selection.py:144-152`

```python
def parameter_budget(op_list: List[MaskedSVDLinear], masks: List[torch.Tensor], p: float):
    device = masks[0].device
    Tm, Tmax = torch.zeros((), device=device), torch.zeros((), device=device)
    for op, m in zip(op_list, masks):
        Tm   = Tm   + op.param_count_given_mask(m)  # Current params
        Tmax = Tmax + (op.in_features + op.out_features) * op.R  # Full SVD params
    Tmax = p * Tmax  # ← BUG: Budget relative to SVD params, not original model params!
    return torch.log(torch.clamp(torch.maximum(Tm, Tmax) / (Tmax + 1e-12), min=1.0 + 1e-12))
```

**Problem:**
- `Tmax` is computed as `sum((M+N) * R)` where R is the **full rank** (min(M, N))
- Budget `p` is then applied to this: `Tmax = p * Tmax`
- **This means budget is relative to SVD decomposition, not original model!**

**Example for a [768×768] Linear layer:**
- Original params: 768 × 768 = 589,824
- Full SVD params: (768 + 768) × 768 = 1,179,648
- **SVD decomposition already uses 2x more parameters!**
- Budget 0.5 → 0.5 × 1,179,648 = 589,824 (same as original!)
- So budget=0.5 actually means "keep 100% of original parameters"

**Correct calculation should be:**
```python
# Original model params (not SVD params):
T_original = sum(op.in_features * op.out_features for op in op_list)
Tmax = p * T_original  # Budget relative to ORIGINAL model
```

---

## 📊 Observed Behavior

### All Budgets Converge to 66.5%

| Target Budget | Actual Params | Ratio | Status |
|--------------|---------------|-------|--------|
| 0.1 (10%)    | 66.57%        | 6.66x over | ❌ Failed |
| 0.2 (20%)    | 66.57%        | 3.33x over | ❌ Failed |
| 0.3 (30%)    | 66.56%        | 2.22x over | ❌ Failed |
| 0.5 (50%)    | 66.55%        | 1.33x over | ❌ Failed |

### Why 66.5%?

Looking at the training logs from our tests:

```
Step   0 | Loss: 510.8442 | Ratio: 0.632 | Target: 0.500
Step  50 | Loss: 129.3506 | Ratio: 0.681 | Target: 0.500
Step 100 | Loss:  23.9243 | Ratio: 0.708 | Target: 0.500
Step 150 | Loss:  14.6720 | Ratio: 0.723 | Target: 0.500
Step 200 | Loss:   8.4343 | Ratio: 0.731 | Target: 0.500
Step 250 | Loss:   6.4497 | Ratio: 0.738 | Target: 0.500
Step 300 | Loss:   5.1290 | Ratio: 0.744 | Target: 0.500
Step 350 | Loss:   4.6155 | Ratio: 0.748 | Target: 0.500
```

**Observations:**
1. Ratio keeps **increasing** (0.632 → 0.748) even though target is 0.500
2. Loss decreases (510 → 4.6), meaning optimizer only cares about task loss
3. **Budget constraint is completely ignored**

The ratio converges to ~0.665 because:
- With weak budget penalty (0.1 coefficient), the model prioritizes accuracy
- It selects enough ranks to maintain reasonable accuracy
- ~66.5% is the "natural" convergence point for this task/model
- Budget target has no influence on the final result

---

## 🔬 Detailed Analysis

### Loss Function Comparison

**Original (adaptive_rank_selection.py:223):**
```python
lambda_param = 16.0  # Budget penalty coefficient
gamma_align = 10.0   # Alignment penalty coefficient

loss = task_loss + lambda_param * rparam + gamma_align * align
```

**Current (adasvd_wrapper.py:110):**
```python
# Coefficients hard-coded to tiny values!
loss = task_loss + 0.1 * budget_loss + 0.01 * align_loss
```

**Relative Strength:**
```
Metric          | Original | Current | Ratio
----------------|----------|---------|--------
Budget weight   | 16.0     | 0.1     | 160x weaker
Alignment weight| 10.0     | 0.01    | 1000x weaker
```

With such weak penalties, the optimizer essentially solves:
```
minimize task_loss  (subject to: almost no constraints)
```

### Expected vs Actual Training Behavior

**Expected (with correct coefficients):**
1. Initially: Task loss high, budget penalty very high
2. Middle: Balance between task accuracy and budget
3. Final: Converge to target budget with acceptable accuracy

**Actual (with bug):**
1. Initially: Task loss high, budget penalty negligible
2. Middle: Task loss decreases, budget grows
3. Final: Task loss minimized, budget ignored → ~66.5%

---

## 🛠️ Fix

### Option 1: Fix adasvd_wrapper.py (Simple)

```python
# File: eval_encoder/adasvd_refactored/adasvd_wrapper.py
# Line 110

# OLD:
loss = task_loss + 0.1 * budget_loss + 0.01 * align_loss

# NEW:
loss = task_loss + 16.0 * budget_loss + 10.0 * align_loss
```

### Option 2: Fix Both Bugs (Complete)

**Step 1:** Fix loss coefficients (as above)

**Step 2:** Fix budget calculation base

```python
# File: eval_encoder/adasvd_refactored/adaptive_rank_selection.py
# Lines 144-152

def parameter_budget(op_list: List[MaskedSVDLinear], masks: List[torch.Tensor], p: float):
    device = masks[0].device
    Tm = torch.zeros((), device=device)
    T_original = torch.zeros((), device=device)

    for op, m in zip(op_list, masks):
        Tm = Tm + op.param_count_given_mask(m)
        T_original = T_original + (op.in_features * op.out_features)

    Tmax = p * T_original  # Budget relative to ORIGINAL params
    return torch.log(torch.clamp(torch.maximum(Tm, Tmax) / (Tmax + 1e-12), min=1.0 + 1e-12))
```

**Step 3:** Fix budget report calculation

```python
# File: eval_encoder/adasvd_refactored/adasvd_wrapper.py
# Lines 119-124, 149-153

# OLD:
max_params = sum((op.in_features + op.out_features) * op.R for op in op_list)

# NEW:
max_params_original = sum(op.in_features * op.out_features for op in op_list)
ratio = total_params / max_params_original  # Relative to ORIGINAL model
```

---

## 🧪 Test to Verify Fix

After applying fixes, run:

```bash
python run_encoder_benchmark.py \
    --method adasvd \
    --budget 0.3 \
    --backend naive \
    --model_id textattack/bert-base-uncased-SST-2 \
    --task sst2 \
    --seq_len 128 \
    --batch_size 32 \
    --calib_batches 4
```

**Expected output:**
```
Step   0 | Loss: XXX | Ratio: 0.XXX | Target: 0.300
Step  50 | Loss: XXX | Ratio: 0.XXX | Target: 0.300
...
Step 350 | Loss: XXX | Ratio: ~0.30 | Target: 0.300
                              ↑
                       Should converge to target!
```

**Check budget report:**
```bash
cat ars_out/budget_report.json
```

Should show: `"achieved_ratio": ~0.30` (not 0.665!)

---

## 📊 Impact on Results

### Current Results (with bugs):

| Method | Budget | Params | Memory | Speed | Accuracy |
|--------|--------|--------|--------|-------|----------|
| AdaSVD | 0.1    | 66.6%  | 1178MB | 172ms | 89.4%    |
| AdaSVD | 0.2    | 66.6%  | 1178MB | 176ms | 89.4%    |
| AdaSVD | 0.3    | 66.6%  | 1178MB | 178ms | 89.5%    |
| AdaSVD | 0.5    | 66.6%  | 1178MB | 181ms | 89.4%    |

All identical! Budget control completely broken.

### Expected After Fix:

| Method | Budget | Params | Memory | Speed | Accuracy |
|--------|--------|--------|--------|-------|----------|
| AdaSVD | 0.1    | ~10%   | ~250MB | ~50ms | ~70%     |
| AdaSVD | 0.2    | ~20%   | ~350MB | ~70ms | ~78%     |
| AdaSVD | 0.3    | ~30%   | ~450MB | ~90ms | ~83%     |
| AdaSVD | 0.5    | ~50%   | ~650MB | ~120ms| ~88%     |

Gradual trade-off between compression and accuracy.

---

## 💡 Why This Matters

### Current Situation:
- AdaSVD advertises "adaptive per-operation ranks with budget control"
- **Reality:** Budget control doesn't work, always uses 66.5%
- **Worse than fixed-rank methods:**
  - DRONE rank=256 (50% params): 88.6% accuracy, 305 MB, 35 ms
  - AdaSVD budget=0.5 (66.5% params): 89.4% accuracy, 1178 MB, 181 ms
  - **DRONE is better: fewer params, 4x less memory, 5x faster!**

### After Fix:
- AdaSVD could actually adapt to different budgets
- Useful for exploring accuracy-compression trade-offs
- But likely still slower than DRONE due to training overhead

---

## 🎯 Recommendation

### Short-term:
1. **Document the bug clearly** in reports
2. **Do NOT recommend AdaSVD** for production
3. **Recommend DRONE** as the best method

### Long-term:
1. Fix both bugs (loss coefficients + budget base)
2. Re-run AdaSVD benchmarks with fixes
3. Compare fixed AdaSVD vs DRONE
4. Update documentation with corrected results

### Why Not Fix Now?
- Fixes require modifying imported code from baselines/
- Need to verify fixes don't break other use cases
- Re-running all AdaSVD tests would take 6-8 hours
- **DRONE already provides better results without these issues**

---

## 📝 Summary

**AdaSVD Budget Control Failed Due To:**

1. ❌ **Loss coefficients 160x too small** (0.1 vs 16.0)
   - Budget penalty negligible
   - Optimizer ignores budget constraint

2. ❌ **Budget calculated wrong** (relative to SVD params, not original)
   - Budget target meaningless
   - Convergence independent of target

3. ⚠️ **Result:** All budgets (0.1-0.5) converge to 66.5%

**Implications:**
- AdaSVD claims of "adaptive control" are false (as implemented)
- DRONE provides better compression at lower memory/latency
- Current AdaSVD results should be marked as "budget control failed"

**Action:**
- ✅ Documented bug in ADASVD_BUDGET_BUG_ANALYSIS.md
- ✅ Updated reports to note budget failure
- 🔲 TODO: Fix bugs and re-run (low priority, DRONE superior anyway)

---

*Bug Analysis | 2026-02-09 | Discovered during systematic benchmarking*
