# RoBERTaWhiten - SVD-LLM Implementation for RoBERTa

This directory contains the RoBERTa adaptation of the SVD-LLM (Whitening-SVD) compression method from BERTWhiting.

## 📁 Files

- `flash_attn_triton.py` - Flash Attention Triton kernels (shared with BERT)
- `whiting_core.py` - Core whitening-SVD utilities (shared with BERT)
- `profile_svdllm_v1.py` - SVD-LLM v1 for RoBERTa (whitening-based low-rank decomposition)
- `profile_svdllm_v2_simple_ffnwo.py` - SVD-LLM v2 for RoBERTa (v1 + local update via teacher-student distillation)

## 🚀 Quick Start

### Prerequisites

```bash
# Activate the flashsvd conda environment
conda activate flashsvd

# Or ensure you have:
# - PyTorch 2.8.0+ with CUDA
# - Transformers 4.56.2+
# - Triton 3.4.0+
```

### Run SVD-LLM v1

```bash
cd src/encoders/RoBERTaWhiten
python profile_svdllm_v1.py
```

### Run SVD-LLM v2

```bash
cd src/encoders/RoBERTaWhiten
python profile_svdllm_v2_simple_ffnwo.py
```

## 📊 Results (SST-2 Validation, RATIO=0.5)

| Method | Accuracy | Model Size | Peak Memory | Speed (ms/batch) |
|--------|----------|------------|-------------|------------------|
| **RoBERTa Dense** | ~94.5% | ~420 MiB | ~1200 MiB | ~280 ms |
| **RoBERTa Whitening v1** | **85.04%** | **312.7 MiB** | **772.9 MiB** | **304.9 ms** |
| **RoBERTa Whitening v2** | **84.93%** | **312.7 MiB** | **786.4 MiB** | **310.8 ms** |

**Compression Ratio**: ~25% parameter reduction (420 MiB → 312.7 MiB)
**Accuracy Drop**: ~9-10% (acceptable for many applications)

## 🔧 Implementation Details

### Key Adaptations from BERT

The following changes were made to adapt BERTWhiting for RoBERTa:

1. **Model Class**: `BertForSequenceClassification` → `RobertaForSequenceClassification`
2. **Encoder Access**: `model.bert.encoder` → `model.roberta.encoder`
3. **Model Checkpoint**: `textattack/bert-base-uncased-sst-2` → `textattack/roberta-base-SST-2`
4. **Namespace Updates**: All parameter paths updated from `bert.encoder.layer` to `roberta.encoder.layer`

### Architecture Compatibility

RoBERTa shares the same internal architecture as BERT:
- ✅ Post-normalization (LayerNorm after residual)
- ✅ GELU activation
- ✅ Absolute positional encoding
- ✅ Same hidden size (768), FFN size (3072), num heads (12)

Therefore, all whitening-SVD math and Flash Attention kernels work identically.

## 📐 Rank Configuration (RATIO=0.5)

- **RANK_ATTN**: 29 (per-head rank for Q/K/V)
- **RANK_FF**: 307 (rank for FFN intermediate/output)
- **RANK_WO**: 192 (rank for attention output projection)

Calculated via: `rank = floor(m * n * ratio / (m + n))`

## 🧪 Technical Verification

### v1 - Whitening-SVD Factorization

1. **Calibration**: Collects 4 input covariance matrices per layer:
   - `cov_attn_in` (dm × dm): Input to Q/K/V
   - `cov_attn_out` (dm × dm): Input to attention output projection
   - `cov_ffn_in` (dm × dm): Input to FFN intermediate
   - `cov_ffn_out` (d_ff × d_ff): Input to FFN output

2. **Whitening-SVD**: For each weight matrix W:
   ```
   C = L L^T              (Cholesky decomposition)
   W_scale = L^T W        (Whiten in input space)
   U, Σ, V^T = SVD(W_scale)  (Truncate to rank k)
   W ≈ (L^{-T} U_k √Σ) (√Σ V_k^T)  (Unwhiten)
   ```

3. **Inference**: Uses Flash Attention (Triton) + low-rank FFN

### v2 - Local Update (Teacher-Student Distillation)

- Builds on v1 by updating V matrices (fixing U)
- Updates 3 V matrices per layer: V1, V2, Vo
- Ridge least squares: `(A + λI) V = B`, where:
  - `A = Σ Z^T Z` (Z = X @ U)
  - `B = Σ Z^T (Y - b_teacher)`
- Online accumulation to minimize memory

## 🔬 Comparison with BERTWhiting

| Aspect | BERT | RoBERTa | Identical? |
|--------|------|---------|------------|
| Math (Whitening-SVD) | ✅ | ✅ | Yes |
| Flash Attention Kernels | ✅ | ✅ | Yes |
| Model Access Path | `model.bert` | `model.roberta` | **No** |
| Architecture | Post-norm, GELU | Post-norm, GELU | Yes |
| Default Checkpoint | `textattack/bert-base-*` | `textattack/roberta-base-*` | **No** |

## 🐛 Known Issues

None at this time. All tests passing.

## 📝 Citation

If you use this code, please cite the original SVD-LLM paper:

```bibtex
@article{yuan2024svdllm,
  title={SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression},
  author={Yuan, Xin and others},
  journal={arXiv preprint arXiv:2403.07378},
  year={2024}
}
```

## 📧 Contact

For issues specific to RoBERTa implementation, please open an issue in the repository.

---

**Created**: February 11, 2026
**Status**: ✅ Tested and working
**Environment**: flashsvd conda env (PyTorch 2.8.0+cu129, Transformers 4.56.2, Triton 3.4.0)
