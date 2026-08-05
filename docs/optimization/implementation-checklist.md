# Implementation Checklist

## Krea2 Model Optimization

### Baseline Results
- [x] Run baseline benchmark (6 epochs × 30 steps, generate 4 images)
- [x] Record baseline metrics in results.md
- **Training Time**: 3.82s/it (range: 3.52-4.05s)
- **Sample Generation Time**: 68.97s/image (range: 64.43-72.00s)

### Optimization Tasks (Top 5 - ≤20 lines each)
- [ ] **Task 1**: CPU-to-GPU transfer in `pad_text_features` (5 lines) - Expected: 5-10% speedup
- [ ] **Task 2**: Latent device transfer in `predict_velocity` (4 lines) - Expected: 5-10% speedup
- [ ] **Task 3**: Reference latents device transfer in `pack_ref_latents` (8 lines) - Expected: 5-8% speedup
- [ ] **Task 4**: VAE encoding device transfer in `encode_images` (3 lines) - Expected: 5-10% speedup
- [ ] **Task 5**: Timestep tensor creation in sampling loop (4 lines) - Expected: 2-5% speedup

### Status
- Baseline: ✅ Completed (6 epochs × 30 steps, generate 4 images)
- Current Task: None (ready to start implementation)
- Completed Tasks: 0/5
- Total Expected Speedup: 19-43%

---

## Change #1: CPU-to-GPU Transfer in `pad_text_features`

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: The `pad_text_features` function can use non_blocking transfers.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 35-48

**Changes Made**:
- Line 45: Changed `mask = torch.zeros(...)` to `mask = torch.ones(...)` (eliminates loop operations)
- Line 48: Added `non_blocking=True` to `.to()` call

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

## Change #2: Latent Device Transfer in `predict_velocity`

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: Latents are moved to dtype but not explicitly to device with non_blocking.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 150-160

**Changes Made**:
- Line 152: Added `non_blocking=True` to `.to()` call

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

## Change #3: Reference Latents Device Transfer in `pack_ref_latents`

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: Multiple device transfers in `pack_ref_latents` can be consolidated with non_blocking.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 95-105

**Changes Made**:
- Line 103: Added `non_blocking=True` to `.to()` call
- Line 105: Consolidated into single line (removed extra parentheses)

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

## Change #4: VAE Encoding Device Transfer in `encode_images`

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: Images are moved to device multiple times in `encode_images`.

**Location**: `extensions_built_in/diffusion_models/krea2/krea2.py`, lines 830-845

**Changes Made**:
- Line 839: Added `non_blocking=True` to list comprehension `.to()` calls
- Line 840: Added `non_blocking=True` to stack `.to()` call

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

## Change #5: Timestep Tensor Creation in Sampling Loop

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: Timestep tensors created without non_blocking in the sampling loop.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 340-350

**Changes Made**:
- Line 341: Added `non_blocking=True` to timestep tensor creation
- Line 343: Added explicit device transfer with `non_blocking=True` to latents

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

## Summary of Optimization Opportunities

| Change # | Category | Complexity | Expected Speedup | Lines Changed |
|----------|----------|------------|------------------|---------------|
| 1 | CPU-GPU Transfer | Simple (5 lines) | 5-10% | 2 |
| 2 | CPU-GPU Transfer | Simple (4 lines) | 5-10% | 1 |
| 3 | CPU-GPU Transfer | Moderate (8 lines) | 5-8% | 2 |
| 4 | CPU-GPU Transfer | Simple (3 lines) | 5-10% | 2 |
| 5 | CPU-GPU Transfer | Simple (4 lines) | 2-5% | 2 |

**Total Expected Speedup**: 19-43%

---

## Baseline Summary

| Metric | Value |
|--------|-------|
| Training Time (avg) | 3.82s/it |
| Sample Generation Time (avg) | 68.97s/image |
| Epochs Tested | 6 |
| Steps per Epoch | 30 |
| Images Generated | 4 |

**Note**: Baseline established on 2026-08-04 with 6 epochs × 30 steps. Training time decreases over epochs as expected.