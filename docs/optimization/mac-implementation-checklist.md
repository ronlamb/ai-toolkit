# MPS Implementation Checklist

## Notes

- Use the **[Results Format](../.github/optimization-workflow.md#results-format)** for documenting detailed results
- This checklist tracks high-level progress of each change
- Reference `mac-results.md` for detailed metrics

---

## Baseline (M5 Max, 128GB Memory)
```
Training:   0%|          | 29/21300 [06:10<15:00:37,  12.34/it, lr: 1.0e-04 loss: 9.570e-01]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:59<00:59, 59.52s/it]
            Generating Samples: 100%|##########| 2/2 [01:59<00:00, 59.73s/it]
```

**Training baseline**: 12.34s/it  
**Sampling baseline**: 59.73s/it

---

## Change #1: Cache Scheduler Weights on Device ⚠️ REVERTING
**Status**: ⚠️ Reverting - Performance degraded across epochs (12.35s → 12.38s → 12.43s)

**Issue**: Cache implementation has issues with device string comparison and cache invalidation causing performance degradation instead of improvement.

**Root Cause**: Device string inconsistency (`mps` vs `mps:0`) and potential cache invalidation each iteration.

**Location**: `extensions_built_in/sd_trainer/SDTrainer.py`, line 820; `toolkit/samplers/custom_flowmatch_sampler.py`

**Changes Made**:
1. Removed `.to(loss.device, dtype=loss.dtype)` call on cached scheduler weights
2. Added caching infrastructure in `CustomFlowMatchEulerDiscreteScheduler`
3. Added explanatory comments

**Expected Improvement**: 5-10% (HIGH) - eliminates redundant device transfers and memory fragmentation

**Actual Result**: Performance degraded across epochs instead of stabilizing

**Memory Impact**: Negligible - cached weights already allocated, no new allocations

**Analysis**: See `change1-analysis.md` for detailed analysis of root causes and recommended fixes

**Verification Protocol**:
- Run 3 epochs × 30 steps test
- Expected: Training time stabilizes or improves (target: ≤11.70s/it)
- Check MPS memory doesn't accumulate across epochs

**Test Results**: ❌ Performance degraded (12.35s → 12.38s → 12.43s) instead of stabilizing

**Status**: ⚠️ REVERTING - Cache implementation has issues with device string comparison and cache invalidation

**Analysis**: See `change1-analysis.md` for detailed analysis of root causes and recommended fixes

**Checklist**:
- [x] Code implemented
- [x] Syntax validated (no errors)
- [x] User tested (results show degradation - reverting)
- [ ] Debug logging added to identify cache hit/miss patterns
- [ ] Device string consistency verified

**Notes**: This fixes the existing caching implementation that was being defeated by the `.to()` call. The caching logic in `custom_flowmatch_sampler.py` was correct, but the `.to()` call in SDTrainer was creating new tensors each iteration.

---

## Change #2: Fix Text IDs CPU Allocation Overhead ✅ TODO
**Status**: ⏳ Planned

**Issue**: Text IDs created on CPU then moved to MPS:
```python
if is_mps:
    text_ids = torch.zeros(bs, ..., dtype=txt_dtype).to(device)  # CPU alloc then move
else:
    text_ids = torch.zeros(bs, ..., device=device, dtype=txt_dtype)  # Direct to device
```

**Location**: `extensions_built_in/diffusion_models/chroma/pipeline.py`, lines ~209-215

**Proposed Fix**: Use direct device allocation for MPS (remove `.to(device)` call)

**Expected Improvement**: 1-2% (LOW-MEDIUM)

**Checklist**:
- [ ] Code implemented
- [ ] User tested (results in `mac-results.md`)
- [ ] Code checked in to git
- [ ] Changes pushed to forked repo

---

## Change #3: Remove Duplicate Latent Image IDs Creation ✅ TODO
**Status**: ⏳ Planned

**Issue**: `prepare_latent_image_ids()` is called **twice identically** in `prepare_latents()`:
- Lines ~118-120 (when latents provided)
- Lines ~144-146 (when creating new latents)

**Location**: `extensions_built_in/diffusion_models/chroma/pipeline.py`, lines ~118-146

**Proposed Fix**: Create latent_image_ids once and reuse

**Expected Improvement**: 2-3% (MEDIUM-HIGH)

**Checklist**:
- [ ] Code implemented
- [ ] User tested (results in `mac-results.md`)
- [ ] Code checked in to git
- [ ] Changes pushed to forked repo

---

## Change #4: Remove Redundant Component Device Transfers ✅ TODO
**Status**: ⏳ Planned

**Issue**: After pipeline creation, components are moved to device multiple times:
- Line ~251: `pipe.transformer.to(self.device_torch)`
- Lines ~255, 258, 261: Additional `.to()` calls on same components

**Location**: `extensions_built_in/diffusion_models/chroma/chroma_model.py`, lines ~251-261

**Proposed Fix**: Remove duplicate `.to()` calls, keep only one per component

**Expected Improvement**: 3-5% (MEDIUM)

**Checklist**:
- [ ] Code implemented
- [ ] User tested (results in `mac-results.md`)
- [ ] Code checked in to git
- [ ] Changes pushed to forked repo

---

## Change #5: Cache Scheduler Tensors with Device Awareness ✅ TODO
**Status**: ⏳ Planned

**Issue**: Scheduler tensors created on CPU with no device awareness:
```python
timesteps = torch.linspace(1000, 1, num_timesteps, device='cpu')  # Line ~55
self.linear_timesteps_weights = bsmntw_weighing  # Created on CPU
```

**Location**: `toolkit/samplers/custom_flowmatch_sampler.py`, lines ~55-59

**Proposed Fix**: Store device info and create tensors on correct device during initialization

**Expected Improvement**: 2-4% (MEDIUM)

**Checklist**:
- [ ] Code implemented
- [ ] User tested (results in `mac-results.md`)
- [ ] Code checked in to git
- [ ] Changes pushed to forked repo

---

## Summary Table

| Change | Status | Location | Expected Impact | Type |
|--------|--------|----------|-----------------|------|
| #1 | ⏳ Planned | custom_flowmatch_sampler.py:77 | 5-10% | Scheduler caching |
| #2 | ⏳ Planned | pipeline.py:209-215 | 1-2% | MPS-specific |
| #3 | ⏳ Planned | pipeline.py:118-146 | 2-3% | General |
| #4 | ⏳ Planned | chroma_model.py:251-261 | 3-5% | MPS-specific |
| #5 | ⏳ Planned | custom_flowmatch_sampler.py:55-60 | 2-4% | MPS-specific |

**Cumulative Expected Improvement**: 13-24%

---

## Test Protocol

Run **3 epochs × 30 steps**, generate **2 images**

### Metrics to Collect
1. **Training time per iteration**: `X.XXs/it` from progress bar
2. **Sample generation time**: Time per image from "Generating Samples" progress

### Baseline Results
```
Training:   2.54s/it (current baseline)
Samples:    59.73s/it (current baseline)
```

### Post-Change Validation
For each change, verify:
- [ ] Improvement ≥2% (cumulative target 13-24%)
- [ ] Unit tests pass
- [ ] No API breaks
- [ ] Code maintainable

---

## Notes

- User tests manually after each implementation
- Check in and push to forked repo before next change
- Update `mac-results.md` with detailed metrics after each test
- If improvement <2%, consider reverting or monitoring
