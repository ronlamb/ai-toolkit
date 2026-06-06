# MPS Implementation Checklist

## Notes

- Use the **[Results Format](../.github/optimization-workflow.md#results-format)** for documenting detailed results
- This checklist tracks high-level progress of each change
- Reference `mac-results.md` for detailed metrics

---

## Baseline (M5 Max, 128GB Memory)
```
Training:   0%|          | 29/21300 [01:13<15:00:37,  2.54s/it, lr: 1.0e-04 loss: 9.570e-01]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:59<00:59, 59.52s/it]
            Generating Samples: 100%|##########| 2/2 [01:59<00:00, 59.73s/it]
```

**Training baseline**: 2.54s/it  
**Sampling baseline**: 59.73s/it

---

## Change #1: Cache Scheduler Weights on Device ✅ TODO
**Status**: ⏳ Planned

**Issue**: In `get_weights_for_timesteps()`, weight tensors are moved to device **every call** via `.to(device=..., dtype=...)`. With 30+ inference steps, this creates 30+ redundant transfers.

**Location**: `toolkit/samplers/custom_flowmatch_sampler.py`, lines ~77-78

**Current Code Pattern**:
```python
weights = self.linear_timesteps_weights.to(
    device=timesteps.device, dtype=timesteps.dtype
)[step_indices]
```

**Proposed Fix**: Cache weights on target device during initialization, detect device changes

**Expected Improvement**: 5-10% (HIGH)

**Checklist**:
- [ ] Code implemented
- [ ] User tested (results in `mac-results.md`)
- [ ] Code checked in to git
- [ ] Changes pushed to forked repo

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
