# HiDream Model - Device Check Cleanup Plan

## Overview
HiDream has 1 device check pattern in 1 file that can be consolidated.

**Note:** Simple `device.type == "mps"` comparisons are left as-is. Scheduler MPS float patterns are also left as-is — the current code is explicit and readable.

## Module 1: `extensions_built_in/diffusion_models/hidream/src/pipelines/hidream_image/pipeline_hidream_image.py`

### Change 1.1: Replace `torch.backends.mps.is_available()`
**Line:** 701
**Current:**
```python
if torch.backends.mps.is_available():
    latents = latents.to(latents_dtype)
```
**After:**
```python
from toolkit.util.torch_util import is_mps_available

if is_mps_available():
    latents = latents.to(latents_dtype)
```
**Why:** Consistent with other MPS checks; centralizes availability logic.
**Test:** Run HiDream sampling, verify latent dtype handling.

---

## Validation Order
1. Apply Change 1.1 → Test sampling

## Rollback Plan
If any change causes issues, revert the specific change and mark as "reverted" in this plan.
