# Chroma Model - Device Check Cleanup Plan

## Overview
Chroma has 2 device check patterns across 2 files that can be consolidated using `torch_util.py`.

**Note:** Simple `device.type == "mps"` comparisons are left as-is — the utility function adds overhead without simplifying the code. We only replace patterns that provide real value (complex logic, cross-platform safety).

## Module 1: `extensions_built_in/diffusion_models/chroma/pipeline.py`

### Change 1.1: Replace `device.type == "mps"` for text_dtype with `get_text_dtype()`
**Line:** 206
**Current:**
```python
text_dtype = torch.float32 if device.type == "mps" else torch.bfloat16
```
**After:**
```python
from toolkit.util.torch_util import get_text_dtype

text_dtype = get_text_dtype(device)
```
**Why:** Encapsulates MPS dtype fallback logic in one place; easy to modify if MPS gains bf16 support.
**Test:** Run Chroma sampling, verify text encoding works correctly.

### Change 1.2: Replace `torch.backends.mps.is_available()` 
**Line:** 305
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
**Test:** Run Chroma sampling, verify latent dtype handling.

---

## Module 2: `extensions_built_in/diffusion_models/chroma/src/layers.py`

### Change 2.1: Replace autocast device check with `get_autocast_context()`
**Line:** 290
**Current:**
```python
autocast_ctx = torch.autocast(device_type='cuda', enabled=False) if inputs.device.type == 'cuda' else contextlib.nullcontext()
```
**After:**
```python
from toolkit.util.torch_util import get_autocast_context

autocast_ctx = get_autocast_context(inputs.device, enabled=False)
```
**Why:** Encapsulates autocast+nullcontext logic; handles device type safely.
**Test:** Run Chroma training, verify autocast behavior unchanged.

---

## Validation Order
1. Apply Change 1.1 → Test sampling
2. Apply Change 1.2 → Test sampling
3. Apply Change 2.1 → Test training

## Rollback Plan
If any change causes issues, revert the specific change and mark as "reverted" in this plan.
