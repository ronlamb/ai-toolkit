# Flux2 Model - Device Check Cleanup Plan

## Overview
Flux2 has 3 CPU device checks that can use `is_cpu_device()` from `torch_util.py`.

## Module 1: `extensions_built_in/diffusion_models/flux2/flux2_model.py`

### Change 1.1: Replace CPU device checks for VAE
**Lines:** 412, 529, 546
**Current:**
```python
if self.vae.device == torch.device("cpu"):
    self.vae.to(self.device_torch)
```
**After:**
```python
from toolkit.util.torch_util import is_cpu_device

if is_cpu_device(self.vae.device):
    self.vae.to(self.device_torch)
```
**Test:** Run Flux2 image encoding/decoding, verify VAE device handling.

---

## Validation Order
1. Apply Change 1.1 (all 3 occurrences) → Test encoding/decoding

## Rollback Plan
If any change causes issues, revert the specific change and mark as "reverted" in this plan.
