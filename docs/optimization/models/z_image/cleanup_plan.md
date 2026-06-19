# Z Image Model - Device Check Cleanup Plan

## Overview
Z Image has 2 CPU device checks that can use `is_cpu_device()` from `torch_util.py`.

## Module 1: `extensions_built_in/diffusion_models/z_image/z_image.py`

### Change 1.1: Replace CPU device checks for model
**Lines:** 303, 329
**Current:**
```python
if self.model.device == torch.device("cpu"):
    self.model.to(self.device_torch)
```
**After:**
```python
from toolkit.util.torch_util import is_cpu_device

if is_cpu_device(self.model.device):
    self.model.to(self.device_torch)
```
**Test:** Run Z Image generation, verify model device handling.

---

## Validation Order
1. Apply Change 1.1 → Test generation

## Rollback Plan
If any change causes issues, revert the specific change and mark as "reverted" in this plan.
