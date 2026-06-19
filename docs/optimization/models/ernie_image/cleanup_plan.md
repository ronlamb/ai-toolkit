# Ernie Image Model - Device Check Cleanup Plan

## Overview
Ernie Image has 5 CPU device checks that can use `is_cpu_device()` from `torch_util.py`.

## Module 1: `extensions_built_in/diffusion_models/ernie_image/ernie_image.py`

### Change 1.1: Replace CPU device checks for VAE
**Lines:** 211, 241
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
**Test:** Run Ernie Image encoding/decoding, verify VAE device handling.

### Change 1.2: Replace CPU device checks for model
**Lines:** 269, 296
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
**Test:** Run Ernie Image generation, verify model device handling.

### Change 1.3: Replace CPU device check for text encoder
**Line:** 317
**Current:**
```python
if self.pipeline.text_encoder.device == torch.device("cpu"):
    self.pipeline.text_encoder.to(self.device_torch)
```
**After:**
```python
from toolkit.util.torch_util import is_cpu_device

if is_cpu_device(self.pipeline.text_encoder.device):
    self.pipeline.text_encoder.to(self.device_torch)
```
**Test:** Run Ernie Image prompt embedding, verify text encoder device handling.

---

## Validation Order
1. Apply Change 1.1 → Test encoding/decoding
2. Apply Change 1.2 → Test generation
3. Apply Change 1.3 → Test prompt embedding

## Rollback Plan
If any change causes issues, revert the specific change and mark as "reverted" in this plan.
