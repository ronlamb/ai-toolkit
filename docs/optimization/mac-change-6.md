# MPS Optimization Change #6: Missing MPS Logic in Chroma Training/Sampling Path

**Date**: 2026-06-08
**Target**: Chroma Model (excluding Chroma Radiance and Zeta Chroma)
**Scope**: Training and sampling path analysis for missing MPS-specific logic

---

## Executive Summary

This change identifies **12 locations** where CUDA/CPU logic exists but MPS is not handled. These range from hardcoded CUDA strings to CUDA-specific API calls that will fail or silently degrade on MPS.

---

## Issue #1: Hardcoded `torch.autocast("cuda")` in Chroma Layers

**File**: `extensions_built_in/diffusion_models/chroma/src/layers.py`
**Line**: 278

### Current Code
```python
class PatchEmbed(nn.Module):
    # ...existing code...
    def forward(self, inputs: Tensor) -> Tensor:
        # ...existing code...
        # Force all operations within this module to run in fp32.
        with torch.autocast("cuda", enabled=False):
            # ...existing code...
        return inputs.to(original_dtype)
```

### Problem
The string `"cuda"` is hardcoded. On MPS, `torch.autocast("cuda", ...)` will raise a `RuntimeError` because the device type doesn't match. This disables the autocast context entirely on MPS, potentially causing dtype mismatches.

### Recommended Fix
```python
# Detect device type from the input tensor
device_type = inputs.device.type  # "mps", "cuda", or "cpu"
with torch.autocast(device_type, enabled=False):
```

### Impact
**HIGH** - This code path runs during every forward pass in the PatchEmbed layer, which is called for every block in the Chroma model.

---

## Issue #2: Hardcoded `torch.autocast(device_type='cuda')` in Losses

**File**: `toolkit/losses.py`
**Line**: 45

### Current Code
```python
def get_gradient_penalty(critic, real, fake, device):
    with torch.autocast(device_type='cuda'):
        real = real.float()
        fake = fake.float()
        alpha = torch.rand(real.size(0), 1, 1, 1).to(device).float()
        # ... rest of function
```

### Problem
Hardcoded `device_type='cuda'` will fail on MPS. This function is used for GAN-style gradient penalties in training.

### Recommended Fix
```python
device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
with torch.autocast(device_type=device_type):
```

### Impact
**MEDIUM** - Only affects training modes that use gradient penalty (GAN/Style training).

---

## Issue #3: `torch.cuda.manual_seed()` Without MPS Fallback

**File**: `toolkit/train_tools.py`
**Line**: 112

### Current Code
```python
def get_noise_from_latents(latents):
    seed_list = get_seeds_from_latents(latents)
    noise = []
    for seed in seed_list:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        noise.append(torch.randn_like(latents[0]))
    return torch.stack(noise)
```

### Problem
`torch.cuda.manual_seed(seed)` will raise an error on MPS. There's no guard checking `torch.cuda.is_available()`.

### Recommended Fix
```python
def get_noise_from_latents(latents):
    seed_list = get_seeds_from_latents(latents)
    noise = []
    for seed in seed_list:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        # MPS uses the global CPU seed via torch.manual_seed()
        noise.append(torch.randn_like(latents[0]))
    return torch.stack(noise)
```

### Impact
**HIGH** - This function is called during noise generation for training steps.

---

## Issue #4: `torch.cuda.manual_seed()` in Stable Diffusion Model

**File**: `toolkit/stable_diffusion_model.py`
**Line**: 1451

### Current Code
```python
# In generate_images() / sampling loop
torch.manual_seed(gen_config.seed)
torch.cuda.manual_seed(gen_config.seed)
generator = torch.manual_seed(gen_config.seed)
```

### Problem
Same issue - no CUDA availability guard. This runs during every image generation.

### Recommended Fix
```python
torch.manual_seed(gen_config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(gen_config.seed)
generator = torch.manual_seed(gen_config.seed)
```

### Impact
**HIGH** - Runs during every image generation in the sampling loop.

---

## Issue #5: `torch.cuda.manual_seed()` in Base Model

**File**: `toolkit/models/base_model.py`
**Line**: 480

### Current Code
```python
# In generate_images() / sampling loop
torch.manual_seed(gen_config.seed)
torch.cuda.manual_seed(gen_config.seed)
generator = torch.manual_seed(gen_config.seed)
```

### Problem
Same pattern as Issue #4. Duplicate code in base_model.py.

### Recommended Fix
Same as Issue #4.

### Impact
**HIGH** - Runs during every image generation.

---

## Issue #6: `torch.cuda.empty_cache()` Without MPS Fallback

**File**: `toolkit/stable_diffusion_model.py`
**Line**: 1746

### Current Code
```python
torch.cuda.empty_cache()
```

### Problem
Will fail on MPS. Note: `toolkit/basic.py:flush()` already handles both CUDA and MPS cache clearing, but this direct call bypasses it.

### Recommended Fix
Replace with `flush()` from `toolkit.basic`, which already handles both:
```python
from toolkit.basic import flush
flush()
```

### Impact
**MEDIUM** - Called after validation image generation.

---

## Issue #7: `torch.cuda.empty_cache()` in Base Model

**File**: `toolkit/models/base_model.py`
**Line**: 672

### Current Code
```python
torch.cuda.empty_cache()
```

### Problem
Same as Issue #6.

### Recommended Fix
Replace with `flush()`.

### Impact
**MEDIUM** - Called after validation image generation.

---

## Issue #8: `torch.cuda.empty_cache()` in GenerateProcess

**File**: `jobs/process/GenerateProcess.py`
**Line**: 173

### Current Code
```python
torch.cuda.empty_cache()
```

### Problem
Same as Issue #6.

### Recommended Fix
Replace with `flush()`.

### Impact
**LOW** - Called during generation process cleanup.

---

## Issue #9: `torch.cuda.OutOfMemoryError` Exception Handling

**File**: `jobs/process/BaseSDTrainProcess.py`
**Lines**: 2250, 2253

### Current Code
```python
try:
    with self.accelerator.accumulate(self.modules_being_trained):
        loss_dict = self.hook_train_loop(batch_list)
except torch.cuda.OutOfMemoryError:
    did_oom = True
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        did_oom = True
    else:
        raise
```

### Problem
The `torch.cuda.OutOfMemoryError` exception is CUDA-specific. On MPS, OOM errors manifest as `RuntimeError` with different messages (e.g., "MPS out of memory" or "metal::device::allocateBuffer"). The string check `"CUDA out of memory"` won't match MPS error messages.

### Recommended Fix
```python
except torch.cuda.OutOfMemoryError:
    did_oom = True
except RuntimeError as e:
    error_str = str(e).lower()
    if "cuda out of memory" in error_str or "mps out of memory" in error_str or "metal" in error_str or "allocatebuffer" in error_str:
        did_oom = True
    else:
        raise
```

### Impact
**HIGH** - OOM handling is critical for training stability. Without this, MPS OOMs will crash training instead of gracefully skipping the batch.

---

## Issue #10: `torch.cuda.ipc_collect()` and `torch.cuda.synchronize()`

**File**: `jobs/process/BaseSDTrainProcess.py`
**Lines**: 2263, 2273

### Current Code
```python
if did_oom:
    # ...existing code...
    flush()
    torch.cuda.ipc_collect()
    # ...existing code...
else:
    # ...existing code...
if self.torch_profiler is not None:
    torch.cuda.synchronize()  # Make sure all CUDA ops are done
    self.torch_profiler.stop()
```

### Problem
- `torch.cuda.ipc_collect()` is CUDA-only and will fail on MPS
- `torch.cuda.synchronize()` is CUDA-only and will fail on MPS

### Recommended Fix
```python
# For ipc_collect:
if torch.cuda.is_available():
    torch.cuda.ipc_collect()

# For synchronize:
if torch.cuda.is_available():
    torch.cuda.synchronize()
elif torch.backends.mps.is_available():
    torch.mps.synchronize()
```

### Impact
**MEDIUM** - Affects OOM recovery and profiler functionality.

---

## Issue #11: `LearnableSNRGamma` Default Device

**File**: `toolkit/train_tools.py`
**Line**: 661

### Current Code
```python
class LearnableSNRGamma:
    def __init__(self, noise_scheduler: Union['DDPMScheduler'], device='cuda'):
        self.device = device
        # ...
        self.offset_1 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
```

### Problem
Default device is hardcoded to `'cuda'`. If instantiated without explicit device parameter, it will fail on MPS.

### Recommended Fix
```python
def __init__(self, noise_scheduler: Union['DDPMScheduler'], device=None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    self.device = device
```

### Impact
**LOW** - Only affects SNR gamma learning feature.

---

## Issue #12: Device String Parsing for CUDA

**File**: `toolkit/stable_diffusion_model.py`
**Line**: 141

### Current Code
```python
def __init__(self, ...):
    # ...existing code...
    self.device = str(device)
    if "cuda" in self.device and ":" not in self.device:
        self.device = f"{self.device}:0"
    self.device_torch = torch.device(device)
```

### Problem
Only handles CUDA device string formatting (adding `:0`). MPS doesn't use device indices in the same way, but this code won't break MPS - it just won't help. This is a **low-priority** issue.

### Recommended Fix
No change needed - this is harmless for MPS.

### Impact
**NEGLIGIBLE** - Cosmetic only.

---

## Already Correctly Handled (No Changes Needed)

The following patterns are **already guarded** and work correctly on MPS:

| File | Line | Pattern | Status |
|------|------|---------|--------|
| `toolkit/basic.py` | 12-16 | `flush()` with both CUDA and MPS cache clearing | ✅ Correct |
| `toolkit/stable_diffusion_model.py` | 1178 | `torch.cuda.get_rng_state() if torch.cuda.is_available() else None` | ✅ Correct |
| `toolkit/stable_diffusion_model.py` | 1750-1751 | `if cuda_rng_state is not None: torch.cuda.set_rng_state(cuda_rng_state)` | ✅ Correct |
| `toolkit/models/base_model.py` | 410 | `torch.cuda.get_rng_state() if torch.cuda.is_available() else None` | ✅ Correct |
| `toolkit/models/base_model.py` | 676-677 | `if cuda_rng_state is not None: torch.cuda.set_rng_state(cuda_rng_state)` | ✅ Correct |
| `jobs/process/BaseTrainProcess.py` | 37-38 | `if torch.cuda.is_available(): torch.cuda.manual_seed(...)` | ✅ Correct |
| `jobs/process/BaseSDTrainProcess.py` | 509-510 | `if torch.cuda.is_available(): torch.cuda.empty_cache()` | ✅ Correct |
| `extensions_built_in/diffusion_models/chroma/pipeline.py` | 125, 151, 206 | `device.type == "mps"` checks for latent_image_ids and text_ids | ✅ Correct |
| `extensions_built_in/diffusion_models/chroma/pipeline.py` | 308 | `torch.backends.mps.is_available()` for latents dtype fix | ✅ Correct |
| `extensions_built_in/diffusion_models/chroma/chroma_model.py` | 345-350 | `is_mps` check for txt_dtype and txt_ids allocation | ✅ Correct |
| `extensions_built_in/diffusion_models/chroma/src/math.py` | 17 | `q.is_cuda` for flash attention (correctly skips MPS) | ✅ Correct |

---

## Priority Order for Implementation

| Priority | Issue | Impact | Complexity |
|----------|-------|--------|------------|
| 1 | #9 OOM Exception Handling | HIGH | Low |
| 2 | #1 Chroma Layers Autocast | HIGH | Low |
| 3 | #3 train_tools.py manual_seed | HIGH | Low |
| 4 | #4 stable_diffusion_model.py manual_seed | HIGH | Low |
| 5 | #5 base_model.py manual_seed | HIGH | Low |
| 6 | #2 losses.py Autocast | MEDIUM | Low |
| 7 | #10 ipc_collect/synchronize | MEDIUM | Low |
| 8 | #6 stable_diffusion_model.py empty_cache | MEDIUM | Low |
| 9 | #7 base_model.py empty_cache | MEDIUM | Low |
| 10 | #8 GenerateProcess empty_cache | LOW | Low |
| 11 | #11 LearnableSNRGamma device | LOW | Low |
| 12 | #12 Device string parsing | NEGLIGIBLE | N/A |

---

## Test Protocol

For each fix:
1. Run 3 epochs × 30 steps training
2. Generate 2 images (4 steps each)
3. Verify no errors on MPS
4. Verify no regressions on CUDA (if testable)

---

**Last Updated**: 2026-06-08
