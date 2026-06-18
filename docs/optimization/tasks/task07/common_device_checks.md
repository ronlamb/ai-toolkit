
# Common Device Checks Analysis

Catalog of all distinct device-check patterns found in modules listed in `gpu_checks_change_7.txt`.

## Patterns That Can Be Added to a Utility Module

Code that is called more than once across multiple modules.

---

### Pattern 1: CUDA Availability Check

**Pattern:** `torch.cuda.is_available()`

**Total occurrences:** 27+ across toolkit/ and jobs/

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `toolkit/basic.py` | 1 | 12 |
| `toolkit/pixel_shuffle_encoder.py` | 2 | 65, 192 |
| `toolkit/layers.py` | 1 | 11 |
| `toolkit/llvae.py` | 3 | 10, 53, 119 |
| `toolkit/style.py` | 3 | 17, 77, 155 |
| `toolkit/train_tools.py` | 2 | 112, 665 |
| `toolkit/stable_diffusion_model.py` | 2 | 1178, 1451 |
| `toolkit/models/FakeVAE.py` | 1 | 31 |
| `toolkit/models/base_model.py` | 2 | 410, 480 |
| `toolkit/util/mask.py` | 1 | 262 |
| `toolkit/memory_management/manager_modules.py` | 1 | 109 |
| `jobs/process/BaseTrainProcess.py` | 1 | 37 |
| `jobs/process/TrainSDRescaleProcess.py` | 2 | 111, 128 |
| `jobs/process/BaseSDTrainProcess.py` | 2 | 2262, 2273 |

**Proposed utility function:**
```python
def is_cuda_available() -> bool:
    return torch.cuda.is_available()
```

---

### Pattern 2: MPS Device Type Check

**Pattern:** `device.type == "mps"`

**Total occurrences:** 8

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `extensions_built_in/diffusion_models/chroma/pipeline.py` | 3 | 125, 151, 206 |
| `extensions_built_in/diffusion_models/hidream/src/schedulers/fm_solvers_unipc.py` | 1 | 767 |
| `extensions_built_in/diffusion_models/hidream/src/schedulers/flash_flow_match.py` | 1 | 151 |
| `extensions_built_in/diffusion_models/hidream/src/models/transformers/transformer_hidream_image.py` | 1 | 310 |
| `extensions_built_in/diffusion_models/chroma/src/layers.py` | 1 | 280 |

**Proposed utility function:**
```python
def is_mps_device(device) -> bool:
    return getattr(device, 'type', str(device).split(':')[0]) == 'mps'
```

---

### Pattern 3: RNG State Save/Restore Pair

**Pattern:**
```python
rng_state = torch.get_rng_state()
cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
```

**Total occurrences:** 5

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `toolkit/stable_diffusion_model.py` | 1 | 1177-1178 |
| `toolkit/models/base_model.py` | 1 | 409-410 |
| `jobs/process/TrainSDRescaleProcess.py` | 1 | 110-111 |
| `toolkit/util/shuffle.py` | 1 | 21 |

**Proposed utility functions:**
```python
def save_rng_state() -> dict:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

def restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])
```

---

### Pattern 4: CUDA Seed Setting (after CPU seed)

**Pattern:**
```python
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
```

**Total occurrences:** 6+

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `jobs/process/BaseTrainProcess.py` | 1 | 36-38 |
| `jobs/process/TrainSDRescaleProcess.py` | 1 | 126-128 |
| `toolkit/train_tools.py` | 1 | 111-113 |
| `toolkit/stable_diffusion_model.py` | 1 | ~1450 |
| `toolkit/models/base_model.py` | 1 | ~479 |

**Proposed utility function:**
```python
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

---

### Pattern 5: Autocast Context with Device Check

**Pattern:**
```python
autocast_ctx = torch.autocast(device_type='cuda', ...) if device.type == 'cuda' else contextlib.nullcontext()
```

**Total occurrences:** 3

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `extensions_built_in/diffusion_models/chroma/src/layers.py` | 1 | 280 |
| `toolkit/losses.py` | 1 | 46-47 |

**Proposed utility function:**
```python
def get_autocast_context(device, enabled: bool = True, dtype=None):
    device_type = getattr(device, 'type', str(device).split(':')[0])
    if device_type == 'cuda':
        return torch.autocast(device_type='cuda', enabled=enabled, dtype=dtype)
    return contextlib.nullcontext()
```

---

### Pattern 6: Default Device Selection (CUDA or CPU)

**Pattern:** `torch.device("cuda" if torch.cuda.is_available() else "cpu")`

**Total occurrences:** 10+

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `toolkit/pixel_shuffle_encoder.py` | 2 | 65, 192 |
| `toolkit/layers.py` | 1 | 11 |
| `toolkit/llvae.py` | 3 | 10, 53, 119 |
| `toolkit/style.py` | 3 | 17, 77, 155 |
| `toolkit/models/FakeVAE.py` | 1 | 31 |
| `toolkit/util/mask.py` | 1 | 262 |

**Proposed utility function:**
```python
def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

**NOTE:** This pattern currently skips MPS. Code that uses this pattern may need MPS awareness. See per-model plans for analysis.

---

### Pattern 7: MPS-Specific Dtype Fallback

**Pattern:** `torch.float32 if device.type == "mps" else torch.bfloat16`

**Total occurrences:** 1

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `extensions_built_in/diffusion_models/chroma/pipeline.py` | 1 | 206 |

**Proposed utility function:**
```python
def get_text_dtype(device) -> torch.dtype:
    device_type = getattr(device, 'type', str(device).split(':')[0])
    return torch.float32 if device_type == 'mps' else torch.bfloat16
```

---

### Pattern 8: CUDA-Specific Device Type Check

**Pattern:** `device.type != "cuda"` or `device.type == "cuda"`

**Total occurrences:** 7

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `toolkit/memory_management/manager_modules.py` | 5 | 28, 158, 199, 334, 378 |
| `toolkit/optimizers/optimizer_utils.py` | 2 | 143, 144 |

**Proposed utility function:**
```python
def is_cuda_device(device) -> bool:
    return getattr(device, 'type', str(device).split(':')[0]) == 'cuda'
```

---

### Pattern 9: MPS Floating Point Check (Scheduler)

**Pattern:**
```python
if sample.device.type == "mps" and torch.is_floating_point(timestep):
    timestep = timestep.to(torch.float32)
```

**Total occurrences:** 2

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `extensions_built_in/diffusion_models/hidream/src/schedulers/fm_solvers_unipc.py` | 1 | 767 |
| `extensions_built_in/diffusion_models/hidream/src/schedulers/flash_flow_match.py` | 1 | 151 |

**Proposed utility function:**
```python
def mps_safe_float(tensor, device=None):
    """Ensure floating point tensor is float32 on MPS."""
    if device is None:
        device = tensor.device
    if is_mps_device(device) and torch.is_floating_point(tensor):
        return tensor.to(torch.float32)
    return tensor
```

---

### Pattern 10: CUDA Synchronize for Profiling

**Pattern:**
```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
elif torch.backends.mps.is_available():
    torch.mps.synchronize()
```

**Total occurrences:** 1

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `jobs/process/BaseSDTrainProcess.py` | 1 | 2273-2276 |

**Proposed utility function:**
```python
def synchronize(device=None) -> None:
    if device is None:
        device = get_default_device()
    device_type = getattr(device, 'type', str(device).split(':')[0])
    if device_type == 'cuda':
        torch.cuda.synchronize()
    elif device_type == 'mps':
        torch.mps.synchronize()
```

---

### Pattern 11: CUDA IPC Collect (OOM handling)

**Pattern:**
```python
if torch.cuda.is_available():
    torch.cuda.ipc_collect()
```

**Total occurrences:** 1

**Modules:**
| Module | Count | Lines |
|--------|-------|-------|
| `jobs/process/BaseSDTrainProcess.py` | 1 | 2262 |

**Proposed utility function:**
```python
def flush_cuda_ipc() -> None:
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
```

---

## Patterns That Can Be Moved Inside Another Function

Code called more than once within a single module.

---

### Pattern 12: MPS Latent Image IDs Handling (Chroma Pipeline)

**Location:** `extensions_built_in/diffusion_models/chroma/pipeline.py`

**Occurrences:** 2 (lines 125-130 and 151-156)

**Pattern:**
```python
if device.type == "mps":
    latent_image_ids = latent_image_ids.to(device)
else:
    latent_image_ids = latent_image_ids.to(device=device, dtype=dtype)
```

**Both are called immediately after `prepare_latent_image_ids`.**

**Recommendation:** Move this logic into `prepare_latent_image_ids` function, accepting `device` and `dtype` as parameters and handling the MPS case internally.

---

## Summary

| # | Pattern | Modules | Total Calls | Utility Function |
|---|---------|---------|-------------|-----------------|
| 1 | `torch.cuda.is_available()` | 14 | 27+ | `is_cuda_available()` |
| 2 | `device.type == "mps"` | 5 | 8 | `is_mps_device()` |
| 3 | RNG state save/restore | 4 | 5 | `save_rng_state()` / `restore_rng_state()` |
| 4 | CUDA seed after CPU seed | 5 | 6+ | `set_seed()` |
| 5 | Autocast context check | 2 | 3 | `get_autocast_context()` |
| 6 | Default device selection | 6 | 10+ | `get_default_device()` |
| 7 | MPS dtype fallback | 1 | 1 | `get_text_dtype()` |
| 8 | CUDA device type check | 2 | 7 | `is_cuda_device()` |
| 9 | MPS float safety | 2 | 2 | `mps_safe_float()` |
| 10 | CUDA/MPS synchronize | 1 | 1 | `synchronize()` |
| 11 | CUDA IPC collect | 1 | 1 | `flush_cuda_ipc()` |
| 12 | MPS latent IDs (intra-module) | 1 | 2 | Move into `prepare_latent_image_ids()` |
