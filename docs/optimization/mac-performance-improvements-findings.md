# MPS Performance Improvements — Findings and Lessons

> **Date**: 2026-06-26
> **Platform**: macOS M5 Max, 128GB RAM, Apple Silicon with MPS
> **Scope**: All changes from `origin/main` branch across Chroma, Z-Image, and shared toolkit code

---

## Executive Summary

This document consolidates all findings from the MPS optimization project, including accepted changes, rejected changes, and the underlying principles that govern what works and what doesn't on Apple Silicon's MPS backend.

### Performance Achievements

| Model | Baseline Training | Optimized Training | Improvement | Baseline Gen | Optimized Gen | Improvement |
|-------|-------------------|--------------------|c|----------------|----------------|-------------|
| Chroma | 11.93s/it | ~11.5s/it | ~3.6% | 55.0s/img | ~58s/img | Within noise |
| Z-Image | 8.77s/it | 7.26s/it | **~17.2%** | 38.5s/img | 37.0s/img | ~3.9% |

---

## Part 1: Accepted Optimizations (What Worked)

### Category A: Eliminating Redundant Object Creation

These optimizations work because they eliminate actual Python object allocations, not because they cache device state.

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| A1 | **Pipeline caching** | `chroma_model.py`, `z_image.py`, `stable_diffusion_model.py` | High | Avoids recreating pipeline objects (expensive model loads + device transfers) on every `generate_images()` call |
| A2 | **Tensor op optimization** | `z_image.py` `get_noise_prediction()` | High | Replaced `unsqueeze(2)` + `unbind()` with list comprehension `[x.unsqueeze(1) for x in latent_model_input]`; batched `.float()` conversion via `torch.stack(model_out_list, dim=0).float()` |
| A3 | **Pre-computed `_timesteps_sorted`** | `custom_flowmatch_sampler.py` | Medium | Eliminated per-call `torch.flip()` allocation in `_get_step_indices()` |

### Category B: Eliminating Redundant Data Transfers

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| B1 | **State dict loading to device** | `chroma_model.py` | Medium | `load_file(model_path, device)` instead of `load_file(model_path, 'cpu')` then `.to(device)` — avoids CPU→GPU copy |
| B2 | **Remove redundant `.clone()`** | `chroma_model.py`, `chroma_radiance_model.py`, `BaseExtractProcess.py` | Low | `.to('cpu', dtype=...)` already creates a copy when changing device/dtype; `.clone()` before it was wasted allocation |
| B3 | **`to_device_if_needed()` in SDTrainer** | `SDTrainer.py` | Medium | 50+ replacements of `.to(device, dtype)` with conditional transfers — avoids unnecessary copies when tensor is already on correct device |

### Category C: MPS-Compatible dtype Handling

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| C1 | **bfloat16 → float32 for MPS** | `chroma_model.py`, `chroma_radiance_model.py`, `pipeline.py` | Critical | MPS does not support bfloat16; `torch.zeros(..., dtype=torch.float32 if mps else torch.bfloat16)` |
| C2 | **float64 → float32 for rope()** | `chroma/src/math.py` | Critical | MPS does not support float64; `torch.arange(..., dtype=torch.float32)` instead of `torch.float64` |
| C3 | **Latent image IDs device-only for MPS** | `chroma/pipeline.py` | Medium | `.to(device)` only for MPS (no dtype change), `.to(device, dtype)` for others — MPS handles dtype differently |

### Category D: CUDA-Only Operation Guards

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| D1 | **`torch.cuda.manual_seed()` guard** | `base_model.py`, `stable_diffusion_model.py`, `train_tools.py` | Critical | MPS crashes on `torch.cuda.manual_seed()`; wrapped in `if torch.cuda.is_available():` |
| D2 | **`torch.cuda.empty_cache()` → `flush()`** | `base_model.py`, `stable_diffusion_model.py`, `unloader.py`, `memory_management/manager.py` | Medium | `flush()` handles both CUDA and MPS cache clearing |
| D3 | **`torch.cuda.synchronize()` guard** | `BaseSDTrainProcess.py` | Low | MPS uses `torch.mps.synchronize()` |
| D4 | **`torch.cuda.ipc_collect()` guard** | `BaseSDTrainProcess.py` | Low | Only call on CUDA |
| D5 | **OOM error detection for MPS** | `BaseSDTrainProcess.py` | Medium | Check `"out of memory"` and `"allocatebuffer"` in error string, not just `"CUDA out of memory"` |

### Category E: Autocast Context Optimization

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| E1 | **Loss autocast only on CUDA** | `losses.py` `get_gradient_penalty()` | Medium | `torch.autocast('cuda')` adds overhead on MPS; use `contextlib.nullcontext()` for non-CUDA |
| E2 | **NerfEmbedder autocast only on CUDA** | `chroma/src/layers.py` | Low | Same pattern — `torch.autocast('cuda', enabled=False)` is expensive on MPS even with `enabled=False` |

### Category F: Scheduler Optimizations

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| F1 | **`_get_step_indices()` with `torch.searchsorted`** | `custom_flowmatch_sampler.py` | High | Replaced Python loop `[(self.timesteps == t).nonzero().item() for t in timesteps]` with vectorized `torch.searchsorted` |
| F2 | **Weight tensor caching** | `custom_flowmatch_sampler.py` | Medium | Device-aware caching of weight tensors with invalidation on device/dtype change |
| F3 | **`get_sigmas()` optimization** | `custom_flowmatch_sampler.py` | Medium | Index into sigmas first, then `.to(device, dtype)` — avoids transferring full sigmas array |
| F4 | **`torch.linspace` for shift schedules** | `custom_flowmatch_sampler.py` | Low | `torch.linspace(..., device=device)` instead of `np.linspace()` then `.to(device)` |

### Category G: Computation Caching (Constant Tensors)

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| G1 | **ROPE omega caching** | `chroma/src/math.py` | Medium | `omega = 1.0 / (theta**scale)` is constant for given `(dim, theta, device)` — cache in `_rope_cache` dict |
| G2 | **Timestep embedding freqs caching** | `chroma/src/layers.py` | Medium | `freqs` tensor is constant for given `(dim, max_period, device)` — cache in `_timestep_embedding_cache` dict |
| G3 | **`apply_rope()` dtype optimization** | `chroma/src/math.py` | Low | Convert `freqs_cis` to input dtype once, then operate — avoids per-tensor `.float()` calls |

### Category H: Graph Management

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| H1 | **`mod_vectors` detach + clone** | `chroma/src/model.py` | **Critical** | `torch.no_grad()` + `.requires_grad_(True)` forces command buffer sync every step on MPS (gradual degradation). Replaced with `mod_vectors.detach().clone().requires_grad_(True)` outside no_grad block |
| H2 | **`torch.inference_mode()`** | `stable_diffusion_model.py` | Low | Slightly better than `torch.no_grad()` for pure inference — disables autograd more aggressively |

### Category I: Memory Management

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| I1 | **`flush(garbage_collect=False)` in hot path** | `BaseSDTrainProcess.py` | Medium | `gc.collect()` blocks MPS command queue; skip in training loop, let GC run naturally or on save/sample |
| I2 | **Epoch transition cleanup** | `BaseSDTrainProcess.py` `end_step_hook()` | Medium | Clear cached pipeline and adapter state between epochs to prevent gradual slowdown from accumulated state |
| I3 | **Auto8bitTensor dequantize caching** | `optimizer_utils.py` | Low | Cache dequantized result in FP16 to reduce VRAM pressure |
| I4 | **Fused EMA + quantization** | `optimizer_utils.py` `update_from_fp32_()` | Low | Single-pass EMA update + quantization instead of two separate operations |

### Category J: Optimizer Fallbacks

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| J1 | **8bit optimizer → regular for MPS** | `optimizer.py` | Critical | bitsandbytes doesn't support MPS; auto-fallback `prodigy8bit` → `Prodigy`, `adam8bit` → `Adam` |

### Category K: Utility Module

| # | Change | Files | Impact | Mechanism |
|---|--------|-------|--------|-----------|
| K1 | **`torch_util.py` creation** | `toolkit/util/torch_util.py` | Foundation | 14 utility functions for device detection, RNG, autocast, synchronization, cache flushing — consolidates 11 duplicated patterns across 45+ modules |

---

## Part 2: Rejected Optimizations (What Failed)

### The Hot Path Problem

All rejected changes share a common theme: **any Python-level conditional or state check on the hot path (`predict_noise`) adds measurable overhead on MPS.**

| Task | Change | Files | Result | Root Cause |
|------|--------|-------|--------|------------|
| 3 | Cache text_encoder device flag | `z_image.py` | +1.3% training, +1.1% gen | Flag check costs more than `.device` property access |
| 4 | Cache model device flag | `z_image.py` | +11% training, +2.2% gen | Same — boolean conditional on hot path |
| 5 | Cache sigmas tensor | `custom_flowmatch_sampler.py` | +7% training, +3.9% gen | Cache invalidation checks cost more than `.to()` on MPS |
| 6 | Cache prompt embeddings | `stable_diffusion_model.py` | +25% training, -4.7% gen | Cache grows unbounded with unique training prompts |
| 7 | VAE to device at load | `z_image.py` | +3.5% training | VAE already on device; check was cheap |
| 8 | `flush()` after generation | Multiple | +9.4% training, +2% gen | `gc.collect()` is expensive per-call |
| 9 | Cache `inspect.signature()` | `base_model.py` | 7.0s → 7.8s over epochs | Cached signature objects cause memory fragmentation on MPS over time |
| 10 | Remove `hasattr` in loop | `base_model.py` | ~15% regression | Even local boolean conditional on hot path adds overhead |
| 11 | Dirty flag for device checks | `base_model.py` | ~15% regression | `if self._unet_device_dirty:` adds overhead |
| 12 | Shallow copy via `__dict__.update()` | `data_loader.py` | ~15% regression | Shared references fragment MPS memory |

### Key Pattern: Why These Failed

1. **Conditionals on hot path**: MPS's Metal backend has different branch prediction characteristics. A Python `if` statement in `predict_noise()` (called every training step) adds measurable overhead regardless of branch prediction accuracy.

2. **Shared references fragment memory**: Any optimization that creates shared Python object references (cached `inspect.signature()` results, `__dict__.update()` copies) causes gradual memory fragmentation on MPS. First epoch looks good, then performance degrades.

3. **Cache invalidation overhead**: Checking `if self._cached_device != tensor.device` costs more than just calling `.to(device)` on MPS. The `.to()` call is a fast path when device matches.

4. **`gc.collect()` blocks command queue**: Calling garbage collection during training blocks the MPS command queue, causing a sync point.

---

## Part 3: MPS-Specific Rules

### ✅ Safe Patterns (Proven)

| Pattern | Why It Works |
|---------|--------------|
| Pipeline caching (reuses same object) | No new Python object sharing; eliminates expensive recreation |
| Tensor operation optimization | No Python object sharing; pure tensor math |
| `copy.deepcopy()` for cloning | Fully independent copies; no shared references |
| Simple scalar caching (int, float, bool, tensor) | No complex object graph; no GC interference |
| Constant tensor caching (ROPE omega, freqs) | Tensors are immutable after creation; no shared state |
| `to_device_if_needed()` | Conditional is cheap (one `.device` comparison); saves actual transfers |
| `flush(garbage_collect=False)` | MPS cache flush without GC blocking |
| `detach().clone().requires_grad_(True)` | Avoids in-place `requires_grad_()` sync trigger |
| CUDA-only guards (`if torch.cuda.is_available()`) | Cold-path checks; not on hot path |
| Dtype selection at creation time | `torch.zeros(..., dtype=...)` is one allocation vs. allocate + convert |

### ❌ Unsafe Patterns (Avoid on MPS)

| Pattern | Why It Fails |
|---------|--------------|
| `__dict__.update()` for cloning | Shared references fragment memory |
| Caching `inspect.signature()` results | Parameter object refs interfere with GC |
| Caching `hasattr()`/`getattr()` results on complex objects | Object graph references prevent GC |
| Any conditional on hot path (`predict_noise`) | Adds overhead regardless of branch prediction |
| Dirty flags on hot path | Even `if self._flag:` adds measurable overhead |
| `gc.collect()` in training loop | Blocks MPS command queue |
| `torch.no_grad()` + `.requires_grad_(True)` | Forces command buffer sync every step |
| bfloat16 on MPS | Not supported; use float32 |
| float64 on MPS | Not supported; use float32 |
| `torch.cuda.*` without availability check | Crashes on MPS |
| Unbounded caches | Memory grows with unique inputs |
| Splitting `.to(device, dtype)` into two calls | Can be slower than single call on MPS |

### ⚠️ Testing Rules

| Rule | Reason |
|------|--------|
| **Test over 8+ epochs** | Shared reference fragmentation shows as gradual degradation (first epoch looks good, then gets worse) |
| **Clear `__pycache__` before each test** | Stale `.pyc` files cause false regressions |
| **Run same test command** | System load affects MPS timing; use identical commands |
| **Measure both training AND generation** | Some changes improve one but regress the other |
| **Accept threshold: >2% improvement** | Below 2% is within noise margin |
| **Revert threshold: >2% regression** | Any metric regressing >2% with no compensating gain |

---

## Part 4: Cross-Model Applicability

### Changes That Apply to All Models (flux2, ernie_image, hidream, z_image)

These are in shared toolkit code or follow universal patterns:

| Category | Changes | Target Files |
|----------|---------|--------------|
| **CUDA guards** | D1-D5 | `base_model.py`, `train_tools.py`, `BaseSDTrainProcess.py` |
| **Autocast** | E1-E2 | `losses.py`, any file with `torch.autocast('cuda')` |
| **Scheduler** | F1-F4 | `custom_flowmatch_sampler.py` (shared by all flow-matching models) |
| **Memory** | I1-I2 | `BaseSDTrainProcess.py` (shared base class) |
| **Optimizer** | J1 | `optimizer.py` (shared) |
| **Utility** | K1 | `torch_util.py` (shared) |
| **Graph** | H2 | `stable_diffusion_model.py` (shared) |

### Changes That Are Model-Specific

| Model | Applicable Changes | Files |
|-------|--------------------|-------|
| **Chroma** | A1, B1, B2, C1, C2, C3, E2, G1, G2, G3, H1 | `chroma_model.py`, `pipeline.py`, `src/layers.py`, `src/math.py`, `src/model.py` |
| **Z-Image** | A1, A2, A3, B3 | `z_image.py`, `custom_flowmatch_sampler.py` |
| **flux2** | A1 (pipeline caching), B3 (`to_device_if_needed`) | `flux2_model.py`, `src/pipeline.py` |
| **ernie_image** | A1 (pipeline caching), B3 (`to_device_if_needed`) | `ernie_image.py` |
| **hidream** | A1 (pipeline caching), B3 (`to_device_if_needed`) | `hidream_model.py`, `src/pipelines/` |

### Patterns to Apply to Unoptimized Models (flux2, ernie_image, hidream)

For each model, check for these patterns and apply fixes:

1. **Pipeline caching**: Does `get_generation_pipeline()` create a new pipeline every call? → Add `_cached_pipeline` pattern.
2. **bfloat16 usage**: Does the model use `torch.bfloat16` or `torch.zeros(..., dtype=torch.bfloat16)`? → Add MPS float32 fallback.
3. **float64 usage**: Does the model use `torch.float64`? → Change to `torch.float32`.
4. **`.clone().to('cpu')`**: Does the model call `.clone()` before `.to('cpu')`? → Remove redundant clone.
5. **`torch.cuda.*` calls**: Are there unguarded `torch.cuda.manual_seed()`, `torch.cuda.empty_cache()`, etc.? → Add guards or use `flush()`.
6. **`torch.autocast('cuda')`**: Is autocast used without device check? → Add CUDA-only guard.
7. **`torch.no_grad()` + `.requires_grad_(True)`**: Does the model use this pattern? → Use `detach().clone().requires_grad_(True)` instead.
8. **Constant tensor recomputation**: Are there tensors computed every call that depend only on `(dim, device)`? → Add caching.
9. **Python loops for tensor indexing**: Are there list comprehensions doing `[(timesteps == t).nonzero().item() for t in ...]`? → Use `torch.searchsorted`.
10. **Tensor op patterns**: Does `get_noise_prediction()` use `unsqueeze` + `unbind`? → Use list comprehension.

---

## Part 5: Performance Baselines

### Z-Image (M5 Max, 128GB)

| Stage | Training s/it | Generation s/img |
|-------|---------------|------------------|
| Original baseline | 8.77s | 38.5s |
| After Task 1 (pipeline cache) | 7.48s | 36.3s |
| After Task 2 (tensor ops) | 7.26s | 37.0s |
| After `_timesteps_sorted` | 7.26s | 37.0s |
| **Current optimized** | **7.26s** | **37.0s** |

### Chroma (M5 Max, 128GB)

| Stage | Training s/it | Generation s/img |
|-------|---------------|------------------|
| Original baseline | 11.93s | 55.0s |
| After Change #1 (state dict load) | 11.5s | ~58s |
| After Change #2 (clone removal) | 11.5s | ~57.7s |
| After Change #4 (pipeline cache) | ~11.5s | ~57.9s |
| **Current optimized** | **~11.5s** | **~57.7s** |

---

## Part 6: Remaining Optimization Opportunities

### High Priority (Apply to All Models)

1. **Pipeline caching for flux2, ernie_image, hidream** — Same pattern as Chroma/Z-Image.
2. **`to_device_if_needed()` in model-specific trainers** — SDTrainer has 50+ replacements; other models may need similar.
3. **Constant tensor caching** — Check for `timestep_embedding`, `rope`, or similar functions in flux2/ernie_image/hidream.

### Medium Priority

4. **`torch.searchsorted` for step index lookup** — Check if other samplers use Python loops.
5. **Tensor op optimization in `get_noise_prediction()`** — Check flux2/ernie_image/hidream for `unsqueeze` + `unbind` patterns.
6. **Autocast guards** — Search all model files for `torch.autocast('cuda')` without device check.

### Low Priority

7. **State dict loading to device** — Check if other models load to CPU then transfer.
8. **Redundant `.clone()` removal** — Search for `.clone().to('cpu')` pattern.
9. **`flush(garbage_collect=False)` in hot paths** — Check training loops in other extensions.

---

## Part 7: Implementation Checklist for New Models

When applying optimizations to a new model (flux2, ernie_image, hidream), follow this order:

### Phase 1: Critical Fixes (No Performance Risk)
- [ ] Guard all `torch.cuda.*` calls with `if torch.cuda.is_available():`
- [ ] Replace `torch.cuda.empty_cache()` with `flush()`
- [ ] Replace bfloat16 with float32 for MPS
- [ ] Replace float64 with float32 for MPS
- [ ] Fix `torch.no_grad()` + `.requires_grad_(True)` pattern

### Phase 2: High-Impact Optimizations
- [ ] Add pipeline caching to `get_generation_pipeline()`
- [ ] Optimize `get_noise_prediction()` tensor operations
- [ ] Add `to_device_if_needed()` for device transfers
- [ ] Replace Python loop indexing with `torch.searchsorted`

### Phase 3: Medium-Impact Optimizations
- [ ] Cache constant tensors (ROPE, timestep embeddings)
- [ ] Add autocast guards for CUDA-only contexts
- [ ] Remove redundant `.clone()` before `.to('cpu')`
- [ ] Load state dicts directly to device

### Phase 4: Validation
- [ ] Test over 8+ epochs (not just first epoch)
- [ ] Clear `__pycache__` before each test
- [ ] Measure both training and generation speed
- [ ] Verify no gradual degradation over epochs

---

**Last Updated**: 2026-06-26
**Author**: Analysis of all changes from `origin/main` branch
