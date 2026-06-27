# Z-Image MPS Optimization Tasks

This file contains the optimization tasks we will work on one by one, ordered by expected impact on MPS (Apple Silicon) performance.

## Task 1: Cache pipeline in get_generation_pipeline()
**File:** `extensions_built_in/diffusion_models/z_image/z_image.py`
**Impact:** High - Avoids recreating pipeline on each generation call
**Description:** 
- Currently `get_generation_pipeline()` creates a new ZImagePipeline instance every time it's called
- The pipeline should be cached and reused across multiple generation calls
- Similar to how `stable_diffusion_model.py` uses `_cached_pipeline`
**Changes:**
- Add `_cached_pipeline` attribute to ZImageModel
- Check for cached pipeline before creating new one
- Move pipeline to device only once

## Task 2: Optimize get_noise_prediction() tensor operations
**File:** `extensions_built_in/diffusion_models/z_image/z_image.py`
**Impact:** High - Reduces CPU↔GPU transfers and redundant operations
**Description:**
- `latent_model_input.unsqueeze(2)` followed by `.unbind(dim=0)` creates unnecessary intermediate tensors
- `torch.stack([t.float() for t in model_out_list], dim=0)` converts each tensor to float individually
- These operations cause extra memory allocations and device transfers on MPS
**Changes:**
- Avoid unbind/stack pattern if possible, or optimize the conversion
- Ensure dtype conversion happens on-device efficiently

## Task 3: Cache text_encoder device state in get_prompt_embeds()
**File:** `extensions_built_in/diffusion_models/z_image/z_image.py`
**Impact:** Medium - Reduces redundant device checks and transfers
**Description:**
- Currently checks `self.pipeline.text_encoder.device != self.device_torch` on every call
- If text encoder is on CPU (low_vram mode), this causes a transfer for every prompt encoding
**Changes:**
- Keep text encoder on device after first use if not in low_vram mode
- Or batch prompt encodings to minimize transfers

## Task 4: Optimize generate_single_image() model device check
**File:** `extensions_built_in/diffusion_models/z_image/z_image.py`
**Impact:** Medium - Reduces redundant device transfers
**Description:**
- Checks `self.model.device == torch.device("cpu")` and moves model on each call
- This check and potential transfer happens for every image generated
**Changes:**
- Cache model device state
- Only move model when necessary (e.g., after training completes)

## Task 5: Scheduler timestep weights caching (already partially implemented)
**File:** `toolkit/samplers/custom_flowmatch_sampler.py`
**Impact:** Medium - Weight tensors are now cached with device/dtype awareness
**Description:**
- The `get_weights_for_timesteps()` method already has caching logic
- Verify the caching is working correctly and covers all code paths
**Changes:**
- Review and potentially enhance the existing cache invalidation logic

## Task 6: Batch prompt encoding for multiple images
**File:** `toolkit/stable_diffusion_model.py` and Z-Image specific code
**Impact:** Medium - Reduces text encoder invocations
**Description:**
- When generating multiple images, prompts are encoded individually
- Text encoder should be invoked once for all unique prompts
**Changes:**
- Implement prompt caching similar to `stable_diffusion_model.py`'s `prompt_cache`

## Task 7: VAE device management
**File:** `extensions_built_in/diffusion_models/z_image/z_image.py`
**Impact:** Low-Medium - Ensures VAE and latents are on same device
**Description:**
- VAE may be on different device than latents during decoding
- Ensure VAE is moved to correct device before use
**Changes:**
- Add VAE device check before decoding latents

## Task 8: Memory management improvements
**File:** Multiple files
**Impact:** Low - Reduces memory pressure on MPS
**Description:**
- Add strategic `flush()` calls after major operations
- Ensure tensors are deleted when no longer needed
**Changes:**
- Review and add memory management calls

## Task 9: Cache inspect.signature() check at init time
**File:** `toolkit/models/base_model.py`
**Impact:** Medium - Eliminates Python reflection call on every training step
**Description:**
- `predict_noise()` calls `inspect.signature(self.get_noise_prediction).parameters` on every step
- This is a pure metadata check that never changes after model load
- Moving this to `__init__` eliminates per-step reflection overhead
**Changes:**
- In `BaseModel.__init__()`, cache three boolean flags from signature inspection
- Replace `inspect.signature()` call in `predict_noise()` with cached flag lookups

## Task 10: Remove hasattr check in scale_model_input loop
**File:** `toolkit/models/base_model.py`
**Impact:** Low-Medium - Eliminates per-batch-item hasattr call
**Description:**
- Inside `scale_model_input`, `hasattr(self.noise_scheduler, '_step_index')` is called per batch item
- This adds N× hasattr calls per training step (N = batch size)
**Changes:**
- Move hasattr check outside the loop or cache the result

## Task 11: Reduce conditionals in predict_noise hot path
**File:** `toolkit/models/base_model.py`
**Impact:** Low-Medium - Removes per-step device/dtype checks
**Description:**
- `self.unet.device != self.device_torch` and `self.unet.dtype != self.torch_dtype` checked every step
- These checks add conditionals to the hot path
**Changes:**
- Evaluate whether these checks can be moved to less frequent points

## Task 12: Optimize data loading / batch preparation
**File:** `toolkit/data_loader.py` and related
**Impact:** Medium - CPU-bound work that could overlap with GPU
**Description:**
- Data loading and batch preparation happen on CPU
- May be able to prefetch or overlap with GPU computation
**Changes:**
- Profile data loading path and identify bottlenecks

---

## Status Tracking

| Task | Status | Before (ms) | After (ms) | Improvement | Notes |
|------|--------|-------------|------------|-------------|-------|
| 1    | Completed | 8.77s/it, 38.5s/img | 7.48s/it, 36.3s/img | ~15% train, ~6% gen | Pipeline caching |
| 2    | Completed | 7.48s/it, 36.3s/img | 7.26s/it, 37.0s/img | ~3% train, flat gen | Tensor op optimization |
| 3    | Reverted | 7.48s/it | 7.58s/it | No improvement | Device check caching adds overhead |
| 4    | Reverted | 7.48s/it | 8.31s/it | Regression | Model device check adds overhead |
| 5    | Reverted | 7.48s/it | 8.00s/it | Regression | Cache invalidation adds overhead |
| 6    | Reverted | 7.48s/it | 9.35s/it | Regression | Cache grows unbounded |
| 7    | Reverted | 7.48s/it | 7.74s/it | Regression | VAE device check adds overhead |
| 8    | Reverted | 7.48s/it | 8.18s/it | Regression | Flush() calls add sync overhead |
| 9    | Reverted | 7.26s/it | 7.0→7.8s/it | Regression | Cached signature objects cause memory fragmentation on MPS over time |
| 10   | Reverted | 7.26s/it | 8.3-8.6s/it | ~15% regression | Even local boolean conditional on hot path adds overhead on MPS |
| 11   | Reverted | 7.26s/it | 8.4-8.6s/it | ~15% regression | Dirty flag conditional on hot path adds overhead, same as Tasks 9/10 |
| 12   | Reverted | 7.26s/it | 8.4-8.6s/it | ~15% regression | Shallow copy via __dict__.update() creates shared references that fragment MPS memory |

## Rejected Optimization Lessons (MPS-Specific)

1. **Device-state caching adds overhead** — Flag checks cost more than `.device` property access on MPS
2. **Operation fusion can regress** — Fused ops trigger different Metal kernel paths with worse fragmentation
3. **`.view()` can force copies** — `.unsqueeze()` is safer (always zero-copy)
4. **Do not remove `.to()` calls on MPS** — Splitting conversions can be faster
5. **Pre-computing allocations works** — Proven pattern (Tasks 1-2)
6. **Micro-benchmarks don't predict real-world MPS performance** — `.reshape()` showed 35-48% speedup in isolation but caused 11% regression in full training
7. **Any conditional on hot path adds overhead** — `hasattr`, `getattr`, `if is_flow_matching` all caused regressions
8. **Stale .pyc cache causes false regressions** — Always clear `__pycache__` before testing
9. **Cached signature objects fragment MPS memory** — `inspect.signature()` results held as instance attributes cause gradual slowdown (7.0s → 7.8s over 8 epochs) due to Parameter object references interfering with GC over time
10. **Dirty flag conditionals add overhead** — `if self._unet_device_dirty:` on hot path caused 7.26s → 8.4-8.6s regression (~15%), confirming any Python-level conditional in `predict_noise` is too expensive on MPS
11. **Shallow copies fragment MPS memory** — `__dict__.update()` on FileItemDTO creates shared references that cause gradual slowdown (7.79s → 8.6s over epochs), same pattern as cached signature objects

## Test Procedure
For each task:
1. **Clear Python bytecode cache:** `find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; find . -name '*.pyc' -delete 2>/dev/null`
2. Measure baseline performance (8 epochs × 30 steps, generate 2 images)
3. Apply the change
4. **Clear Python bytecode cache again**
5. Run the same test
6. Record results in the table above
7. Update status to "Completed" or "Reverted" with notes