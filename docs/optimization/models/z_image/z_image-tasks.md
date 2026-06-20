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

---

## Status Tracking

| Task | Status | Before (ms) | After (ms) | Improvement | Notes |
|------|--------|-------------|------------|-------------|-------|
| 1    | Pending|             |            |             |       |
| 2    | Pending|             |            |             |       |
| 3    | Pending|             |            |             |       |
| 4    | Pending|             |            |             |       |
| 5    | Pending|             |            |             |       |
| 6    | Pending|             |            |             |       |
| 7    | Pending|             |            |             |       |
| 8    | Pending|             |            |             |       |

## Test Procedure
For each task:
1. Measure baseline performance (8 epochs × 30 steps, generate 2 images)
2. Apply the change
3. Run the same test
4. Record results in the table above
5. Update status to "Completed" or "Reverted" with notes