# Z-Image MPS Optimization — Round 2 Instructions

## Key Lessons from Round 1

**What worked (eliminated redundant work):**
- Task 1: Pipeline caching — avoided recreating ZImagePipeline objects
- Task 2: Tensor op optimization — avoided intermediate 5D tensor + batched dtype conversion

**What failed (device-state caching adds overhead on MPS):**
- Device flags (Tasks 3, 4): Flag checks cost more than `.device` property access
- Tensor caching (Task 5): Cache invalidation checks cost more than `.to()` on MPS
- Prompt caching (Task 6): Unbounded cache grew with unique training prompts
- VAE pre-positioning (Task 7): VAE already on device; check was cheap
- Memory flush (Task 8): `gc.collect()` is expensive per-call

**Rule for Round 2:** Focus on **eliminating computation or allocations**, not caching device state.

## Code Files

| File | Hot Path | Called |
|------|----------|--------|
| `extensions_built_in/diffusion_models/z_image/z_image.py` | `get_noise_prediction()`, `generate_single_image()` | Every training step / per image |
| `toolkit/samplers/custom_flowmatch_sampler.py` | `get_weights_for_timesteps()`, `get_sigmas()`, `_get_step_indices()` | Every training step / denoising step |
| `toolkit/models/base_model.py` | Training loop, `encode_images()`, `decode_latents()` | Every epoch / per image |

## New Task Options

### Option A: Pre-compute timestep normalization in `get_noise_prediction()`
**File:** `z_image.py`
**Idea:** `(1000 - timestep) / 1000` is computed every call. If timestep values are reused (e.g., same schedule), pre-normalize once.
**Risk:** Low — arithmetic only, no state caching.

### Option B: Fuse squeeze + negate in `get_noise_prediction()`
**File:** `z_image.py`
**Idea:** `noise_pred = noise_pred.squeeze(2)` then `noise_pred = -noise_pred` creates two temporaries. Try `noise_pred = -noise_pred.squeeze(2)` or `torch.neg()` in-place.
**Risk:** Very low — single-line change.

### Option C: Optimize `_get_step_indices()` device/dtype conversion
**File:** `custom_flowmatch_sampler.py`
**Idea:** `timesteps.to(device=base.device, dtype=base.dtype)` is called every invocation by both `get_weights_for_timesteps()` and `get_sigmas()`. If timesteps are already on the right device/dtype, skip the `.to()`.
**Risk:** Low — conditional check, no persistent state.

### Option D: Eliminate redundant `unsqueeze` in denoising loop
**File:** `custom_flowmatch_sampler.py` — `get_sigmas()`
**Idea:** `while len(sigmas.shape) < n_dim: sigmas = sigmas.unsqueeze(-1)` runs every call. Pre-compute target shape or use `view()` instead.
**Risk:** Low — shape manipulation only.

### Option E: Move text encoder device check outside hot path
**File:** `z_image.py` — `get_prompt_embeds()`
**Idea:** The text encoder device check happens every prompt encoding. During training, prompts change per batch but the device doesn't. Move check to happen once per epoch or after device state changes.
**Risk:** Medium — requires understanding when device changes occur.

### Option F: Reduce Python overhead in training loop
**File:** `toolkit/models/base_model.py`
**Idea:** The training loop has many Python-level checks per step (adapter state, control images, etc.). Profile to find the hottest Python code and minimize per-step work.
**Risk:** Medium — requires profiling first.

## Test Protocol

Run for **8 epochs × 30 steps, generate 2 images per epoch**. Record both:

| Metric | Baseline | After | Improvement |
|--------|----------|-------|-------------|
| Avg training s/it | 7.48s | X.XXs | X% |
| Avg gen time/image | 36.3s | XX.Xs | X% |

**Accept if:** Both metrics improve or one improves significantly without the other regressing >2%.
**Revert if:** Either metric regresses >2% with no compensating gain.

## Workflow

1. Pick a task option (A–F) or propose a new one
2. **Plan the change** — show the diff/snippet, get user approval
3. **Implement** — apply the change (≤20 lines)
4. **User tests** — run speed test
5. **Record results** — update `z_mage-mac-results.md`
6. **Next task** — continue sequentially

## Constraints
- Max **20 modified lines** per change
- No rewrites; only surgical optimizations
- Use `.venv` Python (`.venv/bin/python`)
- Use `torch_util.py` helpers (`get_device_type()`, `is_mps_device()`, `flush()`, etc.)
- **DO NOT commit or push** — user handles version control
