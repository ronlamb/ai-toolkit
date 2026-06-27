# CoPilot Instructions

## MPS (Apple Silicon) Optimization Guidelines

When working with MPS (Apple Silicon) performance in the AI Toolkit codebase, follow these guidelines:

### Core Philosophy

**On MPS, eliminate computation — don't cache state.** The winning optimizations (Tasks 1-2, scheduler improvements) eliminated actual redundant work (allocations, object creation, data transfers). The failing optimizations (Tasks 3-12) tried to cache device state, add conditionals, or create shared references — all of which add overhead on MPS.

### Key Reference Documents

- **[mac-performance-improvements-findings.md](./mac-performance-improvements-findings.md)** — Complete findings from all optimizations (accepted + rejected), cross-model applicability, implementation checklists
- **[z_image-tasks.md](./models/z_image/z_image-tasks.md)** — Z-Image task tracking with status table and rejected lessons
- **[chroma-mac-results.md](./models/chroma/chroma-mac-results.md)** — Chroma MPS test results
- **[z_mage-mac-results.md](./models/z_image/z_mage-mac-results.md)** — Z-Image MPS test results
- **[results.md](./results.md)** — Original Chroma optimization results (DGX)
- **[implement_torch_util.md](./implement_torch_util.md)** — torch_util.py consolidation plan

---

## MPS-Specific Rules (Non-Negotiable)

### ✅ Safe Patterns (Proven — Use These)

| Pattern | Why It Works | Examples |
|---------|--------------|----------|
| **Pipeline caching** | Eliminates expensive object recreation | Tasks 1 (Z-Image, Chroma) |
| **Tensor op optimization** | No Python object sharing; pure tensor math | Task 2 (Z-Image) |
| **`copy.deepcopy()` for cloning** | Fully independent copies | Data loader |
| **Simple scalar caching** | No complex object graph | int, float, bool, tensor |
| **Constant tensor caching** | Immutable after creation | ROPE omega, timestep freqs |
| **`to_device_if_needed()`** | One `.device` comparison; saves transfers | SDTrainer 50+ replacements |
| **`flush(garbage_collect=False)`** | MPS cache flush without GC blocking | Training loop |
| **`detach().clone().requires_grad_(True)`** | Avoids in-place sync trigger | Chroma model.py |
| **CUDA-only guards** | Cold-path checks; not on hot path | `torch.cuda.manual_seed()` |
| **Dtype selection at creation** | One allocation vs. allocate + convert | `torch.zeros(..., dtype=...)` |
| **`torch.searchsorted`** | Vectorized indexing vs. Python loop | `_get_step_indices()` |

### ❌ Unsafe Patterns (Avoid on MPS)

| Pattern | Why It Fails | Evidence |
|---------|--------------|----------|
| **Conditionals on hot path** | Adds overhead regardless of branch prediction | Tasks 9-11: 15% regression |
| **`hasattr`/`getattr` caching** | Object graph references prevent GC | Tasks 3-5: regressions |
| **Dirty flags on hot path** | `if self._flag:` adds overhead | Tasks 10-11 |
| **`__dict__.update()` for cloning** | Shared references fragment memory | Task 12: 15% regression |
| **Cached `inspect.signature()`** | Parameter refs interfere with GC | Task 9: gradual degradation |
| **`gc.collect()` in training loop** | Blocks MPS command queue | Task 8: 9.4% regression |
| **`torch.no_grad()` + `.requires_grad_(True)`** | Forces command buffer sync every step | Chroma model.py fix |
| **bfloat16 on MPS** | Not supported | Chroma pipeline.py |
| **float64 on MPS** | Not supported | Chroma math.py |
| **`torch.cuda.*` without check** | Crashes on MPS | Multiple files |
| **Unbounded caches** | Memory grows with unique inputs | Task 6: 25% regression |
| **Splitting `.to(device, dtype)`** | Can be slower than single call | Lesson 4 |

### ⚠️ Testing Rules

| Rule | Reason |
|------|--------|
| **Test over 8+ epochs** | Shared reference fragmentation shows as gradual degradation |
| **Clear `__pycache__` before each test** | Stale `.pyc` files cause false regressions |
| **Run identical test commands** | System load affects MPS timing |
| **Measure both training AND generation** | Some changes improve one but regress the other |
| **Accept threshold: >2% improvement** | Below 2% is within noise margin |
| **Revert threshold: >2% regression** | Any metric regressing >2% with no compensating gain |

---

## Code Categorization for MPS Optimization

### 1. Shared Toolkit Code (Affects All Models)

| File | Hot Path | Called |
|------|----------|--------|
| `toolkit/models/base_model.py` | `predict_noise()`, training loop | Every step |
| `toolkit/stable_diffusion_model.py` | `generate_images()` | Per generation |
| `toolkit/samplers/custom_flowmatch_sampler.py` | `get_weights_for_timesteps()`, `get_sigmas()` | Every step |
| `toolkit/losses.py` | `get_gradient_penalty()` | Every step |
| `toolkit/optimizer.py` | `get_optimizer()` | At startup |
| `toolkit/train_tools.py` | `get_noise_from_latents()` | Per batch |
| `toolkit/data_loader.py` | `_get_single_item()` | Per batch |
| `toolkit/util/torch_util.py` | Utility functions | Throughout |
| `jobs/process/BaseSDTrainProcess.py` | Training loop, `end_step_hook()` | Every step |
| `extensions_built_in/sd_trainer/SDTrainer.py` | `calculate_loss()`, `predict_noise()` | Every step |

### 2. Chroma-Specific Code

| File | Hot Path | Called |
|------|----------|--------|
| `extensions_built_in/diffusion_models/chroma/chroma_model.py` | `get_generation_pipeline()`, `get_noise_prediction()` | Per gen / every step |
| `extensions_built_in/diffusion_models/chroma/chroma_radiance_model.py` | Same as chroma_model | Per gen / every step |
| `extensions_built_in/diffusion_models/chroma/pipeline.py` | `prepare_latents()`, `prepare_latent_image_ids()` | Per generation |
| `extensions_built_in/diffusion_models/chroma/src/layers.py` | `timestep_embedding()` | Every step |
| `extensions_built_in/diffusion_models/chroma/src/math.py` | `rope()`, `apply_rope()` | Every step |
| `extensions_built_in/diffusion_models/chroma/src/model.py` | Forward pass (mod_vectors) | Every step |

### 3. Z-Image-Specific Code

| File | Hot Path | Called |
|------|----------|--------|
| `extensions_built_in/diffusion_models/z_image/z_image.py` | `get_noise_prediction()`, `generate_single_image()` | Every step / per image |

### 4. Unoptimized Models (Targets for Future Work)

| Model | Files to Check |
|-------|----------------|
| **flux2** | `extensions_built_in/diffusion_models/flux2/flux2_model.py`, `src/pipeline.py`, `src/sampling.py` |
| **ernie_image** | `extensions_built_in/diffusion_models/ernie_image/ernie_image.py` |
| **hidream** | `extensions_built_in/diffusion_models/hidream/hidream_model.py`, `src/models/`, `src/pipelines/`, `src/schedulers/` |

---

## Implementation Order (Inner to Outer)

### Phase 1: Critical Fixes (No Performance Risk)
These are correctness fixes that prevent crashes or undefined behavior on MPS:

1. Guard all `torch.cuda.*` calls with `if torch.cuda.is_available():`
2. Replace `torch.cuda.empty_cache()` with `flush()`
3. Replace bfloat16 with float32 for MPS
4. Replace float64 with float32 for MPS
5. Fix `torch.no_grad()` + `.requires_grad_(True)` → `detach().clone().requires_grad_(True)`

### Phase 2: High-Impact Optimizations
These eliminate redundant work (proven pattern from Tasks 1-2):

6. Add pipeline caching to `get_generation_pipeline()`
7. Optimize `get_noise_prediction()` tensor operations
8. Add `to_device_if_needed()` for device transfers
9. Replace Python loop indexing with `torch.searchsorted`

### Phase 3: Medium-Impact Optimizations
These cache constant values or reduce allocations:

10. Cache constant tensors (ROPE omega, timestep embeddings)
11. Add autocast guards for CUDA-only contexts
12. Remove redundant `.clone()` before `.to('cpu')`
13. Load state dicts directly to device

### Phase 4: Validation
14. Test over 8+ epochs (not just first epoch)
15. Clear `__pycache__` before each test
16. Measure both training and generation speed
17. Verify no gradual degradation over epochs

---

## For Each Change

1. **Propose the change** — Show diff/snippet, explain mechanism
2. **Request approval** — Iterate until user confirms
3. **Clear Python bytecode cache** (see below)
4. **Implement** — Apply approved change (≤20 lines)
5. **Clear Python bytecode cache again**
6. **User tests** — Run speed test (8 epochs × 30 steps, generate 2 images)
7. **Record results** — Update task file and findings document
8. **Next task** — Continue sequentially

### Clear Python Bytecode Cache (REQUIRED Before Each Test)

**Python caches compiled `.pyc` files in `__pycache__` directories. Stale cache from rejected/reverted changes will cause old code to run, making optimizations appear to regress when they don't (or vice versa).**

Before running any speed test, execute:

```bash
cd /Users/rlamb/src/ai-toolkit && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; find . -name '*.pyc' -delete 2>/dev/null
```

**When to clear cache:**
- After reverting any code change
- Before running a baseline comparison test
- Before running a post-optimization test
- If test results seem inconsistent with the code you see

**Signs of stale cache:**
- Test results don't match expected behavior from current code
- `git diff` shows no changes but performance differs from baseline
- Reverted changes still appear to affect timing

---

## Performance Baselines

### Z-Image (M5 Max, 128GB)

| Stage | Training s/it | Generation s/img |
|-------|---------------|------------------|
| Original baseline | 8.77s | 38.5s |
| After Task 1 (pipeline cache) | 7.48s | 36.3s |
| After Task 2 (tensor ops) | 7.26s | 37.0s |
| **Current optimized** | **7.26s** | **37.0s** |

### Chroma (M5 Max, 128GB)

| Stage | Training s/it | Generation s/img |
|-------|---------------|------------------|
| Original baseline | 11.93s | 55.0s |
| **Current optimized** | **~11.5s** | **~57.7s** |

---

## torch_util.py API Reference

All device-related operations should use `toolkit/util/torch_util.py` utilities:

| Function | Purpose |
|----------|---------|
| `get_device_type(device)` | Get device type string: 'cuda', 'mps', 'cpu' |
| `is_cuda_available()` | Check if CUDA is available |
| `is_mps_available()` | Check if MPS is available |
| `get_default_device()` | Best available device: cuda > mps > cpu |
| `is_cuda_device(device)` | Check if device is CUDA |
| `is_mps_device(device)` | Check if device is MPS |
| `get_autocast_context(device, enabled, dtype)` | Autocast for CUDA, nullcontext for others |
| `get_text_dtype(device)` | float32 for MPS, bfloat16 otherwise |
| `mps_safe_float(tensor, device)` | Ensure float32 on MPS |
| `synchronize(device)` | Synchronize CUDA or MPS |
| `flush_cache(garbage_collect)` | Flush CUDA/MPS caches + GC |
| `save_rng_state()` / `restore_rng_state()` | Save/restore CPU and CUDA RNG |
| `set_seed(seed)` | Set both CPU and CUDA random seeds |

---

## Applying Optimizations to New Models

When optimizing flux2, ernie_image, or hidream, follow the checklist in [mac-performance-improvements-findings.md](./mac-performance-improvements-findings.md) Part 7.

**Quick checklist:**
1. [ ] Pipeline caching in `get_generation_pipeline()`
2. [ ] bfloat16 → float32 for MPS
3. [ ] float64 → float32 for MPS
4. [ ] Guard `torch.cuda.*` calls
5. [ ] Replace `torch.cuda.empty_cache()` with `flush()`
6. [ ] Fix `torch.no_grad()` + `.requires_grad_(True)` pattern
7. [ ] Optimize `get_noise_prediction()` tensor ops
8. [ ] Add `to_device_if_needed()` for transfers
9. [ ] Cache constant tensors (ROPE, timestep embeddings)
10. [ ] Add autocast guards for CUDA-only contexts

---

## Agent Behavior

- Inspect tool outputs carefully; if unclear, ask user to check debug view
- Prefer action over hesitation — try changes, revert if needed
- For uncertain changes, ask user to **git commit** first
- **DO NOT commit or push** — User handles version control
- Use `.venv` Python (`.venv/bin/python`, `.venv/bin/pip`)
- Use `torch_util.py` helpers (`get_device_type()`, `is_mps_device()`, `flush()`, etc.)

---

**Last Updated**: 2026-06-26
