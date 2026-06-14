# Implement torch_util.py — Device Check Consolidation Plan

## Overview

Consolidate 11 duplicated device-check patterns (CUDA/MPS/CPU) across 45+ modules into a single `toolkit/util/torch_util.py` utility module, then refactor all modules to use it.

**Goal**: Reduce code duplication, improve MPS support consistency, make device checks maintainable from one location.

**Target models**: chroma, flux2, ernie_image, hidream, z_image

**Platform**: macOS M5 Max, 128GB RAM, Apple Silicon with MPS

---

## Instructions So Far

### Environment Setup
- **Python**: 3.12.13 via pyenv, using `.venv` virtual environment
- **Run Python**: Always use `.venv/bin/python`, never the system Python
- **Install packages**: Use `.venv/bin/pip`, never `pip install` on the main environment
- **Testing**: `pytest` with `-v` flag for verbose output

### Workflow Rules (from copilot-instructions.md)
- Max 20 lines per function change — surgical improvements only, no rewrites
- Each change requires:
  1. Unit tests proving correctness
  2. Speed test: 3 epochs × 30 steps, generate 2 images
- **DO NOT commit or push to repo** — User handles version control
- Talk about code changes and ask user to verify before implementing

### Decision Rules
- **Proceed If**: ≤20 lines, >2% expected improvement (cumulative), passes tests, no API breaks
- **Revert If**: No measurable improvement, test failures, less maintainable, >20 lines

### Validation Against Baseline
After each change, user runs a test consisting of 3 epochs:
- Each epoch: 30 training steps + 2 sample images at 4 steps
- Use baseline times from `mac-results.md` to determine if performance degraded

**Baseline Times** (from `mac-results.md`):
- Training: ~11.93s/it
- Sampling: ~56.75s/it

---

## Current Status

### Completed Work
- ✅ Phase 1: Static import tracing from run.py
- ✅ Phase 1: Created per-model `_modules.txt` files (5 files: chroma, flux2, ernie_image, hidream, z_image)
- ✅ Phase 1: Created `gpu_checks_change_7.txt` (45 modules total)
- ✅ Phase 1: Updated `gpu_checks_leftover.txt` (50+ leftover modules not in target models)
- ✅ Phase 2: Created `common_device_checks.md` (11 patterns cataloged)
- ✅ Phase 3: Created `torch_util.py` utility module (14 functions)
- ✅ Phase 3: Created unit tests (44/44 passing)

### Performance Regression Investigation
- ⚠️ Initial implementation caused ~1s/it regression (~12.89s/it vs baseline 11.93s/it)
- ⚠️ All 5 refactored modules reverted (basic.py, losses.py, mask.py, pixel_shuffle_encoder.py, layers.py)
- ✅ PR #2 changes fully audited (34 files) — ruled out as root cause
- ✅ Dependencies verified (all match baseline)
- ⚠️ Root cause of regression still unknown — possibly measurement variance or MPS driver issue

### Current File States
| File | Status |
|------|--------|
| `toolkit/util/torch_util.py` | ✅ Complete (14 functions) |
| `tests/test_torch_util.py` | ✅ Complete (44 tests, all passing) |
| `toolkit/basic.py` | ⏳ Reverted (pending re-implementation) |
| `toolkit/losses.py` | ⏳ Reverted (pending re-implementation) |
| `toolkit/util/mask.py` | ⏳ Reverted (pending re-implementation) |
| `toolkit/pixel_shuffle_encoder.py` | ⏳ Reverted (pending re-implementation) |
| `toolkit/layers.py` | ⏳ Reverted (pending re-implementation) |

---

## torch_util.py API Reference

### Device Detection
| Function | Purpose |
|----------|---------|
| `get_device_type(device)` | Get device type string: 'cuda', 'mps', 'cpu' |
| `is_cuda_available()` | Check if CUDA is available |
| `is_mps_available()` | Check if MPS is available |
| `get_default_device()` | Best available device: cuda > mps > cpu |
| `is_cuda_device(device)` | Check if device is CUDA |
| `is_mps_device(device)` | Check if device is MPS |

### RNG Management
| Function | Purpose |
|----------|---------|
| `save_rng_state()` | Save CPU and CUDA RNG states |
| `restore_rng_state(state)` | Restore CPU and CUDA RNG states |
| `set_seed(seed)` | Set both CPU and CUDA random seeds |

### Context & Dtype
| Function | Purpose |
|----------|---------|
| `get_autocast_context(device, enabled, dtype)` | Autocast for CUDA, nullcontext for others |
| `get_text_dtype(device)` | float32 for MPS, bfloat16 otherwise |
| `mps_safe_float(tensor, device)` | Ensure float32 on MPS |

### Synchronization & Cleanup
| Function | Purpose |
|----------|---------|
| `synchronize(device)` | Synchronize CUDA or MPS |
| `flush_cuda_ipc()` | Flush CUDA IPC resources |
| `flush_cache(garbage_collect)` | Flush CUDA/MPS caches + GC |

---

## Implementation Plan

### Strategy
1. **Start with cold-path modules** (startup, config, one-time setup) — low risk, no hot-path impact
2. **Move to warm-path modules** (per-epoch, per-dataset) — medium risk
3. **End with hot-path modules** (per-step, per-batch) — highest risk, most careful validation

### Phase 1: Cold-Path Modules (Startup/Config)

These modules load once at startup. Changes here have zero impact on training/sampling speed.

#### 1.1 `toolkit/config_modules.py`
- **Pattern**: `torch.cuda.is_available()` checks
- **Action**: Replace with `is_cuda_available()` from torch_util
- **Risk**: None (startup only)

#### 1.2 `toolkit/embedding.py`
- **Pattern**: Device type checks
- **Action**: Replace with `get_device_type()` / `is_mps_device()`
- **Risk**: None (loaded once)

#### 1.3 `toolkit/dequantize.py`
- **Pattern**: CUDA availability checks
- **Action**: Replace with `is_cuda_available()`
- **Risk**: None (loaded once)

#### 1.4 `toolkit/ip_adapter.py`
- **Pattern**: Device checks for adapter loading
- **Action**: Replace with torch_util functions
- **Risk**: None (loaded once)

#### 1.5 `toolkit/image_utils.py`
- **Pattern**: Device checks for image processing
- **Action**: Replace with torch_util functions
- **Risk**: None (loaded once)

### Phase 2: Warm-Path Modules (Per-Epoch/Per-Dataset)

These modules run per epoch or per dataset iteration.

#### 2.1 `toolkit/dataloader_mixins.py`
- **Pattern**: Device checks in data loading
- **Action**: Replace with torch_util functions
- **Risk**: Low (not per-step)

#### 2.2 `toolkit/kohya_lora.py`
- **Pattern**: Device checks for LoRA loading
- **Action**: Replace with torch_util functions
- **Risk**: Low (per-epoch)

#### 2.3 `toolkit/kohya_model_util.py`
- **Pattern**: Device checks for model utilities
- **Action**: Replace with torch_util functions
- **Risk**: Low (per-epoch)

#### 2.4 `toolkit/network_mixins.py`
- **Pattern**: Device checks in network mixins
- **Action**: Replace with torch_util functions
- **Risk**: Low (per-epoch)

#### 2.5 `toolkit/lycoris_utils.py`
- **Pattern**: Device checks for lycoris
- **Action**: Replace with torch_util functions
- **Risk**: Low (per-epoch)

#### 2.6 `toolkit/lorm.py`
- **Pattern**: Device checks for LoRM
- **Action**: Replace with torch_util functions
- **Risk**: Low (per-epoch)

### Phase 3: Hot-Path Modules (Per-Step)

These modules run every training step. **Most careful validation required.**

#### 3.1 `toolkit/losses.py`
- **Pattern**: `get_autocast_context()` usage
- **Action**: Replace inline autocast check with `get_autocast_context()`
- **Risk**: **HIGH** — per-step execution
- **Validation**: 3 epochs × 30 steps + 2 samples after change

#### 3.2 `toolkit/basic.py`
- **Pattern**: `is_cuda_available()` checks
- **Action**: Replace with torch_util function
- **Risk**: **HIGH** — per-step execution
- **Validation**: 3 epochs × 30 steps + 2 samples after change

#### 3.3 `toolkit/util/mask.py`
- **Pattern**: Device checks for mask operations
- **Action**: Replace with torch_util functions
- **Risk**: **HIGH** — per-step execution
- **Validation**: 3 epochs × 30 steps + 2 samples after change

#### 3.4 `toolkit/pixel_shuffle_encoder.py`
- **Pattern**: `get_default_device()` usage
- **Action**: Replace inline device selection with `get_default_device()`
- **Risk**: **HIGH** — per-step execution
- **Validation**: 3 epochs × 30 steps + 2 samples after change

#### 3.5 `toolkit/layers.py`
- **Pattern**: Device checks in layer operations
- **Action**: Replace with torch_util functions
- **Risk**: **HIGH** — per-step execution
- **Validation**: 3 epochs × 30 steps + 2 samples after change

### Phase 4: Model-Specific Modules

#### 4.1 Chroma Model
- `extensions_built_in/diffusion_models/chroma/pipeline.py`
  - **Pattern**: `is_mps_device()`, `get_text_dtype()`, latent image IDs handling
  - **Action**: Replace inline MPS checks with torch_util functions
- `extensions_built_in/diffusion_models/chroma/src/layers.py`
  - **Pattern**: Autocast context, MPS device checks
  - **Action**: Replace with `get_autocast_context()`, `is_mps_device()`
- `extensions_built_in/diffusion_models/chroma/chroma_model.py`
  - **Pattern**: Device checks
  - **Action**: Replace with torch_util functions

#### 4.2 Flux2 Model
- `extensions_built_in/diffusion_models/flux2/flux2_model.py`
  - **Pattern**: Device checks
  - **Action**: Replace with torch_util functions

#### 4.3 Ernie Image Model
- `extensions_built_in/diffusion_models/ernie_image/ernie_image.py`
  - **Pattern**: Device checks
  - **Action**: Replace with torch_util functions

#### 4.4 HiDream Model
- `extensions_built_in/diffusion_models/hidream/hidream_model.py`
- `extensions_built_in/diffusion_models/hidream/src/models/transformers/transformer_hidream_image.py`
- `extensions_built_in/diffusion_models/hidream/src/schedulers/fm_solvers_unipc.py`
  - **Pattern**: `mps_safe_float()` for timestep handling
  - **Action**: Replace with `mps_safe_float()`
- `extensions_built_in/diffusion_models/hidream/src/schedulers/flash_flow_match.py`
  - **Pattern**: `mps_safe_float()` for timestep handling
  - **Action**: Replace with `mps_safe_float()`

#### 4.5 Z Image Model
- `extensions_built_in/diffusion_models/z_image/z_image.py`
  - **Pattern**: Device checks
  - **Action**: Replace with torch_util functions

### Phase 5: Memory Management Modules

#### 5.1 `toolkit/memory_management/manager.py`
- **Pattern**: Device state imports, CUDA checks
- **Action**: Replace with torch_util functions
- **Risk**: Medium (called during memory management)

#### 5.2 `toolkit/memory_management/manager_modules.py`
- **Pattern**: `is_cuda_device()` checks (5 occurrences)
- **Action**: Replace with `is_cuda_device()`
- **Risk**: Medium (called during memory management)

### Phase 6: Model Base Classes

#### 6.1 `toolkit/models/base_model.py`
- **Pattern**: `set_seed()`, `flush_cache()`
- **Action**: Replace with torch_util functions
- **Risk**: Medium (called during model initialization)

#### 6.2 `toolkit/models/FakeVAE.py`
- **Pattern**: `get_default_device()` usage
- **Action**: Replace with `get_default_device()`
- **Risk**: Low (VAE operations)

#### 6.3 `toolkit/llvae.py`
- **Pattern**: `get_default_device()` usage (3 occurrences)
- **Action**: Replace with `get_default_device()`
- **Risk**: Low (VAE operations)

### Phase 7: Jobs/Process Modules

#### 7.1 `jobs/process/BaseSDTrainProcess.py`
- **Pattern**: `synchronize()`, `flush_cuda_ipc()`
- **Action**: Replace with torch_util functions
- **Risk**: Medium (training loop)

#### 7.2 `jobs/process/TrainSDRescaleProcess.py`
- **Pattern**: `save_rng_state()`, `set_seed()`
- **Action**: Replace with torch_util functions
- **Risk**: Low (rescale process)

#### 7.3 `jobs/ExtensionJob.py`
- **Pattern**: Device checks
- **Action**: Replace with torch_util functions
- **Risk**: Low (job setup)

#### 7.4 `jobs/TrainJob.py`
- **Pattern**: Device checks
- **Action**: Replace with torch_util functions
- **Risk**: Low (job setup)

### Phase 8: Remaining Toolkit Modules

#### 8.1 `toolkit/style.py`
- **Pattern**: `get_default_device()` usage (3 occurrences)
- **Action**: Replace with `get_default_device()`
- **Risk**: Low (style operations)

#### 8.2 `toolkit/train_tools.py`
- **Pattern**: `set_seed()`, `is_cuda_available()`
- **Action**: Replace with torch_util functions
- **Risk**: Low (training utilities)

#### 8.3 `toolkit/stable_diffusion_model.py`
- **Pattern**: `save_rng_state()`, `set_seed()`, `flush_cache()`
- **Action**: Replace with torch_util functions
- **Risk**: Medium (model operations)

#### 8.4 `toolkit/control_generator.py`
- **Pattern**: Device checks
- **Action**: Replace with torch_util functions
- **Risk**: Low (control generation)

---

## Implementation Procedure

For each module in the plan above:

### Step 1: Analyze
1. Read the module file
2. Identify all device-check patterns
3. Map each pattern to the corresponding torch_util function
4. Count total occurrences

### Step 2: Propose Changes
1. List each change with:
   - Line number(s)
   - Current code snippet
   - Proposed replacement
   - Expected impact (none/low/medium/high)

### Step 3: User Review
1. Present the full list of changes for the module
2. Ask user: "Does this look correct? Any concerns?"
3. Wait for user approval before proceeding

### Step 4: Implement
1. Apply all changes to the module using `multi_replace_string_in_file`
2. Verify no syntax errors
3. Run unit tests for torch_util.py to ensure no regressions

### Step 5: Validate
1. Ask user to run validation test:
   - 3 epochs × 30 steps training
   - 2 sample images at 4 steps
2. Compare against baseline:
   - Training: ~11.93s/it
   - Sampling: ~56.75s/it
3. If performance degrades:
   - Check logic for issues
   - If unfixable, revert the change
   - Mark module as "skipped" in this document

### Step 6: Document Results
1. Update this document with:
   - Module name
   - Number of changes
   - Performance impact (if any)
   - Status: ✅ passed / ⚠️ degraded / ❌ reverted

---

## Performance Tracking

| Module | Changes | Training s/it | Sampling s/it | Status |
|--------|---------|---------------|---------------|--------|
| Baseline | - | 11.93 | 56.75 | - |
| *(to be filled as implementation progresses)* | | | | |

---

## Known Risks

### Performance Overhead
- **Previous regression**: ~1s/it slowdown observed during initial implementation
- **Root cause**: Unknown (PR #2 ruled out, dependencies match)
- **Mitigation**: 
  - Implement cold-path modules first (no hot-path impact)
  - Validate each hot-path module individually
  - Revert immediately if degradation detected

### Function Call Overhead
- **Concern**: Python function calls in hot paths may add overhead
- **Mitigation**: 
  - Keep torch_util functions simple (inline-friendly)
  - Consider `@inline` or macro-like patterns if needed
  - Profile with `cProfile` if overhead suspected

### MPS-Specific Issues
- **Concern**: MPS has different behavior than CUDA for certain operations
- **Mitigation**:
  - All torch_util functions are MPS-aware
  - Test on MPS hardware (M5 Max)
  - Use `mps_safe_float()` for floating point tensors

---

## Remaining TODO Items

- [ ] Phase 3: Create per-model refactoring plans (detailed plans for each model)
- [ ] Phase 3: Create integration tests (end-to-end validation)
- [ ] Implement Phase 1: Cold-path modules (5 modules)
- [ ] Implement Phase 2: Warm-path modules (6 modules)
- [ ] Implement Phase 3: Hot-path modules (5 modules)
- [ ] Implement Phase 4: Model-specific modules (5 models)
- [ ] Implement Phase 5: Memory management modules (2 modules)
- [ ] Implement Phase 6: Model base classes (3 modules)
- [ ] Implement Phase 7: Jobs/process modules (4 modules)
- [ ] Implement Phase 8: Remaining toolkit modules (4 modules)
- [ ] Final validation: full training run with all changes

---

## References

- [mac-change-7.md](./mac-change-7.md) — Original change request
- [common_device_checks.md](./common_device_checks.md) — 11 patterns catalog
- [gpu_checks_change_7.txt](./gpu_checks_change_7.txt) — 45 modules list
- [gpu_checks_leftover.txt](./gpu_checks_leftover.txt) — 50+ leftover modules
- [mac-results.md](./mac-results.md) — Baseline performance times
- [chroma_differences.md](./models/chroma/chroma_differences.md) — PR #2 impact analysis
- [optimization-workflow.md](./optimization-workflow.md) — Detailed protocols
- [skills/optimization-documentation/SKILL.md](./skills/optimization-documentation/SKILL.md) — Change templates
- [skills/optimization/SKILL.md](./skills/optimization/SKILL.md) — Platform detection
- [skills/cuda-optimization/SKILL.md](./skills/cuda-optimization/SKILL.md) — NVIDIA optimizations
- [skills/mps-optimization/SKILL.md](./skills/mps-optimization/SKILL.md) — Apple Silicon optimizations
- [skills/optimization-validate/SKILL.md](./skills/optimization-validate/SKILL.md) — Validation protocol
