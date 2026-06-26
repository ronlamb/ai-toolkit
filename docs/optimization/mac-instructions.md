# CoPilot Instructions

## MPS (Apple Silicon) Optimization Guidelines

When working with MPS (Apple Silicon) performance in the AI Toolkit codebase, follow these guidelines:

### Code Categorization for MPS Optimization

#### 1. Main Pipeline Code
- `toolkit/stable_diffusion_model.py` - `generate_images()` method (line ~1200)
- `jobs/process/GenerateProcess.py` - Main generation orchestration
- `toolkit/models/base_model.py` - Base model class with pipeline management

#### 2. General Image Creation Code
- `toolkit/sampler.py` - Sampler utilities
- `toolkit/pipelines.py` - Custom pipeline implementations

#### 3. Flux Specific Code
- `extensions_built_in/diffusion_models/flux2/` - FLUX.2 model implementation
- `extensions_built_in/diffusion_models/flux_kontext/` - FLUX Kontext
- `toolkit/samplers/custom_flowmatch_sampler.py` - Custom flow match scheduler

#### 4. Chroma Specific Code
- `extensions_built_in/diffusion_models/chroma/` - Chroma model family
  - `chroma_model.py` - Main Chroma model class
  - `pipeline.py` - Chroma pipeline with denoising loop
  - `chroma_radiance_model.py` - Radiance variant

### Development Tools Guidelines

**IMPORTANT**: Always use editor tools (read_file, edit_notebook_file, replace_string_in_file, etc.) to view and modify files directly. Do NOT use terminal commands like grep, sed, cat, or find to examine or edit files.

### MPS Performance Issues to Watch For

1. **Excessive CPU↔GPU Transfers** - Tensors bouncing between CPU and GPU
2. **Timesteps Weights Not Cached** - No device change detection for weights
3. **Latent Image IDs Created on Wrong Device** - CPU then moved to device
4. **VAE Device Mismatch Risk** - VAE on different device than latents
5. **No Pipeline Caching** - Pipeline recreated unnecessarily
6. **Pipeline Not Moved to Device** - `pipeline.to(device)` not called in get_generation_pipeline()

### Implementation Order (Inner to Outer)

1. **Phase 1**: Core sampler optimizations (`custom_flowmatch_sampler.py`)
2. **Phase 2**: Model-level optimizations (`base_model.py`, `chroma_model.py`)
3. **Phase 3**: Pipeline-level optimizations (`pipeline.py`)

### For Each Change

1. Create a test to measure performance **before** making the change
2. **Clear Python bytecode cache** before running the test (see below)
3. Apply the fix
4. **Clear Python bytecode cache** again
5. Run the same test to verify improvement
6. Update `/memories/session/mps_improvements_plan.md` with results

### Clear Python Bytecode Cache (REQUIRED Before Each Test)

**Python caches compiled `.pyc` files in `__pycache__` directories. Stale cache from rejected/ reverted changes will cause old code to run, making optimizations appear to regress when they don't (or vice versa).**

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

### Key Files to Modify

1. `toolkit/samplers/custom_flowmatch_sampler.py` - Scheduler caching
2. `toolkit/models/base_model.py` - Pipeline caching
3. `extensions_built_in/diffusion_models/chroma/pipeline.py` - Device consistency, prepare_latent_image_ids device handling
4. `extensions_built_in/diffusion_models/chroma/chroma_model.py` - Pipeline device assignment
5. `extensions_built_in/diffusion_models/chroma/chroma_radiance_model.py` - Pipeline device assignment
6. `toolkit/optimizers/optimizer_utils.py` - MPS sync point optimization

### Test Pattern for Each Change (see above)

## MPS Optimization Status

See the following for the current status of all MPS optimizations:
- [mac-results.md](./mac-results.md) - Detailed test results for each change
  - Update this as each change is completed.
- [mac-change-6.md](./tasks/archive/mac-change-6.md) - Missing MPS logic analysis (Change #6)

---

**Last Updated**: 2026-06-08
