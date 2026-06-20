# CoPilot Instructions

## MPS (Apple Silicon) Optimization Guidelines for Z-Image

When working with MPS (Apple Silicon) performance in the AI Toolkit codebase, follow these guidelines:

### Code Categorization for MPS Optimization

#### 1. Main Pipeline Code
- `toolkit/stable_diffusion_model.py` - `generate_images()` method (line ~1200)
- `jobs/process/GenerateProcess.py` - Main generation orchestration
- `toolkit/models/base_model.py` - Base model class with pipeline management

#### 2. General Image Creation Code
- `toolkit/sampler.py` - Sampler utilities
- `toolkit/pipelines.py` - Custom pipeline implementations

### 3. Zimage Specific Code
- `extensions_built_in/diffusion_models/chroma/` - Chroma model family
  - `z_image.py` - Main Chroma model class

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
2. **Phase 2**: Model-level optimizations (`base_model.py`, `z_image.py`)
3. **Phase 3**: Pipeline-level optimizations (`pipeline.py`)

### For Each Change

1. Create a test to measure performance **before** making the change
2. Apply the fix
3. Run the same test to verify improvement
4. Update `/memories/session/mps_improvements_plan.md` with results

### Test Pattern for Each Change (see above)

## MPS Optimization Status

See the following for the current status of all MPS optimizations:
- [z_image-mac-results.md](z_image-mac-instructions.md) - Detailed test results for each change
  - Update this as each change is completed.
