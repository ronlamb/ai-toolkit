# Krea2 Pipeline Optimization - Current State (Set 2)

## Overview
This document tracks pending and completed optimization changes for the Krea2 pipeline (set 2).

**Previous set**: See `docs/code-optimization/archive/krea2/set-1/` for changes #1–#5.

**Current best metrics** (after set-1, change #5 full-run validation):
- Training time: **3.03s/it** (bottom-out)
- Sample generation: **65.12s/image**

## Optimization Opportunities (Set 2)

### Change #6: Cache VAE normalization constants (latents_mean / latents_std)
**Status**: ⬜ PROPOSED  
**Complexity**: Simple (1-5 lines)  
**Expected Impact**: 1-2% (eliminates repeated CPU→GPU tensor creation on every encode/decode)  

**Issue**: Both `encode_images` and `decode_latents` create `latents_mean` and `latents_std` tensors from Python config lists on every call. These values are constant for the lifetime of the model. Caching them as model attributes eliminates repeated `torch.tensor(...)` + `.to(device, dtype)` operations (CPU→GPU copies).

**Location**: `extensions_built_in/diffusion_models/krea2/krea2.py`, lines 780-850

**Details**: See `implementation-proposal-change-6.md`

---

### Change #7: Cache position grid and mask in `prepare()`
**Status**: ⚠️ REVERTED — No measurable improvement, training slower  
**Complexity**: Moderate (6-10 lines)  
**Expected Impact**: 2-4% (both training and sampling)  
**Actual Result**: Training 3.53s/it (+8.6% vs baseline), Samples 67.81s/image (+0.1% vs baseline). Reverted.  

**Issue**: The `prepare()` function creates the image position grid (`imgids`), text positions (all zeros), and image mask on every call. For a fixed resolution, these tensors are identical across all steps. In the sampling loop (28 steps), this means 28 redundant allocations of the same position/mask tensors. Caching them per `(h_, w_, txtlen, b)` eliminates ~5 tensor allocations and a `repeat` operation every step.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 74-95

**Details**: See `implementation-proposal-change-7.md`

---

### Change #8: Cache RoPE frequencies for fixed positions
**Status**: ⚠️ REVERTED — Cannot work without Change #7; shape-based key produces wrong results  
**Complexity**: Moderate (6-10 lines)  
**Expected Impact**: 3-5% (both training and sampling)  
**Actual Result**: Corrupted output. Shape-based cache key `(tuple(pos.shape), pos.device)` is unsafe — different aspect ratios (e.g., 512×1024 vs 1024×512) produce the same `pos.shape` but different position values, returning wrong RoPE frequencies. Original proposal correctly required Change #7's stable `data_ptr()` identity. Without it, no safe cache key exists.  

**Issue**: `freqs = self.posemb(pos)` computes RoPE frequencies on every forward pass. Caching requires a safe key that uniquely identifies position content.

**Location**: `extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `SingleStreamDiT.forward` (~line 570)

**Details**: See `implementation-proposal-change-8.md`

---

### Change #9: Single dtype conversion in CFG sampling loop
**Status**: ⬜ PROPOSED  
**Complexity**: Simple (1-5 lines)  
**Expected Impact**: 1-2% sample generation speedup  

**Issue**: In `Krea2Pipeline.__call__`, when CFG is enabled, `latents.to(dtype)` is called twice per denoising step (once for the conditional pass, once for the unconditional pass). Computing it once and reusing saves a redundant tensor copy per step (28 steps × 1 extra copy eliminated).

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 360-380

**Details**: See `implementation-proposal-change-9.md`

---

### Change #10: Pre-compute text fusion context in sampling loop
**Status**: ⬜ PROPOSED  
**Complexity**: Complex (11-20 lines)  
**Expected Impact**: 5-8% sample generation speedup  

**Issue**: In `SingleStreamDiT.forward`, the text context passes through `txtfusion` (4 transformer blocks with attention + MLP) and `txtmlp` (norm + 2 linears) on EVERY call. In the 28-step sampling loop, the text context is identical across all steps, so this is 27 redundant computations of a non-trivial sub-network. Pre-computing the fused context once before the loop and passing it directly eliminates ~27/28 of these computations.

**Location**: `extensions_built_in/diffusion_models/krea2/src/mmdit.py` (add `fuse_context` method) + `extensions_built_in/diffusion_models/krea2/src/pipeline.py` (pipeline loop)

**Details**: See `implementation-proposal-change-10.md`

## Testing Protocol

For each change:
1. Implement the optimization
2. Run benchmark test: 3 epochs × 30 steps, generate 4 images
3. Compare against current best (set-1 results)
4. Keep change only if improvement >5%

## Notes

- Set-1 established that baseline variation is ~21% training, ~5% samples
- Only changes with >5% improvement should be kept (per set-1 verdict criteria)
- Changes #7, #8, and #10 compound: caching positions enables caching RoPE freqs, and pre-fusing context is independent
- User will run benchmark tests and provide logs
