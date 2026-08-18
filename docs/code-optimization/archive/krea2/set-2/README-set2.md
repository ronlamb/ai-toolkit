# Krea2 Pipeline Optimization - Set 2 Summary

## Overview
This document summarizes the 5 new optimization opportunities identified for the Krea2 pipeline (set 2), building on set-1 (changes #1–#5, archived in `docs/code-optimization/archive/krea2/set-1/`).

**Focus areas for set 2**:
- Excessive memory copies between CPU and GPU
- Memory usage waste
- Inefficient loops, including calls that don't change and can be moved outside the loop

**Current best metrics entering set 2** (after set-1, change #5 full-run validation):
- Training time: **3.03s/it** (bottom-out)
- Sample generation: **65.12s/image**

## Optimization Opportunities (Set 2)

### Change #6: Cache VAE normalization constants
- **Status**: ⬜ PROPOSED
- **Complexity**: Simple (1-5 lines)
- **Expected Impact**: 1-2%
- **File**: `extensions_built_in/diffusion_models/krea2/krea2.py`
- **Focus**: CPU→GPU copies / memory waste
- **Description**: `encode_images` and `decode_latents` rebuild `latents_mean`/`latents_std` from Python config lists (CPU→GPU copy) on every call. Cache them once at load time as model attributes.

### Change #7: Cache position grid and mask in `prepare()`
- **Status**: ⬜ PROPOSED
- **Complexity**: Moderate (6-10 lines)
- **Expected Impact**: 2-4%
- **File**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`
- **Focus**: Inefficient repeated work / memory waste
- **Description**: `prepare()` rebuilds the RoPE position grid, text positions (zeros), and image mask on every call. For a fixed resolution these are constant — cache them per `(b, txtlen, h_, w_)`.

### Change #8: Cache RoPE frequencies for fixed positions
- **Status**: ⬜ PROPOSED (depends on Change #7)
- **Complexity**: Moderate (6-10 lines)
- **Expected Impact**: 1-3% sampling
- **File**: `extensions_built_in/diffusion_models/krea2/src/mmdit.py`
- **Focus**: Calls that don't change, moved outside the loop
- **Description**: `freqs = self.posemb(pos)` recomputes RoPE trig/einsum over ~4600 tokens every forward pass. Since `pos` is constant per sampling run, cache the freqs keyed on `(data_ptr, shape)`. **Requires Change #7 for a stable `pos` object.**

### Change #9: Single dtype conversion in CFG sampling loop
- **Status**: ⬜ PROPOSED
- **Complexity**: Simple (1-5 lines)
- **Expected Impact**: 1-2% sampling
- **File**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`
- **Focus**: Excessive memory copies
- **Description**: `latents.to(dtype)` is called twice per denoising step when CFG is on (cond + uncond passes). Hoist to a single cast per step.

### Change #10: Pre-compute text fusion context in sampling loop
- **Status**: ⬜ PROPOSED — **highest impact**
- **Complexity**: Complex (11-20 lines)
- **Expected Impact**: 5-8% sampling
- **File**: `extensions_built_in/diffusion_models/krea2/src/mmdit.py` + `src/pipeline.py`
- **Focus**: Calls that don't change, moved outside the loop
- **Description**: The text context passes through `txtfusion` (4 transformer blocks) + `txtmlp` on every forward pass. In the 28-step sampling loop the text is constant, so pre-fuse it once before the loop and pass `fused_context` into `forward`, skipping 27/28 recomputations per image.

## Recommended Implementation Order

1. **Change #6** (VAE constants) — trivial, independent
2. **Change #9** (single dtype cast) — trivial, independent
3. **Change #7** (position grid cache) — enables Change #8
4. **Change #10** (pre-fuse text context) — highest impact, independent
5. **Change #8** (RoPE freqs cache) — apply after Change #7

Each change should be tested individually per the benchmark protocol before moving to the next.

## Testing Protocol

For each change:
1. Implement the optimization
2. Run benchmark test: 3 epochs × 30 steps, generate 4 images
3. Compare against current best (set-1 results)
4. Keep change only if improvement >5%

## State Files (Set 2)

- `docs/code-optimization/current-state.md` — tracks pending/completed changes
- `docs/code-optimization/implementation-proposal-change-N.md` — detailed proposal (N = 6..10)
- `docs/code-optimization/results-baseline.md` — baseline (reuse set-1 best as reference)
- `docs/code-optimization/results-change-N.md` — per-change benchmark results

## Notes

- Set-1 established baseline variation of ~21% training, ~5% samples; only >5% improvements are kept.
- Changes #7 and #8 compound (position cache enables RoPE freqs cache).
- Change #10 is the headline win; it targets the 28-step sampling loop directly.
- User runs benchmark tests and provides logs; user handles all git commits/pushes.
