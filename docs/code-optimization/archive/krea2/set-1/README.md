# Krea2 Pipeline Optimization - Summary

## Overview
This document summarizes the 5 optimization opportunities identified for the Krea2 pipeline.

## Optimization Opportunities

### Change #1: VAE Frame Dimension Optimization
- **Status**: ⚠️ PROPOSED
- **Complexity**: Simple (1-5 lines)
- **Expected Impact**: 2-3%
- **File**: `extensions_built_in/diffusion_models/krea2/krea2.py`
- **Description**: Eliminate redundant `unsqueeze(2)`/`squeeze(2)` operations in VAE encode/decode by processing images individually

### Change #2: torch.compile for predict_velocity
- **Status**: ⚠️ PROPOSED
- **Complexity**: Moderate (6-10 lines)
- **Expected Impact**: 5-8%
- **File**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`
- **Description**: Add `@torch.compile(mode="reduce-overhead", dynamic=True)` decorator to the `predict_velocity` function

### Change #3: Text Feature Padding Optimization
- **Status**: ⚠️ PROPOSED
- **Complexity**: Moderate (6-10 lines)
- **Expected Impact**: 3-5%
- **File**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`
- **Description**: Replace loop-based tensor assignment with vectorized operations using `torch.stack()` and `torch.arange()`

### Change #4: Aggressive Gradient Checkpointing
- **Status**: ⚠️ PROPOSED
- **Complexity**: Complex (11-20 lines)
- **Expected Impact**: 5-7% (VRAM reduction, potential speedup)
- **File**: `extensions_built_in/diffusion_models/krea2/src/mmdit.py`
- **Description**: Apply gradient checkpointing to `TextFusionBlock`, `TextFusionTransformer`, and `SingleStreamDiT`

### Change #5: Dtype Conversion Optimization
- **Status**: ⚠️ PROPOSED
- **Complexity**: Simple (1-5 lines)
- **Expected Impact**: 2-3%
- **File**: `extensions_built_in/diffusion_models/krea2/krea2.py`
- **Description**: Use model dtype directly for timesteps instead of converting to float32

## Baseline Metrics

| Metric | Value |
|--------|-------|
| Training Time (s/it) | 3.82s (avg) |
| Sample Generation Time (s/image) | 69.73s (avg) |

## Implementation Files

- `docs/code-optimization/current-state.md` - Current state of optimizations
- `docs/code-optimization/results-baseline.md` - Baseline benchmark results
- `docs/code-optimization/implementation-proposal-change-1.md` - Change #1 details
- `docs/code-optimization/implementation-proposal-change-2.md` - Change #2 details
- `docs/code-optimization/implementation-proposal-change-3.md` - Change #3 details
- `docs/code-optimization/implementation-proposal-change-4.md` - Change #4 details
- `docs/code-optimization/implementation-proposal-change-5.md` - Change #5 details

## Testing Protocol

For each change:
1. Implement the optimization
2. Run benchmark test: 3 epochs × 30 steps, generate 4 images
3. Compare against baseline results
4. Keep change only if improvement >5%

## Notes

- Previous optimizations (latents.to, non_blocking transfers) did not improve performance
- Baseline variation is ~21% training, ~5% samples - require >5% improvement to confirm real benefit
- User will run benchmark tests and provide logs
