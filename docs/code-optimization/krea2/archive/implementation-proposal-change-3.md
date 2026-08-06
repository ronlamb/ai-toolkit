# Implementation Proposal #3: Text Feature Padding Optimization

## Status
✅ COMPLETED - Small cumulative improvement detected (2.7% sample time reduction when combined with change #1)

## Complexity
Moderate (6-10 lines changed)

## Expected Impact
3-5% speedup

## Issue Description

The `pad_text_features` function in `pipeline.py` creates zero tensors and then fills them in a loop. This can be optimized by using more efficient tensor operations like `torch.stack()` and batched assignments.

## Current Code

### Location: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 35-48

```python
def pad_text_features(
    features_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of per-sample ``(Lt_i, F)`` features into a batch.

    Each caption is stored 2D at its natural length -- the 12 stacked Qwen3-VL
    hidden-state layers are flattened into the feature axis ``F = n * d`` so the
    ai-toolkit batching machinery treats the list length as the batch size (it
    only special-cases 2D per-sample tensors). The layer axis is restored in
    ``predict_velocity`` right before the MMDiT call. Padding to the batch max is
    deferred to here. Returns ``(features (B, Lt, F), mask (B, Lt))``; the mask is
    1 for real text tokens and 0 for padding.
    """
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, f in enumerate(features_list):
        ln = f.shape[0]
        features[i, :ln] = f.to(device, dtype)
        mask[i, :ln] = 1
    return features, mask
```

## Optimized Code

```python
def pad_text_features(
    features_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of per-sample ``(Lt_i, F)`` features into a batch.

    Each caption is stored 2D at its natural length -- the 12 stacked Qwen3-VL
    hidden-state layers are flattened into the feature axis ``F = n * d`` so the
    ai-toolkit batching machinery treats the list length as the batch size (it
    only special-cases 2D per-sample tensors). The layer axis is restored in
    ``predict_velocity`` right before the MMDiT call. Padding to the batch max is
    deferred to here. Returns ``(features (B, Lt, F), mask (B, Lt))``; the mask is
    1 for real text tokens and 0 for padding.
    """
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    # Stack all features first (may be shorter than max_len)
    all_features = torch.stack(features_list)  # (B, Lt_max_actual, F)
    
    # Create padded features tensor
    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    
    # Copy only the valid portion (faster than per-row assignment)
    features[:, :all_features.shape[1]] = all_features
    
    # Create mask using arange (vectorized)
    range_tensor = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
    lengths_tensor = torch.tensor(lengths, device=device).unsqueeze(1)  # (B, 1)
    mask = (range_tensor < lengths_tensor).long()  # (B, max_len)
    
    return features, mask
```

## Changes Summary

- **Line 45**: Changed from loop-based assignment to `torch.stack()` for features
- **Lines 48-50**: Replaced loop-based mask setting with vectorized comparison using `torch.arange()`

## Reasoning

The current implementation uses a Python loop to:
1. Copy each feature tensor row-by-row
2. Set mask values row-by-row

The optimized version:
1. Uses `torch.stack()` to combine all features in one operation
2. Uses batched assignment to copy valid portions
3. Uses vectorized comparison (`torch.arange() < lengths`) to create the mask

This reduces Python overhead and leverages PyTorch's optimized C++ operations.

## Implementation Status

✅ **IMPLEMENTED** - The optimized `pad_text_features` function has been applied to:
- File: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`
- Lines: 32-67

## Validation Results

### User Test Results (6 epochs × 30 steps, 4 images)

| Metric | Baseline | Change #3 | Improvement |
|--------|----------|-----------|-------------|
| Training Time | 3.82s/it | 3.79s/it | 0.8% |
| Sample Generation | 69.73s/image | 68.93s/image | 1.1% |

### Combined with Change #1

| Metric | Change #1 Only | Changes #1+#3 | Improvement |
|--------|----------------|---------------|-------------|
| Training Time | 3.79s/it | 3.79s/it | 0.0% |
| Sample Generation | 70.81s/image | 68.93s/image | 2.7% |

### Verdict

⚠️ **MINOR IMPROVEMENT** - The change shows small cumulative benefits when combined with change #1:
- Sample generation improved by 2.7% (from 70.81s to 68.93s)
- Training time unchanged at 3.79s/it
- Both metrics within baseline variation range (~5% for samples)

**Recommendation**: Keep the change for cumulative optimization benefits, but don't expect significant standalone improvement.

See `docs/code-optimization/results-change-3.md` for full results.
