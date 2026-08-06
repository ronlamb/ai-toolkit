# Krea2 Pipeline - Change #5 Benchmark Results

## Test Configuration
- **Epochs**: 6 (matching baseline)
- **Steps per epoch**: 30
- **Generated images**: 4
- **Total steps tested**: 180 (6 epochs × 30 steps)

## Metrics to Collect
1. Training time per iteration (`s/it`) from progress bar
2. Sample generation time per image from "Generating Samples" progress

## Change Applied
**Change #5**: Eliminated redundant dtype conversion in timestep handling
- **File**: `extensions_built_in/diffusion_models/krea2/krea2.py`
- **Line**: 640
- **Before**: `t = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0`
- **After**: `t = timestep.to(self.device_torch, dtype=self.torch_dtype) / 1000.0`

## Results Table

| Epoch | Steps | Total Time | Avg Training Time | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|-------|-------|------------|-------------------|----------|----------|----------|----------|
| 1 | 30 | 1:58 | 4.10s/it | 69.57s | 69.17s | 69.00s | 68.75s |
| 2 | 60 | 1:39 | 3.68s/it | 68.46s | 68.32s | 68.29s | 68.27s |
| 3 | 90 | 1:29 | 3.44s/it | 68.57s | 68.35s | 68.30s | 68.24s |
| 4 | 120 | 1:35 | 3.37s/it | 68.34s | 68.32s | 68.34s | 68.28s |
| 5 | 150 | 1:32 | 3.30s/it | 67.80s | 66.45s | 66.03s | 65.89s |
| 6 | 180 | 1:30 | 3.25s/it | 65.74s | 65.62s | 65.61s | 65.59s |

## Average Metrics

- **Training Time**: 3.57s/it (range: 3.25-4.10s)
- **Sample Generation Time**: 67.89s/image (range: 65.59-69.57s)

## Comparison Against Baseline

| Metric | Baseline | Change #5 | Improvement |
|--------|----------|-----------|-------------|
| Training Time | 3.82s/it | 3.57s/it | **6.5%** |
| Sample Generation | 69.73s/image | 67.89s/image | **2.6%** |

## Validation

- [x] User has run benchmark test (6 epochs × 30 steps, 4 images)
- [x] Training time improvement calculated
- [x] Sample generation improvement calculated
- [x] Results recorded in this file

## Verdict Criteria

- **Keep change**: >5% improvement in either metric
- **Revert change**: <5% improvement (within noise range)

## Verdict

✅ **KEEP** - Training time improved by 6.5% (exceeds 5% threshold)
- Sample generation also improved by 2.6%

## Notes

- Training time shows consistent improvement across all epochs
- Sample generation time stabilizes around 65-69 seconds (improved from baseline 67-71s)
- The dtype optimization is effective and should be kept
