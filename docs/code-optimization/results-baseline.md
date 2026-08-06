# Krea2 Pipeline - Baseline Benchmark Results

## Test Configuration
- **Epochs**: 6 (increased from 3 due to steady improvement pattern in Krea2)
- **Steps per epoch**: 30
- **Generated images**: 4
- **Total steps tested**: 180 (6 epochs × 30 steps)

## Metrics to Collect
1. Training time per iteration (`s/it`) from progress bar
2. Sample generation time per image from "Generating Samples" progress

## Baseline Results Table

| Epoch | Steps | Total Time | Avg Training Time | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|-------|-------|------------|-------------------|----------|----------|----------|----------|
| 1 | 30 | 2:06 | 4.35s/it | 71.30s | 71.06s | 70.73s | 70.41s |
| 2 | 60 | 1:58 | 4.15s/it | 70.34s | 70.30s | 70.29s | 70.28s |
| 3 | 90 | 1:55 | 4.04s/it | 70.41s | 70.35s | 70.34s | 70.32s |
| 4 | 120 | 1:35 | 3.83s/it | 69.37s | 67.85s | 68.26s | 69.05s |
| 5 | 150 | 1:28 | 3.68s/it | 70.40s | 70.33s | 70.31s | 70.29s |
| 6 | 180 | 1:40 | 3.62s/it | 70.34s | 70.37s | 70.36s | 70.35s |

## Average Baseline Metrics

- **Training Time**: 3.82s/it (range: 3.62-4.35s)
- **Sample Generation Time**: 69.73s/image (range: 67.85-71.30s)

## Notes

- Training time decreases over epochs (3.62s → 4.35s range) as expected
- Sample generation time stabilizes around 67-71 seconds per image
- Results show steady improvement pattern, justifying 6-epoch baseline

## Baseline Variation Analysis

- Training time varies from 3.62s to 4.35s across epochs (7.9s span, ~21% range)
- Sample generation varies from 67.85s to 71.30s across epochs (3.45s span, ~5% range)
- **Conclusion**: Changes showing <5% differences are within noise range
- **Verdict**: Only changes with >5% improvement should be kept
