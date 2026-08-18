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
| 1 | 30 | 1:58 | 4.10s/it | 69.57s | 69.17s | 69.00s | 68.75s |
| 2 | 60 | 1:39 | 3.68s/it | 68.46s | 68.32s | 68.29s | 68.27s |
| 3 | 90 | 1:29 | 3.44s/it | 68.57s | 68.35s | 68.30s | 68.24s |
| 4 | 120 | 1:35 | 3.37s/it | 68.34s | 68.32s | 68.34s | 68.28s |
| 5 | 150 | 1:32 | 3.30s/it | 67.80s | 66.45s | 66.03s | 65.89s |
| 6 | 180 | 1:30 | 3.25s/it | 65.74s | 65.62s | 65.61s | 65.59s |

## Average Baseline Metrics

- **Training Time**: Average 3.24/it: (range: 3.25-4.10)
  - Bottomed out 3.24s/it = (sum of total time)/180 steps = 583 / 180
- **Sample Generation Time**: 67.72s/image (range: 65.74-69.57s)

## Notes

- Training time decreases over epochs (3.25s → 4.10s range) as expected
- Sample generation time stabilizes around 66-70 seconds per image
- Results show steady improvement pattern, justifying 6-epoch baseline

## Baseline Variation Analysis

- Training time varies from 3.25s to 4.10s across epochs
- Sample generation varies from 65.72s to 69.67s across epochs
