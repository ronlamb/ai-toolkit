# Results: Change 7 — Cache Position Grid and Mask in `prepare()`

## Status: ⚠️ REVERTED

## Test Configuration
- **Epochs**: 6
- **Steps per epoch**: 30
- **Generated images**: 4
- **Total steps tested**: 180

## Results Table

| Epoch | Steps | Total Time | Avg Training Time | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|-------|-------|------------|-------------------|----------|----------|----------|----------|
| 1 | 30 | 1:52 | 3.89s/it | 69.65s | 68.80s | 68.54s | 68.57s |
| 2 | 60 | 3:41 | 3.75s/it | 68.61s | 68.36s | 67.94s | 67.95s |
| 3 | 90 | 5:08 | 3.47s/it | 67.45s | 67.59s | 67.59s | 67.71s |
| 4 | 120 | 6:45 | 3.41s/it | 68.54s | 68.32s | 68.59s | 68.53s |
| 5 | 150 | 8:23 | 3.38s/it | 69.06s | 69.14s | 68.48s | 67.28s |
| 6 | 180 | 9:45 | 3.27s/it | 65.61s | 65.70s | 65.75s | 65.71s |

## Average Metrics

- **Training Time**: 3.53s/it average (range: 3.27–3.89)
- **Sample Generation Time**: 67.81s/image average (range: 65.61–69.65)

## Comparison

| Metric | Baseline (asof change 5) | Change 6 | Change 7 | Delta vs Baseline | Delta vs Change 6 | Verdict |
|--------|--------------------------|----------|----------|-------------------|-------------------|---------|
| Training (avg s/it) | 3.24 | 3.36 | 3.53 | +8.6% | +4.8% | ⚠️ slower |
| Samples (avg s/image) | 67.72 | 66.38 | 67.81 | +0.1% | +2.2% | ⚠️ no improvement |

## Notes

- Neither training nor sample generation shows a measurable improvement over baseline or change 6.
- Training time is ~8.6% slower than baseline on average, but this falls within the natural epoch-to-epoch variance observed in the baseline (3.25–4.10s range). The first epoch (3.89s) skews the average; epochs 2–6 average 3.46s/it.
- Sample generation (67.81s) is essentially identical to baseline (67.72s) — a 0.1% difference is within noise.
- The position grid tensors are small (~4600 tokens × 3 floats = ~55KB), so the allocation overhead being eliminated is minimal compared to the overall forward pass cost.
- The `repeat` and `cat` operations are lightweight tensor views/copies, not the bottleneck.

## Recommendation

**REVERTED** — Training time increased 8.6% over baseline (3.24 → 3.53s/it), beyond acceptable noise. Sample generation showed no improvement (67.72 → 67.81s/image). Change has been reverted to original code.
