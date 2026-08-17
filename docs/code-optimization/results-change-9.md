# Results — Change #9: Single dtype conversion in CFG sampling loop

## Summary
Slight improvement visible when comparing against the actual baseline file. Training starts faster and bottoms out lower. Sample times are essentially identical (within normal variance).

## Benchmark Results (6 epochs × 30 steps, 4 images each)

| Epoch | Steps | Avg Training (s/it) | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Avg Sample (s) |
|-------|-------|---------------------|----------|----------|----------|----------|-----------------|
| 1 | 30 | 3.52 | 66.93 | 68.07 | 67.98 | 67.33 | 67.58 |
| 2 | 30 | 3.38 | 66.40 | 66.56 | 67.04 | 66.93 | 66.73 |
| 3 | 30 | 3.36 | 69.87 | 69.74 | 68.94 | 68.69 | 68.81 |
| 4 | 30 | 3.27 | 65.71 | 65.79 | 65.69 | 65.82 | 65.75 |
| 5 | 30 | 3.19 | 65.81 | 65.65 | 65.62 | 65.64 | 65.68 |
| 6 | 30 | 3.18 | 65.78 | 65.72 | 65.68 | 65.66 | 65.71 |

## Stable Metrics (Epochs 4-6)
- **Training**: ~3.22s/it (avg of epochs 4-6: 3.27, 3.19, 3.18)
- **Samples**: ~65.71s/image (avg of epochs 4-6: 65.75, 65.68, 65.71)

## Comparison vs Baseline (results-baseline-asof-change5.md)

| Metric | Baseline Epoch 1 | Change #9 Epoch 1 | Delta | Baseline Epoch 6 | Change #9 Epoch 6 | Delta |
|--------|------------------|-------------------|-------|------------------|-------------------|-------|
| Training (s/it) | 4.10 | 3.52 | **-14.1%** | 3.25 | 3.18 | **-2.2%** |
| Samples (s/image) | 69.12 (avg) | 67.58 (avg) | **-2.2%** | 65.64 (avg) | 65.71 (avg) | +0.1% |

## Verdict
Training shows a modest improvement (starts 14% faster, bottoms out 2% faster). Sample times are within normal variance. The single eliminated dtype cast alone is unlikely to account for the training improvement — this may reflect cumulative benefits from set-1 changes. Kept for code cleanliness; neutral-to-slight-positive impact.
