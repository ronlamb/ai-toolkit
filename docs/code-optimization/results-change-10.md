# Results — Change #10: Pre-compute text fusion context in sampling loop

## Summary
No measurable improvement. Training and sample times are within normal variance of the baseline. The text fusion sub-network (4 blocks on text tokens) is a small fraction of total compute compared to the 28 main transformer blocks on the full combined sequence, so eliminating 27/28 of the fusion recomputes doesn't move the needle.

## Benchmark Results (6 epochs × 30 steps, 4 images each)

| Epoch | Steps | Avg Training (s/it) | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Avg Sample (s) |
|-------|-------|---------------------|----------|----------|----------|----------|-----------------|
| 1 | 30 | 3.42 | 66.48 | 65.93 | 65.73 | 65.61 | 65.94 |
| 2 | 30 | 3.23 | 67.05 | 66.11 | 65.82 | 66.32 | 66.33 |
| 3 | 30 | 3.20 | 68.28 | 66.88 | 66.24 | 65.94 | 66.84 |
| 4 | 30 | 3.23 | 65.64 | 65.63 | 65.58 | 65.58 | 65.61 |
| 5 | 30 | 3.22 | 66.27 | 66.69 | 67.16 | 67.50 | 66.91 |
| 6 | 30 | 3.22 | 69.93 | 69.76 | 68.90 | 67.88 | 69.12 |

## Stable Metrics (Epochs 4-6)
- **Training**: ~3.22s/it (avg of epochs 4-6: 3.23, 3.22, 3.22)
- **Samples**: ~67.21s/image (avg of epochs 4-6: 65.61, 66.91, 69.12)

## Comparison vs Previous (Change #9)

| Metric | Change #9 Epoch 6 | Change #10 Epoch 6 | Delta |
|--------|-------------------|--------------------|-------|
| Training (s/it) | 3.18 | 3.22 | +1.3% |
| Samples (s/image) | 65.71 (avg) | 69.12 (avg) | +5.2% |

## Comparison vs Baseline (results-baseline-asof-change5.md)

| Metric | Baseline Epoch 6 | Change #10 Epoch 6 | Delta |
|--------|------------------|--------------------|-------|
| Training (s/it) | 3.25 | 3.22 | -0.9% |
| Samples (s/image) | 65.64 (avg) | 69.12 (avg) | +5.3% |

## Verdict
No measurable improvement. Training is unchanged (as expected — training uses the inline path). Sample times are within normal variance (±5% baseline variation was observed in set-1). The text fusion sub-network (4 transformer blocks on text tokens) is a small fraction of total forward-pass compute compared to the 28 main SingleStreamBlocks on the full combined [text|image] sequence, so eliminating 27/28 redundant fusion calls doesn't produce a visible speedup. Kept for code cleanliness — zero runtime cost when not pre-fused, and the optimization is logically correct for future models where text fusion may be a larger fraction of compute.
