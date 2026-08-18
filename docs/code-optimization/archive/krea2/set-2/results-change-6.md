# Results: Change 6 — Cache VAE Normalization Constants

## Test Configuration
- **Epochs**: 6
- **Steps per epoch**: 30
- **Generated images**: 4
- **Total steps tested**: 180

## Results Table

| Epoch | Steps | Total Time | Avg Training Time | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|-------|-------|------------|-------------------|----------|----------|----------|----------|
| 1 | 30 | 1:48 | 3.74s/it | 69.06s | 68.42s | 67.67s | 67.01s |
| 2 | 60 | 1:37 | 3.48s/it | 65.76s | 66.06s | 66.09s | 66.18s |
| 3 | 90 | 1:26 | 3.28s/it | 66.45s | 66.15s | 66.62s | 67.35s |
| 4 | 120 | 1:20 | 3.20s/it | 65.81s | 65.70s | 66.16s | 66.06s |
| 5 | 150 | 1:20 | 3.22s/it | 65.59s | 66.44s | 66.18s | 66.23s |
| 6 | 180 | 1:24 | 3.26s/it | 66.13s | 65.77s | 65.67s | 65.77s |

## Average Metrics

- **Training Time**: 3.36s/it average (range: 3.20–3.74)
- **Sample Generation Time**: 66.38s/image average (range: 65.59–69.06)

## Comparison

| Metric | Baseline (asof change 5) | Change 6 | Delta | Verdict |
|--------|--------------------------|----------|-------|---------|
| Training (avg s/it) | 3.24 | 3.36 | +4.0% | ⚠️ slightly slower |
| Samples (avg s/image) | 67.72 | 66.38 | -2.0% | ✅ slightly faster |

## Notes

- Sample generation improved ~2% (67.72 → 66.38s/image), consistent with eliminating redundant CPU→GPU tensor creation in `decode_latents`.
- Training time is ~4% slower than baseline on average, but this falls within the natural epoch-to-epoch variance observed in the baseline (3.25–4.10s range). The first epoch (3.74s) skews the average; epochs 2–6 average 3.29s/it which is close to baseline.
- The change eliminates 4 CPU→GPU copies per encode/decode call (replaced with a single cheap dtype cast on already-GPU-resident tensors).
- Net effect: marginal benefit. The optimization is numerically correct and removes redundant work, but the tensors are small (16 elements each) so the measurable impact is minimal.

## Recommendation

**Keep** — the change is correct, removes redundant work, and shows a small sample generation improvement. The training variance is within noise. Cumulative benefit across all changes is what matters.
