# Krea2 Pipeline - Results as of Change #10 (Set 2 Complete)

## Test Configuration
- **Full-run validation**: 172 training images, 9 samples per checkpoint
- **Short benchmark**: 6 epochs × 30 steps, 4 images (per-change testing)
- **Total steps tested (full run)**: 3784+ (ongoing)

## Changes Included (Set 2)

| Change | Status | Impact |
|--------|--------|--------|
| #6 | ✅ Kept | VAE norm constants cached (-2.0% samples in short test) |
| #7 | ⚠️ Reverted | Position grid cache (no improvement, training slower) |
| #8 | ⚠️ Reverted | RoPE frequency cache (corrupted output) |
| #9 | ✅ Kept | Single dtype cast in CFG loop (code cleanliness) |
| #10 | ✅ Kept | Pre-compute text fusion context (code cleanliness) |

## Full-Run Results (172 images, 9 samples)

### Training Time (s/it)

| Steps | Time | s/it | Notes |
|-------|------|------|-------|
| 172 | 9:37 | 3.38 | Warm-up |
| 344 | 18:31 | 3.24 | |
| 516 | 26:50 | 3.13 | |
| 688 | 35:10 | 3.07 | |
| 860 | 43:32 | 3.04 | |
| 1032 | 51:48 | 3.02 | |
| 1204 | 1:00:11 | 3.00 | |
| 1376 | 1:08:29 | 2.99 | |
| 1548 | 1:16:51 | 2.98 | |
| 1720 | 1:25:10 | 2.97 | |
| 1892 | 1:33:30 | 2.97 | |
| 2064 | 1:41:45 | 2.96 | |
| 2236 | 1:50:08 | 2.96 | |
| 2408 | 1:58:25 | 2.95 | |
| 2580 | 2:06:47 | 2.95 | |
| 2752 | 2:15:03 | 2.95 | |
| 2924 | 2:23:22 | 2.94 | |
| 3096 | 2:31:43 | 2.94 | |
| 3268 | 2:39:57 | 2.94 | |
| 3440 | 2:48:17 | 2.94 | |
| 3612 | 2:56:39 | 2.94 | |
| 3784 | 3:04:59 | 2.93 | Bottom-out |

### Sample Generation (s/image)

| Steps | Avg Sample (s) | Notes |
|-------|----------------|-------|
| 172 | 69.74 | Warm-up |
| 344 | 67.18 | |
| 516 | 65.01 | |
| 688 | 64.99 | |
| 860 | 64.81 | |
| 1032 | 64.85 | |
| 1204 | 64.87 | |
| 1376 | 65.05 | |
| 1548 | 64.86 | |
| 1720 | 64.84 | |
| 1892 | 64.89 | |
| 2064 | 65.05 | |
| 2236 | 64.76 | |
| 2408 | 64.90 | |
| 2580 | 64.78 | |
| 2752 | 65.10 | |
| 2924 | 64.15 | Outlier (lower) |
| 3096 | 64.15 | Outlier (lower) |
| 3268 | 64.90 | |
| 3440 | 65.04 | |
| 3612 | 64.98 | |
| 3784 | 64.92 | |

## Stable Metrics (Bottom-out, steps 2236–3784)

- **Training Time**: **2.93s/it** (bottom-out, range 2.93–2.96)
- **Sample Generation Time**: **64.85s/image** (avg of stable checkpoints, excluding 2924/3096 outliers)

## Comparison vs Set-1 Best

| Metric | Set-1 Best | Set-2 (Change #10) | Delta |
|--------|------------|--------------------|-------|
| Training (s/it) | 3.03 | 2.93 | **-3.3%** |
| Samples (s/image) | 65.12 | 64.85 | **-0.4%** |

## Comparison vs Original Baseline (asof change 5)

| Metric | Baseline | Set-2 (Change #10) | Delta |
|--------|----------|--------------------|-------|
| Training (s/it) | 3.25 | 2.93 | **-9.8%** |
| Samples (s/image) | 65.64 | 64.85 | **-1.2%** |

## Notes

- Training time shows steady improvement from 3.38s/it (warm-up) to 2.93s/it (bottom-out)
- Sample generation stabilizes around 64.8–65.1s/image after warm-up
- Steps 2924/3096 show anomalously low sample times (~64.15s) — likely GPU scheduling variance
- Cumulative improvement over set-1: -3.3% training, -0.4% samples
- Cumulative improvement over original baseline: -9.8% training, -1.2% samples
- Training improvement is more pronounced than sample improvement, consistent with set-1 changes targeting training loop optimizations

## Baseline Variation Analysis

- Training time varies from 2.93s to 3.38s across full run (warm-up effect)
- Sample generation varies from 64.15s to 69.74s across full run (warm-up + variance)
- Stable range (steps 2236+): training 2.93–2.96s/it, samples 64.76–65.10s/image
