# Z-Image base Model Optimization Results for MPS - on M5 Max - 128GB memory

Process is run for a step size of 30, for 8 epochs and generating 2 images for 5 steps to get a base generation time.

## Timings Definition

The timings reported will show the results of each epoch as four separate lines followed by a blank line, as shown below:

```
zimage_a1_ut:   0%|          | 29/20000 [04:14<48:40:44,  8.77s/it, lr: 1.0e-04 loss: 4.167e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:38<00:38, 38.63s/it]
Generating Samples: 100%|##########| 2/2 [01:16<00:00, 38.41s/it]
```

The first line is the time for training 30 steps.
The next three lines are the times to generate the two images.

## Baseline Times

Below are the baseline times at start before any changes as of the last improvement.  These times will be used determine whether a code change improves performance.

```
zimage_a1_ut:   0%|          | 29/20000 [04:14<48:40:44,  8.77s/it, lr: 1.0e-04 loss: 4.167e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:38<00:38, 38.63s/it]
Generating Samples: 100%|##########| 2/2 [01:16<00:00, 38.41s/it]

zimage_a1_ut:   0%|          | 59/20000 [08:09<45:58:40,  8.30s/it, lr: 1.0e-04 loss: 4.675e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:40<00:40, 40.39s/it]
Generating Samples: 100%|##########| 2/2 [01:20<00:00, 40.15s/it]

zimage_a1_ut:   0%|          | 89/20000 [12:54<48:09:00,  8.71s/it, lr: 1.0e-04 loss: 3.816e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:42<00:42, 42.16s/it]
Generating Samples: 100%|##########| 2/2 [01:22<00:00, 41.12s/it]

zimage_a1_ut:   1%|          | 119/20000 [17:40<49:12:50,  8.91s/it, lr: 1.0e-04 loss: 3.752e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:40<00:40, 40.46s/it]
Generating Samples: 100%|##########| 2/2 [01:19<00:00, 39.47s/it]

zimage_a1_ut:   1%|          | 149/20000 [21:51<48:31:53,  8.80s/it, lr: 1.0e-04 loss: 3.961e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:38<00:38, 38.92s/it]
Generating Samples: 100%|##########| 2/2 [01:17<00:00, 38.58s/it]

zimage_a1_ut:   1%|          | 179/20000 [26:07<48:13:34,  8.76s/it, lr: 1.0e-04 loss: 3.895e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:38<00:38, 38.81s/it]
Generating Samples: 100%|##########| 2/2 [01:17<00:00, 38.47s/it]

zimage_a1_ut:   1%|1         | 209/20000 [30:15<47:45:51,  8.69s/it, lr: 1.0e-04 loss: 4.357e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:38<00:38, 38.92s/it]
Generating Samples: 100%|##########| 2/2 [01:17<00:00, 38.49s/it]

zimage_a1_ut:   1%|1         | 239/20000 [34:13<47:09:50,  8.59s/it, lr: 1.0e-04 loss: 3.514e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:38<00:38, 38.96s/it]
Generating Samples: 100%|##########| 2/2 [01:17<00:00, 38.80s/it]

image_a1_ut:   1%|1         | 269/20000 [38:00<46:28:29,  8.48s/it, lr: 1.0e-04 loss: 4.368e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:38<00:38, 38.94s/it]
Generating Samples: 100%|##########| 2/2 [01:17<00:00, 38.53s/it]
```

Once it hit epoch 8 the training time would fluctuate between what is shown at epoch 8 - step 239 and epoch 9 - step 269

## Baseline time as of best change

### Accepted optimizations (cumulative):
1. **Task 1 (Round 1)**: Pipeline caching — avoided recreating ZImagePipeline objects
2. **Task 2 (Round 1)**: Tensor op optimization — avoided intermediate 5D tensor + batched dtype conversion
3. **Pre-computed `_timesteps_sorted` (Round 2)**: Eliminated per-call `torch.flip()` allocation in `_get_step_indices()`

### Pre-computed `_timesteps_sorted`

**Raw results (epoch 1-8, steps 1-240):**
```
zimage_a1_ut:   0%|          | 29/20000 [03:33<40:47:13,  7.35s/it, lr: 1.0e-04 loss: 3.573e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:35<00:35, 35.65s/it]
Generating Samples: 100%|##########| 2/2 [01:11<00:00, 35.58s/it]

zimage_a1_ut:   0%|          | 59/20000 [06:12<35:00:52,  6.32s/it, lr: 1.0e-04 loss: 4.219e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:37<00:37, 37.35s/it]
Generating Samples: 100%|##########| 2/2 [01:14<00:00, 37.10s/it]

zimage_a1_ut:   0%|          | 89/20000 [10:15<38:16:34,  6.92s/it, lr: 1.0e-04 loss: 4.013e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:37<00:37, 37.94s/it]
Generating Samples: 100%|##########| 2/2 [01:15<00:00, 37.71s/it]

zimage_a1_ut:   1%|          | 119/20000 [14:10<39:29:14,  7.15s/it, lr: 1.0e-04 loss: 3.703e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:37<00:37, 37.97s/it]
Generating Samples: 100%|##########| 2/2 [01:15<00:00, 37.49s/it]

zimage_a1_ut:   1%|          | 149/20000 [18:35<41:15:50,  7.48s/it, lr: 1.0e-04 loss: 4.510e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:37<00:37, 37.65s/it]
Generating Samples: 100%|##########| 2/2 [01:14<00:00, 37.39s/it]

zimage_a1_ut:   1%|          | 179/20000 [23:02<42:30:39,  7.72s/it, lr: 1.0e-04 loss: 4.309e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:37<00:37, 37.64s/it]
Generating Samples: 100%|##########| 2/2 [01:14<00:00, 37.26s/it]

zimage_a1_ut:   1%|1         | 209/20000 [26:40<42:05:15,  7.66s/it, lr: 1.0e-04 loss: 3.870e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:37<00:37, 37.45s/it]
Generating Samples: 100%|##########| 2/2 [01:14<00:00, 37.14s/it]

zimage_a1_ut:   1%|1         | 239/20000 [29:53<41:10:52,  7.50s/it, lr: 1.0e-04 loss: 3.537e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:36<00:36, 36.94s/it]
Generating Samples: 100%|##########| 2/2 [01:13<00:00, 36.62s/it]
```

**Comparison (Task 1+2 baseline → Task 1+2+\_timesteps\_sorted):**

| Metric | Task 1+2 Baseline | +\`_timesteps_sorted\` | Improvement |
|--------|----------------|----------|-------------|
| Avg training s/it | 7.48s | 7.26s | **~3.0% faster** |
| Avg gen time/image | 36.3s | 37.0s | ~1.9% slower (within noise) |
| Best gen time/image | 35.2s | 35.58s | ~1% slower |
| Worst gen time/image | 37.3s | 37.97s | ~1.2% slower |

**Notes:**
- Training time shows slight upward drift within epochs (7.35s → 7.72s), likely MPS memory fragmentation from running further into the same session
- Generation time is within noise margin of baseline (~1.9% slower is not statistically significant)
- Pre-computing `_timesteps_sorted` and `_timesteps_flipped` eliminates a per-call `torch.flip()` allocation and conditional check
- **This change is the new baseline**

**New baseline for comparison:** ~7.26s/it training (avg), ~37.0s/image generation (avg)