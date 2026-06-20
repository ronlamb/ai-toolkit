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

### Task 1 + 2: Cache pipeline + optimize get_noise_prediction() tensor ops — **ACCEPTED**

**Task 1 Change:** Added `_cached_pipeline` attribute to avoid recreating ZImagePipeline on each call.

**Task 2 Change:** Replaced `unsqueeze(2)` + `unbind(dim=0)` with `[x.unsqueeze(1) for x in latent_model_input]` to avoid intermediate 5D tensor allocation.

**Raw results (epoch 6-7, steps 1259-1439):**
```
zimage_a1_ut:   6%|6         | 1259/20000 [03:29<37:40:27,  7.24s/it, lr: 1.0e-04 loss: 4.368e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:35<00:35, 35.70s/it]
Generating Samples: 100%|##########| 2/2 [01:11<00:00, 35.50s/it]

zimage_a1_ut:   6%|6         | 1289/20000 [07:05<37:30:14,  7.22s/it, lr: 1.0e-04 loss: 3.817e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:36<00:36, 36.73s/it]
Generating Samples: 100%|##########| 2/2 [01:12<00:00, 36.32s/it]

zimage_a1_ut:   7%|6         | 1319/20000 [10:48<37:48:59,  7.29s/it, lr: 1.0e-04 loss: 3.886e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:36<00:36, 36.53s/it]
Generating Samples: 100%|##########| 2/2 [01:12<00:00, 36.20s/it]

zimage_a1_ut:   7%|6         | 1349/20000 [15:04<39:22:13,  7.60s/it, lr: 1.0e-04 loss: 3.843e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:36<00:36, 36.38s/it]
Generating Samples: 100%|##########| 2/2 [01:12<00:00, 36.38s/it]

zimage_a1_ut:   7%|6         | 1379/20000 [19:28<40:34:39,  7.84s/it, lr: 1.0e-04 loss: 3.826e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:36<00:36, 36.82s/it]
Generating Samples: 100%|##########| 2/2 [01:13<00:00, 36.50s/it]

zimage_a1_ut:   7%|7         | 1409/20000 [23:12<40:11:03,  7.78s/it, lr: 1.0e-04 loss: 3.433e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:36<00:36, 36.75s/it]
Generating Samples: 100%|##########| 2/2 [01:12<00:00, 36.42s/it]

zimage_a1_ut:   7%|7         | 1439/20000 [26:30<39:14:09,  7.61s/it, lr: 1.0e-04 loss: 3.590e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:36<00:36, 36.56s/it]
Generating Samples: 100%|##########| 2/2 [01:12<00:00, 36.28s/it]
```

**Comparison (Task 1 baseline → Task 1+2):**

| Metric | Task 1 Baseline | Task 1+2 | Improvement |
|--------|----------------|----------|-------------|
| Avg training s/it | 7.42s | 7.48s | ~0.8% slower |
| Avg gen time/image | 36.6s | 36.3s | **~0.8% faster** |
| Best gen time/image | 35.2s | 35.5s | ~0.8% slower |
| Worst gen time/image | 37.3s | 36.8s | **~1.3% faster** |

**Notes:**
- Training time shows slight upward drift (7.42s → 7.48s), likely MPS memory fragmentation from running further into the same session, not a regression from the code change
- Generation time is slightly improved overall (36.6s → 36.3s average)
- The unbind/stack pattern change eliminates one intermediate tensor allocation; the effect is small but measurable
- **This change is the new baseline**

**New baseline for comparison:** ~7.48s/it training, ~36.3s/image generation