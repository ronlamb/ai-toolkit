# Z-Image base Model Optimization Results for MPS - on M5 Max - 128GB memory

Process is run for a step size of 30, for 8 epochs and generating 2 images for 5 steps to get a base generation time.

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

This section will contain the new baseline as of the best change.

Initially not set.