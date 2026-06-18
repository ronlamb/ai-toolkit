# Chroma Model Optimization Results for MPS - on M5 Max - 128GB memory

Process is run for a step size of 30, for 3 epochs and generating images for 4 steps, due to slowness on macos.

## Baseline Times

Below are the baseline times before all changes described here.  These times will be used help
determine whether a code change is bad or not.

```
Training:   0%|          | 29/24000 [05:33<76:37:59, 11.51s/it, lr: 1.0e-04 loss: 5.397e-01]
Sampling:   Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:52<00:52, 52.15s/it]
            Generating Samples: 100%|##########| 2/2 [01:45<00:00, 52.83s/it]

Training:   0%|          | 59/24000 [11:32<78:03:24, 11.74s/it, lr: 1.0e-04 loss: 2.708e-01]
            Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:55<00:55, 55.25s/it]
            Generating Samples: 100%|##########| 2/2 [01:50<00:00, 55.46s/it]

Training:   0%|          | 89/24000 [17:36<78:49:14, 11.87s/it, lr: 1.0e-04 loss: 3.594e-01]
Sampling:   Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:55<00:55, 55.26s/it]
            Generating Samples: 100%|##########| 2/2 [01:50<00:00, 55.39s/it]

Training:   0%|          | 119/24000 [23:37<79:01:41, 11.91s/it, lr: 1.0e-04 loss: 2.849e-01]
Sampling:   Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:54<00:54, 54.95s/it]
            Generating Samples: 100%|##########| 2/2 [01:50<00:00, 55.03s/it]

Training:   1%|          | 149/24000 [29:40<79:10:10, 11.95s/it, lr: 1.0e-04 loss: 3.374e-01]
Sampling:   Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:54<00:54, 54.96s/it]
            Generating Samples: 100%|##########| 2/2 [01:50<00:00, 55.04s/it]

Training:   1%|          | 179/24000 [35:39<79:05:42, 11.95s/it, lr: 1.0e-04 loss: 3.181e-01]
Sampling:   Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:54<00:54, 54.81s/it]
            Generating Samples: 100%|##########| 2/2 [01:49<00:00, 55.02s/it]

Training:   1%|          | 209/24000 [41:39<79:02:03, 11.96s/it, lr: 1.0e-04 loss: 3.457e-01]
Sampling:   Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:54<00:54, 54.87s/it]
            Generating Samples: 100%|##########| 2/2 [01:50<00:00, 55.34s/it]

Training:   1%|          | 239/24000 [47:42<79:03:03, 11.98s/it, lr: 1.0e-04 loss: 3.115e-01]
Sampling:   Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:54<00:54, 54.96s/it]
            Generating Samples: 100%|##########| 2/2 [01:50<00:00, 55.05s/it]
```

