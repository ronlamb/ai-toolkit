# Chroma Model Optimization Results for MPS - on M5 Max - 128GB memory

Process is run for a step size of 30, for 3 epochs and generating images for 4 steps, due to slowness on macos.

## Baseline (Before Any Changes) - first 3 epochs
```
Training:   1%|          | 29/3000 [05:59<10:14:08, 12.40s/it, lr: 1.0e-04 loss: 4.543e-01]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:59<00:59, 59.22s/it]
            Generating Samples: 100%|##########| 2/2 [01:58<00:00, 59.42s/it]

Training:   2%|1         | 59/3000 [12:10<10:06:37, 12.38s/it, lr: 1.0e-04 loss: 2.865e-01]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:58<00:58, 58.72s/it]
            Generating Samples: 100%|##########| 2/2 [01:57<00:00, 58.80s/it]

Training: 3%|2         | 89/3000 [18:19<9:59:13, 12.35s/it, lr: 1.0e-04 loss: 2.840e-01]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:59<00:59, 59.04s/it]
            Generating Samples: 100%|##########| 2/2 [01:58<00:00, 59.08s/it]  
```

**Note**: This is the initial baseline before any MPS compatibility fixes or optimizations were applied. All subsequent MPS-specific optimizations should be compared against this baseline.