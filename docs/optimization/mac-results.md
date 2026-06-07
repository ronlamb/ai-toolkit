# Chroma Model Optimization Results for MPS - on M5 Max - 128GB memory

Process is run for a step size of 30, for 3 epochs and generating images for 4 steps, due to slowness on macos.

## Baseline (Before Any Changes)
```
Training:   0%|          | 29/21300 [06:10<15:00:37,  12.34/it, lr: 1.0e-04 loss: 9.570e-01]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:59<00:59, 59.52s/it]
            Generating Samples: 100%|##########| 2/2 [01:59<00:00, 59.73s/it]
```

**Note**: This is the initial baseline before any MPS compatibility fixes or optimizations were applied. All subsequent MPS-specific optimizations should be compared against this baseline.