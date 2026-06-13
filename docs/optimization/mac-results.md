# Chroma Model Optimization Results for MPS - on M5 Max - 128GB memory

Process is run for a step size of 30, for 3 epochs and generating images for 4 steps, due to slowness on macos.

## Baseline Times

Below are the baseline times before all changes described here.  These times will be used help
determine whether a code change is bad or not.

```
Training:   0%|          | 29/6000 [05:45<19:46:50, 11.93s/it, lr: 1.0e-04 loss: 4.103e-01]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:56<00:56, 56.42s/it]
            Generating Samples: 100%|##########| 2/2 [01:53<00:00, 56.75s/it]

Training:   1%|          | 59/6000 [11:39<19:34:06, 11.86s/it, lr: 1.0e-04 loss: 2.875e-01]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:56<00:56, 56.42s/it]
            Generating Samples: 100%|##########| 2/2 [01:53<00:00, 56.62s/it]

Training:   1%|1         | 89/6000 [17:35<19:28:13, 11.86s/it, lr: 1.0e-04 loss: 2.951e-01]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [00:57<00:57, 57.75s/it]
            Generating Samples: 100%|##########| 2/2 [01:55<00:00, 57.91s/it]
```

