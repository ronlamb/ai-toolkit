It finally worked.

Here are the initial times, for a baseline.  We can add them to the mac-results.md file for validation of later changes.

```
7%|6         | 209/3000 [05:52<9:26:11, 12.17s/it, lr: 1.0e-04 loss: 5.225e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]DEBUG pipeline.py: device=mps, device.type=mpsDEBUG pipeline.py: device=mps, device.type=mps
Generating Samples:  50%|#####     | 1/2 [01:00<01:00, 60.18s/it]DEBUG pipeline.py: device=mps, device.type=mpsDEBUG pipeline.py: device=mps, device.type=mps
Generating Samples: 100%|##########| 2/2 [02:00<00:00, 60.38s/it]

8%|7         | 239/3000 [12:19<9:36:40, 12.53s/it, lr: 1.0e-04 loss: 3.282e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]DEBUG pipeline.py: device=mps, device.type=mpsDEBUG pipeline.py: device=mps, device.type=mps
Generating Samples:  50%|#####     | 1/2 [01:02<01:02, 62.51s/it]DEBUG pipeline.py: device=mps, device.type=mpsDEBUG pipeline.py: device=mps, device.type=mps
Generating Samples: 100%|##########| 2/2 [02:05<00:00, 62.53s/it]
```

Now that it's working, please remove all of the DEBUG statements you added.