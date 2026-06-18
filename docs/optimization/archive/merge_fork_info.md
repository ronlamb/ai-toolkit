# Summary

I merged the upstream main branch of https://github.com/ostris/ai-toolkit into the main branch of my fork:
https://github.com/ronlamb/ai-toolkit

The merge corresponds to commit 6c3b826.
The branch currently checked out is mac_gpu_util.

## Get merge history

Here is the git merge history:

```
commit 9ea67341eed6a1521acf8ae93cd6351f2f22bdff
Merge: fe3b8c2 232426f
Author: Ron Lamb <ron.lamb@gmail.com>
Date:   Sun Jun 14 15:54:45 2026 -0400
    Merge branch 'mac_mps_improvements' into mac_gpu_util

commit 232426fde20645d059ef35d2a26766ee950f88e7
Merge: 98ab293 0a302f3
Author: Ron Lamb <ron.lamb@gmail.com>
Date:   Sun Jun 14 15:54:00 2026 -0400
    Merge branch 'main' into mac_mps_improvements

commit 891adf955b68441afaf1c1d490dec921273785dd
Merge: c9c3878 a86238d
Author: Ron Lamb <ron.lamb@gmail.com>
Date:   Sat Jun 13 18:30:58 2026 -0400
    Merge branch 'mac_gpu_improvements' into mac_gpu_util

commit a589aea744ef3b6b207f5b7bdac6280d53d822ad
Merge: 3f6e76b 0a302f3
Author: ronlamb <ron.lamb@gmail.com>
Date:   Sat Jun 13 16:20:59 2026 -0400
    updates

commit 0a302f3fe843555862df1fbe538801c03d38841f
Merge: 90a2084 6c3b826
Author: ronlamb <30437503+ronlamb@users.noreply.github.com>
Date:   Sat Jun 13 13:46:28 2026 -0400
    Merge pull request #2 from ostris/main
``` 

# I am seeing a bit of degredation.  Here is how it compares from before and after the merge

## Before Merge

- Per‑step times (first 30 steps):
    - Steps 1–10: ~10.2–10.3 s/it
    - Steps 11–20: ~10.9–11.1 s/it
    - Steps 21–30: ~11.6–11.9 s/it

### Epoch 1 log 

```
0%|          | 29/6000 [05:45<19:46:50, 11.93s/it, lr: 1.0e-04 loss: 4.103e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:56<00:56, 56.42s/it]
Generating Samples: 100%|##########| 2/2 [01:53<00:00, 56.75s/it]
```

### Epoch 2 log

```
x1%|          | 59/6000 [11:39<19:34:06, 11.86s/it, lr: 1.0e-04 loss: 2.875e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:56<00:56, 56.42s/it]
Generating Samples: 100%|##########| 2/2 [01:53<00:00, 56.62s/it]
```

### Epoch 3 log

```
1%|1         | 89/6000 [17:35<19:28:13, 11.86s/it, lr: 1.0e-04 loss: 2.951e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:57<00:57, 57.75s/it]
Generating Samples: 100%|##########| 2/2 [01:55<00:00, 57.91s/it]
```

## After merge
- Per‑step times (first 30 steps):
  - **Steps 1–10**: ~11.1–11.3 s/it
  - **Steps 11–20**: ~11.9–12.3 s/it
  - **Steps 21–30**: ~11.9–12.3 s/it

### Epoch 1 log 

```
5%|5         | 329/6000 [06:06<19:55:29, 12.65s/it, lr: 1.0e-04 loss: 3.863e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:57<00:57, 57.19s/it]
Generating Samples: 100%|##########| 2/2 [01:55<00:00, 57.69s/it]
```

### Epoch 2 log 

```
6%|5         | 359/6000 [12:34<20:02:20, 12.79s/it, lr: 1.0e-04 loss: 2.591e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:58<00:58, 58.71s/it]
Generating Samples: 100%|##########| 2/2 [01:57<00:00, 58.89s/it]
```

### Epoch 3 log 

```
chroma_a1:   6%|6         | 389/6000 [18:59<19:57:03, 12.80s/it, lr: 1.0e-04 loss: 3.127e-01]
Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Samples:  50%|#####     | 1/2 [00:59<00:59, 59.12s/it]
Generating Samples: 100%|##########| 2/2 [01:58<00:00, 59.09s/it]
```

# Code inspection request

**Diff to inspect**: `chroma_differences.md` (changes introduced by commit 6c3b826 into main).
**Scope**: all files listed in chroma_modules.txt.

# Tasks

- **Identify** any changes that could impact training speed or device utilization.
- **Do not assume a change is harmless** — treat all diffs as potentially relevant.
- **Pay special attention to**:
    - **Removed or altered MPS‑specific code** (Apple GPU / mps paths).
    - **CUDA‑specific code** that might now be preferred or incorrectly selected over MPS.
    - Any logic that changes:
        - device selection
        - tensor placement
        - synchronization or blocking behavior
        - dataloader or prefetch behavior.