I misread the comparisons.  I looked at the first run with 1024x1024 images, this was with 768x768.  Speed is slightly faster than the about 0.025 sec/it from the original.  While it might be within error, it was consistent.  
```
First of three epochs new code:
1%|          | 199/20000 [07:45<12:51:17,  2.34s/it, lr: 8.0e-05 loss: 5.187e-01]
2%|1         | 399/20000 [15:21<12:34:40,  2.31s/it, lr: 8.0e-05 loss: 5.382e-01]
3%|2         | 599/20000 [23:13<12:32:01,  2.33s/it, lr: 8.0e-05 loss: 5.112e-01]


vs base
1%|          | 199/20000 [07:48<12:56:53,  2.35s/it, lr: 8.0e-05 loss: 5.444e-01]
2%|1         | 399/20000 [15:32<12:43:46,  2.34s/it, lr: 8.0e-05 loss: 5.037e-01]
3%|2         | 599/20000 [23:41<12:47:06,  2.37s/it, lr: 8.0e-05 loss: 5.364e-01]
One thing I did notice, the samples for each change seem to be converging better.  
Can you explain why the convergeance seems more smoother.
```

```
The numbers look slightly better.  In the base run it bottomed out at 2.29 sec/it at epoch 24

24%|##3       | 4799/20000 [3:03:32<9:41:21,  2.29s/it, lr: 8.0e-05 loss: 5.016e-01]

And in the new 2.26s/it at epoch 20

20%|#9        | 3999/20000 [2:30:37<10:02:39,  2.26s/it, lr: 8.0e-05 loss: 4.085e-01]

And step 24 arrived 3 minutes earlier
24%|##3       | 4799/20000 [3:00:30<9:31:46,  2.26s/it, lr: 8.0e-05 loss: 3.665e-01]

Images also steadily improved throughout, where in the old I had to pull it back and restart.  Memory usage was also about the same.
```

--- Final ---
First of three epochs new code:
1%|          | 199/20000 [07:36<12:37:31,  2.30s/it, lr: 8.0e-05 loss: 5.249e-01]
2%|1         | 399/20000 [15:11<12:26:12,  2.28s/it, lr: 8.0e-05 loss: 3.807e-01]
3%|2         | 599/20000 [22:45<12:17:07,  2.28s/it, lr: 8.0e-05 loss: 3.035e-01]

vs base:
1%|          | 199/20000 [07:48<12:56:53,  2.35s/it, lr: 8.0e-05 loss: 5.444e-01]
2%|1         | 399/20000 [15:32<12:43:46,  2.34s/it, lr: 8.0e-05 loss: 5.037e-01]
3%|2         | 599/20000 [23:41<12:47:06,  2.37s/it, lr: 8.0e-05 loss: 5.364e-01]

Which is now a 0.05 to 0.06 increase from the original code.
```

Update Optimizer Utils
```
First of three epochs new code:
1%|          | 199/20000 [07:36<12:37:15,  2.29s/it, lr: 8.0e-05 loss: 3.918e-01]
2%|1         | 399/20000 [15:10<12:25:23,  2.28s/it, lr: 8.0e-05 loss: 3.827e-01]
3%|2         | 599/20000 [22:41<12:14:42,  2.27s/it, lr: 8.0e-05 loss: 6.864e-01]

Base:
1%|          | 199/20000 [07:36<12:37:31,  2.30s/it, lr: 8.0e-05 loss: 5.249e-01]
2%|1         | 399/20000 [15:11<12:26:12,  2.28s/it, lr: 8.0e-05 loss: 3.807e-01]
3%|2         | 599/20000 [22:45<12:17:07,  2.28s/it, lr: 8.0e-05 loss: 3.035e-01]

First of three epochs of first set of chamges, before this one  bottomed out at epoch 7 at 2.25:
1%|          | 199/20000 [07:36<12:37:15,  2.29s/it, lr: 8.0e-05 loss: 3.918e-01]
2%|1         | 399/20000 [15:10<12:25:23,  2.28s/it, lr: 8.0e-05 loss: 3.827e-01]
3%|2         | 599/20000 [22:41<12:14:42,  2.27s/it, lr: 8.0e-05 loss: 6.864e-01]

Current version bottomed out at step 6, just as you expect, s/it was a bit higher, but within error.
1%|          | 199/20000 [07:41<12:45:22,  2.32s/it, lr: 8.0e-05 loss: 4.407e-01]
2%|1         | 399/20000 [15:17<12:30:50,  2.30s/it, lr: 8.0e-05 loss: 4.867e-01]
3%|2         | 599/20000 [22:51<12:20:15,  2.29s/it, lr: 8.0e-05 loss: 3.831e-01]
4%|3         | 799/20000 [30:26<12:11:30,  2.29s/it, lr: 8.0e-05 loss: 5.977e-01]
5%|4         | 999/20000 [37:56<12:01:34,  2.28s/it, lr: 8.0e-05 loss: 4.791e-01]
6%|5         | 1199/20000 [45:25<11:52:10,  2.27s/it, lr: 8.0e-05 loss: 4.322e-01]
7%|6         | 1399/20000 [52:53<11:43:20,  2.27s/it, lr: 8.0e-05 loss: 3.008e-01]

The main difference I noticed is that in this version the Lora, which is a character lora, converged a lot quicker.  
Epoch 3: 3 of the 7 sample images were close, missing fine details but face was spot on.  Old run it took until epoch 8 for this to occur
Epoch 4: 4 of the images were close, again missing fine details, but overall matched.  Took until epoch 11 for this to occur
Epoch 5: Same number of images close, but more accurate. 
Epoch 6: five of the 7 images were close, about the same accuracy matched.  In the old run it took until epoch 15.
Epoch 7: 6 of the 7 images, missing fine details on medium and full body, but closeup were spot on, minute details were still off, but it didn't hit that until epoch 20 in the old, and it went in and out for a bit.
Epoch 8: Still 6 good images.  Seems to be converging much better at this point, more detail showing up.  Before it would cycle a bit. Also the one image that was typically off is now very close, which took until epoch 30-35 to get there.

Also the smoothed loss at 10 steps granularity the graph stabilized at .43 or so, before it was around .5, at 20 steps it was .41, before that it was also around .5

----

1%|          | 199/20000 [07:37<12:38:10,  2.30s/it, lr: 8.0e-05 loss: 3.191e-01]
2%|1         | 399/20000 [15:10<12:25:36,  2.28s/it, lr: 8.0e-05 loss: 5.944e-01]
3%|2         | 599/20000 [22:42<12:15:14,  2.27s/it, lr: 8.0e-05 loss: 5.078e-01]
4%|3         | 799/20000 [30:26<12:11:27,  2.29s/it, lr: 8.0e-05 loss: 3.648e-01]
5%|4         | 999/20000 [38:00<12:02:48,  2.28s/it, lr: 8.0e-05 loss: 3.603e-01]
6%|5         | 1199/20000 [45:31<11:53:45,  2.28s/it, lr: 8.0e-05 loss: 3.259e-01]
7%|6         | 1399/20000 [53:01<11:44:57,  2.27s/it, lr: 8.0e-05 loss: 5.029e-01]
8%|7         | 1599/20000 [1:00:29<11:36:07,  2.27s/it, lr: 8.0e-05 loss: 4.259e-01]
9%|8         | 1799/20000 [1:07:58<11:27:39,  2.27s/it, lr: 8.0e-05 loss: 4.056e-01]
10%|9         | 1999/20000 [1:15:26<11:19:21,  2.26s/it, lr: 8.0e-05 loss: 3.121e-01]
11%|#         | 2199/20000 [1:22:55<11:11:13,  2.26s/it, lr: 8.0e-05 loss: 5.898e-01]
12%|#1        | 2399/20000 [1:30:23<11:03:08,  2.26s/it, lr: 8.0e-05 loss: 6.068e-01]
13%|#2        | 2599/20000 [1:37:51<10:55:11,  2.26s/it, lr: 8.0e-05 loss: 4.587e-01]
14%|#3        | 2799/20000 [1:45:19<10:47:18,  2.26s/it, lr: 8.0e-05 loss: 3.392e-01]
15%|#4        | 2999/20000 [1:52:48<10:39:29,  2.26s/it, lr: 8.0e-05 loss: 5.580e-01]
16%|#5        | 3199/20000 [2:00:16<10:31:43,  2.26s/it, lr: 8.0e-05 loss: 4.609e-01]
17%|#6        | 3399/20000 [2:07:45<10:23:59,  2.26s/it, lr: 8.0e-05 loss: 2.561e-01]
18%|#7        | 3599/20000 [2:15:13<10:16:15,  2.25s/it, lr: 8.0e-05 loss: 5.969e-01]

---- update_from_fp32_ added

1%|          | 199/20000 [07:37<12:38:43,  2.30s/it, lr: 8.0e-05 loss: 4.862e-01]
2%|1         | 399/20000 [15:10<12:25:51,  2.28s/it, lr: 8.0e-05 loss: 2.972e-01]
3%|2         | 599/20000 [22:45<12:17:16,  2.28s/it, lr: 8.0e-05 loss: 3.378e-01]
4%|3         | 799/20000 [30:20<12:09:11,  2.28s/it, lr: 8.0e-05 loss: 2.725e-01]
5%|4         | 999/20000 [37:54<12:00:58,  2.28s/it, lr: 8.0e-05 loss: 4.062e-01]
6%|5         | 1199/20000 [45:28<11:53:04,  2.28s/it, lr: 8.0e-05 loss: 5.248e-01]
7%|6         | 1399/20000 [53:02<11:45:08,  2.27s/it, lr: 8.0e-05 loss: 5.117e-01]
8%|7         | 1599/20000 [1:00:35<11:37:18,  2.27s/it, lr: 8.0e-05 loss: 4.194e-01]
9%|8         | 1799/20000 [1:08:09<11:29:34,  2.27s/it, lr: 8.0e-05 loss: 4.072e-01]
10%|9         | 1999/20000 [1:15:43<11:21:51,  2.27s/it, lr: 8.0e-05 loss: 4.903e-01]
11%|#         | 2199/20000 [1:23:17<11:14:14,  2.27s/it, lr: 8.0e-05 loss: 3.951e-01]
12%|#1        | 2399/20000 [1:30:51<11:06:38,  2.27s/it, lr: 8.0e-05 loss: 3.341e-01]

--- fixed

1%|          | 199/20000 [07:36<12:37:46,  2.30s/it, lr: 8.0e-05 loss: 5.687e-01]
2%|1         | 399/20000 [15:13<12:27:45,  2.29s/it, lr: 8.0e-05 loss: 5.250e-01]
3%|2         | 599/20000 [22:45<12:16:54,  2.28s/it, lr: 8.0e-05 loss: 4.680e-01]
4%|3         | 799/20000 [30:16<12:07:24,  2.27s/it, lr: 8.0e-05 loss: 3.978e-01]
5%|4         | 999/20000 [37:45<11:58:04,  2.27s/it, lr: 8.0e-05 loss: 5.716e-01]
6%|5         | 1199/20000 [45:13<11:49:13,  2.26s/it, lr: 8.0e-05 loss: 2.693e-01]
7%|6         | 1399/20000 [52:42<11:40:48,  2.26s/it, lr: 8.0e-05 loss: 5.322e-01]
8%|7         | 1599/20000 [1:00:11<11:32:42,  2.26s/it, lr: 8.0e-05 loss: 2.758e-01]
9%|8         | 1799/20000 [1:07:41<11:24:46,  2.26s/it, lr: 8.0e-05 loss: 4.430e-01]
10%|9         | 1999/20000 [1:15:10<11:16:56,  2.26s/it, lr: 8.0e-05 loss: 3.980e-01]
11%|#         | 2199/20000 [1:22:39<11:09:06,  2.26s/it, lr: 8.0e-05 loss: 5.521e-01]
12%|#1        | 2399/20000 [1:30:08<11:01:20,  2.25s/it, lr: 8.0e-05 loss: 4.086e-01]
13%|#2        | 2599/20000 [1:37:37<10:53:35,  2.25s/it, lr: 8.0e-05 loss: 3.390e-01]
14%|#3        | 2799/20000 [1:45:06<10:45:54,  2.25s/it, lr: 8.0e-05 loss: 4.316e-01]
15%|#4        | 2999/20000 [1:52:44<10:39:06,  2.26s/it, lr: 8.0e-05 loss: 3.105e-01]
