# Codebase Analysis

Please analyze the processing loop for the given model and identify ways to improve its speed in both the training loop and the image generation loop.

## Focus Areas

Look for:
1. **Excessive CPU to GPU copies** - Data transfers that could be eliminated or reduced
2. **Not utilizing most up to date CUDA or MPS capabilities** - Outdated patterns that miss GPU optimizations

## Requirements

- **Categorize by complexity**: Simple (1-5 lines), Moderate (6-10 lines), Complex (11-20 lines)
- **Line Limit per function**: ≤20 lines changed
- **No rewrites**: Only surgical, incremental improvements
- **Top 5 optimizations**: Highest impact, lowest effort changes

## Process

For each optimization opportunity:
1. Write the optimized code (≤20 lines)
2. Run unit tests
3. Benchmark speed using the benchmark protocol below

## Benchmark Protocol

### User's Test

Run 3 epochs of 30 steps each and generate 4 images

### Metrics to Collect

1. **Training time per iteration**: `X.XXs/it` from progress bar
2. **Sample generation time**: Time per image from "Generating Samples" progress

## Validation

For each change, evaluate benchmark results:

- If speed improves → keep the change
- If speed does not improve → revert the change

## Important Notes
- Test each change individually
- User manually tests after implementation
- Commit and push changes before the next optimization

## State files

Use the following two files:

- **`implementation-checklist.md`** – Track progress of each change
- **`results.md`** – Record detailed results for each optimization

Update both after each change. 
Reset them only when starting a new optimization session.

---

## Test Protocol

Use the benchmark protocol:
- 3 epochs of 30 steps each 
- 4 generated images.
- User will run this and provide logs.

### Training Time

Each epoch’s training time entry will look similar to:

```
<some_model>:   0%|          | 29/30000 [02:17<39:26:19,  4.74s/it, lr: 3.0e-04 loss: 6.365e-02]
```

### Samples

Followed by sampling entries such as:

```
ex:
Generating Samples:   0%|          | 0/4 [00:00<?, ?it/s]
Generating Samples:  25%|#1        | 1/4 [01:07<09:00, 67.55s/it]
Generating Samples:  50%|##2       | 2/4 [02:14<07:49, 67.07s/it]
Generating Samples:  75%|###3      | 3/4 [03:21<06:41, 66.94s/it]
Generating Samples: 100%|####4     | 4/4 [04:27<05:33, 66.63s/it]
```

### Summarize Data

Summarize results in a table with columns:

- epoch
- steps
- total time
- average training time
- sample 1 time
- sample 2 time
- sample 3 time
- sample 4 time

## Baseline Results

Before any change, create a baseline results table using the format in **Summarize Data**.

## Change Results

After each change, create a new results table (same format) and compare it to:

- the baseline
- the previous change

## Validate Each Test

Use benchmark results to confirm whether the change improved speed.

## If There Is Little to No Speed Improvement

If a change shows little or no improvement:

- Propose alternative optimizations for the same area
- If no viable alternatives exist, revert and move on to a different opportunity

## If There Are Speed Improvements

If a change improves speed:

- Optionally explore small, additional tweaks in the same area
- Once exhausted, proceed to the next optimization opportunity

## Before Doing the Next Change Verify That the User:

- [ ] Checked in the code in a new branch
- [ ] Pushed the change to his forked git repo

## Check list

Make sure the user:

- Manually tests each implemented change
- Commits and pushes changes before starting the next optimization
