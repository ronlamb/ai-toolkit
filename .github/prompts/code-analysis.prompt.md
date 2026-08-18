---
agent: 'agent'
description: 'Analyze codebase processing loop'
---

Please analyze the processing loop for the given model and identify ways to improve its speed in both the training loop and the image generation loop.

The user will give you the model name being optimized.  If not provided, ask.
Look for previous code optimizations that are documented in the `docs/code-optimization/archive/<model>/set-N` folders, so you can avoid suggesting optimizations that have already been implemented and tested.

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
4. Document the change in:
   - `docs/code-optimization/implementation-proposal-change-N.md` 
5. Update `docs/code-optimization/current-state.md` to mark the change as 
   - PROPOSED (if not yet tested)
   - ✅ COMPLETED (if speed improved)
   - ⚠️ REVERTED (if no measurable improvement)
6. Once all optimizations are complete, the user will move all the change files to the `docs/code-optimization/archive/<model>/set-N` folder.

## Benchmark Protocol

### User's Test

Run 3 (or more) epochs of 30 steps each and generate 4 images

### Metrics to Collect

1. **Training time per iteration**: `X.XXs/it` from progress bar
2. **Sample generation time**: Time per image from "Generating Samples" progress

## Validation

- If speed improves → keep the change
- If speed does not improve → revert the change

## Important Notes
- Test each change individually
- User manually tests after implementation
- Commit and push changes before the next optimization

## State files

Use **`docs/code-optimization/current-state.md`** as the single source of truth. This file contains:
- All pending and completed changes with status
- Current best metrics (training time, sample generation)
- Full benchmark results (training time per step, sample generation per checkpoint)
- Comparison tables (vs previous sets, vs original baseline)
- Baseline variation analysis

When starting a new set of optimizations:
1. Read `current-state.md` to understand the current state and metrics
2. Add new changes to `current-state.md` as they are proposed
3. Update status and actual results in `current-state.md` after testing
4. Create detailed implementation proposals in `docs/code-optimization/implementation-proposal-change-N.md`

**Do not create separate results files.** All benchmark data should be recorded in `current-state.md`.

## Test Protocol

Use the benchmark protocol:
- 3 or more epochs of 30 steps each 
- 4 generated images.

User will run this and provide logs.

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

## Recording Results

All benchmark results go into **`docs/code-optimization/current-state.md`**.

### Baseline Results

Before any change, record baseline metrics in `current-state.md`:
- Training time per epoch
- Sample generation times
- Stable/bottom-out metrics

### Change Results

After each change, update `current-state.md` with:
- New benchmark data
- Comparison against baseline and previous change
- Status (✅ COMPLETED or ⚠️ REVERTED)
- Actual results summary

Compare against:
- The baseline (original unoptimized metrics)
- The previous change

## Validate Each Test

Use benchmark results to confirm whether the change improved speed.

## If There Is Little to No Speed Improvement

- Optionally explore small tweaks
- Then proceed to next optimization

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
