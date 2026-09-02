---
agent: 'agent'
description: 'Analyze codebase processing loop'
---

Analyze the model's processing loop and propose speed optimizations for both **training** and **image generation**. 

The user provide the model name; if missing, ask.
Before suggesting changes, check `docs/code-optimization/archive/<model>/set-N` to avoid already-tested ideas.

## Benchmark Ownership & Limits

- The user runs all benchmarks.
- You must **never**:
  - run benchmarks,
  - simulate benchmarks,
  - fabricate benchmark results.
- Your role:
  - propose code changes,
  - explain how to benchmark them,
  - wait for real logs from the user.

## Focus Areas

Look for:
1. **Excessive CPU to GPU transfers**
2. Replace **outdated CUDA/MPS patterns** with modern equivalents

## Constraints

- Categorize each change as:
  - **Simple:** 1–5 lines
  - **Moderate:** 6–10 lines
  - **Complex:** 11–20 lines
- Change ≤20 lines per optimization.
- **No rewrites**—surgical edits only.
- Provide the **Top 5 highest-impact, lowest-effort** optimizations.

## Per-optimization workflow

For each optimization:

1. Show the modified code (≤20 lines changed).
2. Describe how to run unit tests (user runs them).
3. Describe how to run benchmarks using the protocol below (user runs them).
4. Document the change in `docs/code-optimization/implementation-proposal-change-N.md` 
5. Update `docs/code-optimization/current-state.md` with:
   - status: `PROPOSED`, `COMPLETED`, `REVERTED`, or `USER DECISION`;
   - a short summary of the intended impact.

The user will:
- run tests and benchmarks,
- archive completed change files to `docs/code-optimization/archive/<model>/set-N`.
   
## Benchmark protocol (user-run)

- Default: ≥3 epochs × 30 steps (Krea: 6 epochs × 30 steps).
- Generate 4 images per epoch.

Collect:

- training time per iteration (s/it),
- sample generation time per image.

### Metric interpretation

- Use **bottom-out s/it** (minimum over the run), not early warm-up values.
- Compare runs with the **same dataset mix**, especially for mixed resolutions.
- A change is “faster” only if bottom-out training and/or sampling time improves.

## Validation

## Validation rules

- **Faster:** keep.
- **Slower:** revert.
- **Negligible (±1–2%):** present data and ask the user; do not decide unilaterally.

When explaining metrics, remember:

- Progress-bar `s/it` is a **cumulative average** since training start.
- Compare minimum `s/it` and sample times between runs, not just end slices.
- Mixed-resolution datasets must be compared against runs with the same mix.

## State files

Treat `docs/code-optimization/current-state.md` as the **single source of truth**:

- pending and completed changes with status,
- current best metrics,
- benchmark tables and comparisons (vs baseline and previous sets),
- baseline variation analysis.

When starting a new set:

1. Read `current-state.md` to understand current metrics and history.
2. Add new proposed changes to `current-state.md`.
3. Update status and results there after each benchmark.
4. Draft detailed proposals in `docs/code-optimization/implementation-proposal-change-N.md`.

**Do not create separate results files.** All benchmark data should be recorded in the 'implementation-proposal-change-N.md' files.

**Once a change is completed, archive the proposal file to `docs/code-optimization/archive/<model>/set-N/` and update `current-state.md` baseline results if they improved.

## Test & result recording

Use the benchmark protocol above. The user will provide logs similar to:

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

Summarize each run in a table with:
- epoch
- steps
- total time
- average training time
- sample 1 time
- sample 2 time
- sample 3 time
- sample 4 time

Record:

- **Baseline** metrics before any change.
- **Per-change** metrics, comparison vs baseline and previous change.
- Status: ✅ `COMPLETED` or ⚠️ `REVERTED`, plus a short result summary.

## Iteration rules

- Test each change individually.
- Use benchmark results to decide impact.
- If improvement is small, you may suggest minor follow-up tweaks, then move on.
- If improvement is clear, you may suggest small additional tweaks in the same area, then move on.

## User checklist (pre-next change)

Before proposing the next optimization, confirm the user has:

- [ ] Checked in the code on a new branch.
- [ ] Pushed the change to their forked git repo.
- [ ] Manually tested the implemented change.