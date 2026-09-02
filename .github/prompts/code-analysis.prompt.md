---
agent: 'agent'
description: 'Analyze codebase processing loop'
---

Analyze the model's processing loop and propose speed optimizations for both **training** and **image generation**. 

The user provide the model name; if missing, ask.
Before suggesting changes, check `docs/code-optimization/archive/<model>/set-N` to avoid repeating previously tested optimizations.

## Benchmark Ownership & Limits

- The user runs all benchmarks.
- You must **never** run, simulate, or fabricate benchmarks results.
- Your role is to propose code changes, explain how to test them, and- wait for real logs.

## Focus On

1. Eliminating **unnecessary CPU to GPU transfers**
2. Updating **outdated CUDA/MPS patterns** to modern equivalents

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

1. Show the modified code (≤20 changed lines).
2. Describe unit tests (user runs them).
3. Describe benchmark steps (user runs them).
4. Write a detailed proposal in `docs/code-optimization/implementation-proposal-change-N.md` 
5. Add a short entyr to `docs/code-optimization/current-state.md` with:
   - status (`PROPOSED`, `COMPLETED`, `REVERTED`, or `USER DECISION`);
   - a 1-3 sentence  summary of the intended impact.

The user will:
- run tests and benchmarks,
- archive completed change files to `docs/code-optimization/archive/<model>/set-N`.
   
## Benchmark protocol (user-run)

- Default: ≥3 epochs × 30 steps (Krea: 6×30 steps).
- Generate 4 images per epoch.

Collect:

- training time per iteration (s/it),
- sample generation time per image.
  - Use **bottom-out s/it** (minimum over the run), not early warm-up values.

## Validation

## Validation rules
- Faster → keep
- Slower → revert
- Negligible (±1–2%) → present data and let the user decide
  - Compare bottom-out s/it and sample times only.

## State files

Treat `docs/code-optimization/current-state.md` is the single source of truth for status:

- pending and completed changes,
- current best metrics,
- short summaries.

Each propasal file `implementation-proposal-change-N.md` contains:

- full benchmark summaries
- comparisons
- detailed analysis

Archive completed proposal files under `docs/code-optimization/archive/<model>/set-N/`

**Do not create separate results files.** All benchmark data should be recorded in the 'implementation-proposal-change-N.md' files.

## Test & result recording

Record results in the proposal file:
- baseline metrics
- per-change metrics
- comparisons vs baseline and previous change

Update `current-state.md` with status + short summary only.

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
- If improvement is small, you may suggest minor follow-up tweaks.
- If improvement is clear, finish that area and move on.

## User checklist (pre-next change)

Before proposing the next optimization, confirm the user has:

- [ ] Checked in the code on a new branch.
- [ ] Pushed the change to their forked git repo.
- [ ] Manually tested the implemented change.