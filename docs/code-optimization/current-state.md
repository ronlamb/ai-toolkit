# Krea2 Pipeline Optimization — Current State

**This file is the single source of truth for the current best metrics and pending work.**
Per-change details, benchmark tables, and verdicts live in `archive/krea2/set-N/`.
Historical progress, kept/reverted changes, and qualitative validation live in `change-progress.md`.

## Current best metrics (as of change #21, 2026-09-02)

Short benchmark = 6 epochs × 30 steps, 4 images, same dataset throughout (mixes 512×512 and
1024×1024 sets, so compare only same-mix numbers).

| Metric | Value | Where measured |
|--------|-------|----------------|
| Training (short bench) | **~3.08–3.09 s/it** | bottom-out cumulative @ step 179 (#20/#21 runs) |
| Samples (short bench, 1024-mix) | **~64.7 s/image** | epochs 4–6 avg (#16 run; historical best) |
| Full-run bottom-out (training) | **2.86 s/it** | 22 checkpoints, step 3784 (#21-state full runs, pre/post-merge) |
| Full-run bottom-out (samples) | **~62.6–63.7 s/img** | fast-mode checkpoints (#21-state full runs, pre/post-merge) |
| Convergence | **~epoch 20** (~3,440 steps) | +20 pts visual accuracy, ~31% faster convergence vs #10 state |

> Current code stack: #10 + #14 + #16 + #18 + #19 + #20 + #21 (all kept).
> #15 and #17 reverted. See `change-progress.md` for full history.

### Full-run validation (#21 state, post-merge, tested 2026-09-04) — CURRENT BASELINE

172 images/epoch, checkpoint every epoch, 9 samples per checkpoint, run to step 3784.
Run name `anna_bell_sex_krea_ut_2`. Same code state as the pre-merge run below (change #21,
plus the upstream main merge `0dbc7f5`, which is performance-neutral — verified below).
Note: bar elapsed now excludes sampling/saves (upstream progress-bar pause rework `27a03a9`).

| Steps | Cum s/it | Per-step avg (Δ/172) | Samples avg (s/img) | Mode |
|-------|----------|----------------------|---------------------|------|
| 172 | 3.16 | warm-up | 69.8 | — |
| 344 | 3.04 | 2.91 | 67.8 | slow |
| 516 | 3.01 | 2.95 | 68.0 | slow |
| 688 | 2.98 | 2.87 | 64.3 | fast-ish |
| 860 | 2.94 | 2.80 | 65.5 | fast-ish |
| 1032 | 2.91 | 2.76 | 63.4 | fast |
| 1204 | 2.89 | 2.75 | 63.7 | fast |
| 1376 | 2.87 | 2.73 | 63.0 | fast |
| 1548 | 2.86 | 2.80 | 67.8 | mixed |
| 1720 | 2.87 | 2.98 | 67.6 | slow |
| 1892 | 2.86 | 2.71 | 62.8 | fast |
| 2064 | 2.86 | 2.88 | 67.8 | slow |
| 2236 | 2.86 | 2.94 | 67.9 | slow |
| 2408 | 2.87 | 2.97 | 67.8 | slow |
| 2580 | 2.88 | 2.94 | 67.8 | slow |
| 2752 | 2.88 | 2.95 | 67.9 | slow |
| 2924 | 2.89 | 2.96 | 67.9 | slow |
| 3096 | 2.89 | 2.95 | 67.9 | slow |
| 3268 | 2.90 | 2.98 | 67.8 | slow |
| 3440 | 2.90 | 2.94 | 67.9 | slow |
| 3612 | 2.90 | 2.95 | 67.9 | slow |
| 3784 | 2.90 | 2.95 | 67.8 | slow |

Bottom-out cumulative **2.86** (steps 1548–2236); final @ 3784 = 2.90 (late-run slow-mode
streak, not code — fast-mode values match the pre-merge run exactly).

### Previous full-run validation (#21 state, pre-merge, tested 2026-08-30)

172 images/epoch, checkpoint every epoch, 9 samples per checkpoint, run to step 3784.
Run name `anna_bell_sex_krea_ut_2`. 

| Steps | Cum s/it | Per-step avg (Δ/172) | Samples avg (s/img) | Mode |
|-------|----------|----------------------|---------------------|------|
| 172 | 3.19 | warm-up | 66.1 | — |
| 344 | 2.99 | 2.79 | 65.0 | fast-ish |
| 516 | 2.92 | 2.78 | 67.7 | mixed |
| 688 | 2.93 | 2.95 | 67.7 | slow |
| 860 | 2.94 | 2.98 | 67.7 | slow |
| 1032 | 2.94 | 2.94 | 67.7 | slow |
| 1204 | 2.94 | 2.97 | 67.7 | slow |
| 1376 | 2.94 | 2.95 | 67.8 | slow |
| 1548 | 2.95 | 2.96 | 67.7 | slow |
| 1720 | 2.95 | 2.96 | 66.3 | slow |
| 1892 | 2.93 | 2.71 | 62.6 | fast |
| 2064 | 2.91 | 2.73 | 62.9 | fast |
| 2236 | 2.90 | 2.74 | 62.6 | fast |
| 2408 | 2.89 | 2.74 | 62.7 | fast |
| 2580 | 2.88 | 2.73 | 66.0 | mixed |
| 2752 | 2.88 | 2.97 | 67.9 | slow |
| 2924 | 2.88 | 2.94 | 67.8 | slow |
| 3096 | 2.89 | 2.97 | 63.4 | mixed |
| 3268 | 2.88 | 2.74 | 63.5 | fast |
| 3440 | 2.87 | 2.74 | 62.7 | fast |
| 3612 | 2.87 | 2.75 | 62.8 | fast |
| 3784 | **2.86** | 2.74 | 63.6 | fast |

**Mode oscillation**: the run oscillates between fast (~2.74 s/it + ~62.7 s/img) and slow
(~2.96 s/it + ~67.8 s/img) modes. See `gpu-performance-modes.md` for analysis.

### Post-merge vs pre-merge (both #21 state)

| Metric | Pre-merge (08-30) | Post-merge (09-04) | Verdict |
|--------|-------------------|--------------------|---------|
| Bottom-out cumulative (s/it) | **2.86** @ 3784 | **2.86** @ 1548–2236 | identical |
| Fast-mode per-step | 2.71–2.75 | **2.71–2.75** | identical |
| Fast-mode samples (s/img) | 62.6–63.6 | **62.8–63.7** | identical |
| Slow-mode samples (s/img) | 67.7–67.9 | **67.8–67.9** | identical |
| Final cum @ 3784 | 2.86 | 2.90 (+1.4%) | mode luck only |

**Conclusion: the upstream merge (`0dbc7f5`) is performance-neutral.** Both full runs are the
same code state (#21); differences are explained entirely by GPU mode oscillation.
### Comparison vs #10 full-run baseline (2.93 s/it bottom-out, 64.85 s/img)

| Metric | #10 baseline | #16-state run | Delta |
|--------|--------------|---------------|-------|
| Bottom-out cumulative (s/it) | 2.93 | **2.86** | **−2.4%** |
| Fast-mode per-step | ~2.91 (flat) | **2.73–2.75** | **−5.9%** |
| Fast-mode samples | ~64.85 | **62.6–63.6** | **−2.5…−3.4%** |

---

## Pending work

| # | Item | Status |
|---|------|--------|
| 22? | Scheduler weight-cache thrash (`cuda` vs `cuda:0`) | separate task per user decision |
| — | txtfusion design question (reverted #17) | open — see `change-progress.md` |
| — | `set_train_timesteps` sigmoid/shift correctness | open — see `change-progress.md` |

---

## Testing Protocol

For each change:
1. Implement the optimization (≤20 lines, surgical).
2. `pytest tests/` (44 passed baseline) + unit/equivalence check where numerics change.
3. User benchmark: 6 epochs × 30 steps, 4 images. Compare **bottom-out s/it** and sample times
   against the current best above, same dataset mix.
4. Keep only if it improves speed beyond run-to-run variance (~5%); negligible → user decides;
   slower → revert.
5. **Cross-session baselines go stale** (learned 2026-08-31/09-01 during #19): machine state
   drifted ~5–8% below the historical band within days, making a neutral change look like a
   regression. If a bench result disagrees with the mechanism analysis by more than noise,
   run a **same-session control** (revert the code, bench, re-apply) before reverting or
   keeping.

## Notes

- Short-benchmark variation observed: training 2.96–4.43 s/it across warm-up/plateau, samples
  64–70 s/image depending on dataset resolution mix and run conditions.
- `.tmp_opt_test/mmdit_old.py` holds the pre-change `mmdit.py` (from git HEAD) for equivalence
  checks; re-extract with `git show HEAD:extensions_built_in/diffusion_models/krea2/src/mmdit.py`
  after each commit.
- Never benchmark two copies of the real-size model side-by-side — they don't fit in 24 GB VRAM.
- **Do not commit or push** — user handles version control.
- See `gpu-performance-modes.md` for the bimodal GPU performance observation and its implications
