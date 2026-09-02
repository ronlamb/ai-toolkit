# GPU Performance Modes — Bimodal Behavior Observation

**Discovered**: 2026-08-30 during full-run validation of change #16 state.
**Machine**: RTX 4090, Windows, same driver/session.

## Observation

During a full training run (172 images/epoch, 9 samples/checkpoint, 22 checkpoints to step 3784),
training per-step time and sample times oscillate between two distinct performance modes:

| Mode | Training (s/it) | Samples (s/img) |
|------|-----------------|-----------------|
| **Fast** | ~2.73–2.75 | ~62.6–63.6 |
| **Slow** | ~2.94–2.98 | ~67.7–67.9 |
| **Gap** | **~7–8% slower** | **~7–8% slower** |

The two metrics track each other checkpoint-by-checkpoint — when training is fast, samples are
fast; when training is slow, samples are slow. This rules out a code-path explanation (e.g.,
different bucket sizes) and points to a **GPU state difference**.

## Key characteristics

1. **Multi-hour persistence**: each mode lasts for multiple consecutive checkpoints (several hours),
   not a single-step spike.
2. **Both phases affected**: training AND sampling slow together — a GPU-wide effect, not a
   code-specific regression.
3. **Baseline run was flat**: the #10 baseline full run showed no fast mode at all — flat
   ~2.91–2.95 s/it, ~64.8–65.1 s/img. The fast mode appeared in the #16-state run.
4. **Not new-run noise**: the slow mode in the #16 run matches the #10 baseline speed, suggesting
   the "slow" mode is the default thermal/power state.

## Likely causes (unverified)

- **GPU thermal throttling / power limit cycling**: the GPU boosts clocks when cool, then
  throttles after sustained load, then cools and boosts again.
- **Environmental load**: background processes, Windows updates, or other GPU consumers that
  come and go over multi-hour periods.
- **NVIDIA driver power management**: some drivers cycle between performance and power-saving
  states based on workload patterns.

## Implications for benchmarking

1. **Cross-session comparisons are confounded**: a bench run that happens to land in the "fast"
   mode will look ~7% better than one in "slow" mode, even with identical code.
2. **Same-session controls are essential**: when a bench result disagrees with the mechanism
   analysis, revert the code and re-bench in the same session to establish the current mode.
3. **Bottom-out values are more reliable than averages**: the cumulative average smooths over
   mode transitions, but the minimum (bottom-out) value is more likely to reflect the fast mode.
4. **Variance band is ±5–8%**: the total swing between modes is ~7–8%, so changes showing
   ±1–2% improvement are at the edge of statistical significance without same-session controls.

## Protocol learned

This observation led to the "Testing Protocol #5" in `current-state.md`:

> **Cross-session baselines go stale**: machine state drifted ~5–8% below the historical band
> within days, making a neutral change look like a regression. If a bench result disagrees with
> the mechanism analysis by more than noise, run a **same-session control** (revert the code,
> bench, re-apply) before reverting or keeping.

## See also

- `current-state.md` — full-run validation table showing the mode oscillation
- `change-progress.md` — historical context on benchmark methodology evolution
