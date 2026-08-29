# Krea2 Pipeline Optimization — Current State

**This file is the single source of truth for the current best metrics and pending work.**
Per-change details, benchmark tables, and verdicts live in `archive/krea2/set-N/` (one file per
change: `implementation-proposal-change-N.md`, plus `results-change-N.md` for kept changes).

## Current best metrics (as of change #14)

Short benchmark = 6 epochs × 30 steps, 4 images, same dataset throughout (mixes 512×512 and
1024×1024 sets, so compare only same-mix numbers).

| Metric | Value | Where measured |
|--------|-------|----------------|
| Training (short bench) | **~3.15–3.18 s/it** | epochs 4–6 avg; final cumulative 3.15–3.16 |
| Samples (short bench, 1024-mix) | **~65.8 s/image** | epochs 4–6 avg |

> **Baseline calibration**: earlier docs quote **3.02 s/it** as "current best" — that figure comes
> from a longer (~179+ step) run and is *not* comparable to the short benchmark. The short-benchmark
> baseline in the change #10 state is **~3.22–3.26 s/it** (epochs 4–6: 3.23 / 3.22 / 3.22; samples
> ~67.2 s/img). Use that for per-change comparisons.

### Progress across sets

| Milestone | Training (s/it) | Samples (s/img) | Notes |
|-----------|-----------------|-----------------|-------|
| Set-1 baseline (change #5) | 3.25 | 65.64 | short bench |
| Change #10 state (set 2 end) | 3.22 | ~67.2 | short bench; set-2 kept #6, #9, #10 |
| **Change #14 (current best)** | **~3.15–3.18** | **~65.8** | padding `% 256` → `% 32` |
| Full-run bottom-out (#10 state) | 2.93 | 64.85 | long run (172 imgs, 9 samples/checkpoint) — different scale; per-checkpoint data in `archive/krea2/set-3/results-baseline-asof-change10.md` |

**Net vs set-1 baseline (short bench)**: training **−3%**, samples **flat/−0.5%**.

## Kept changes (summary)

| # | Change | Impact | Archive |
|---|--------|--------|---------|
| 1–5 | Set 1 | — | `archive/krea2/set-1/` |
| 6 | Cache VAE norm constants (`latents_mean`/`std`) | samples −2.0% (short test) | `set-2/results-change-6.md` |
| 9 | Single dtype conversion in CFG loop | neutral; kept for cleanliness | `set-2/results-change-9.md` |
| 10 | Pre-compute text-fusion context in sampling loop | neutral; kept for cleanliness | `set-2/results-change-10.md` |
| 14 | Re-align sequence padding `% 256` → `% 32` | training −1.2…−2.5%, samples −2.1% | `set-4/results-change-14.md` |

Reverted (no measurable improvement or regression): #7, #8 (set 2); #11, #12, #13 (set 3);
#14 variant A — remove padding entirely (+10–17% training regression; see
`set-4/results-change-14.md`).

## Pending work (Set 4)

Proposals in `implementation-proposal-change-15..17.md`. None implemented yet.

| # | Change | Expected impact | Status |
|---|--------|-----------------|--------|
| 15 | Lean RMSNorm — drop per-call fp32 round-trip | ~1% both loops (block fwd+bwd −3.4% measured) | 💡 awaiting approval |
| 16 | Lean `ropeapply` — bf16 instead of fp32 round-trip | ~1–2% both loops | 💡 awaiting approval; test after #15 so deltas are attributable |
| 17 | Fix silently-dropped gradients in `txtfusion` (reentrant checkpoint) | **not a speedup** — correctness fix; adds ~100 ms/step of previously-skipped backward | 🐛 user decision |

### Audited and rejected (no change proposed)
- **SDPA backend preference list** (let torch pick backends instead of the forced cuDNN pin in
  `attention()`) — measured: forced cuDNN is already fastest for these shapes (GQA 48/12 heads,
  d=128). The existing pin is optimal.
- **`_mask` broadcast-view instead of materialized (B,1,L,L)** — measured ~1% of attention time;
  below threshold; cuDNN may also de-vectorize on broadcast strides.
- **CFG cond+uncond batching** (one forward with B=2 vs two with B=1) — B=2 costs 2.01× B=1; GPU
  already saturated at B=1 for these shapes.
- **RoPE `omega`/freqs caching** — set-3 #12/#13 reverted; do not re-propose.
- **`prepare()` per-step grid/mask rebuild** — set-2 #7 reverted (+8.6% training); the small tensor
  allocations are not worth the cache-key complexity.
- **`pad_text_features`** — optimized in set 1 (#3); current stack+slice version is fine.
- **`.to(device)` no-op removals in `predict_noise` / `get_noise_prediction`** — no-ops when
  already on device; not worth touching shared code. (Part B of #13, the `encode_images` `.to()`
  removal, was never benchmarked on its own.)
- **`calculate_loss` fp32 MSE accumulation** — intentional precision for bf16 training; keep.
- **Text encoder re-encode per step** — only when `cache_text_embeddings` is off; user's config
  has it on.

## Testing Protocol

For each change:
1. Implement the optimization (≤20 lines, surgical).
2. `pytest tests/` (44 passed baseline) + unit/equivalence check where numerics change.
3. User benchmark: 6 epochs × 30 steps, 4 images. Compare **bottom-out s/it** and sample times
   against the current best above, same dataset mix.
4. Keep only if it improves speed beyond run-to-run variance (~5%); negligible → user decides;
   slower → revert.

## Notes

- Short-benchmark variation observed: training 2.96–4.43 s/it across warm-up/plateau, samples
  64–70 s/image depending on dataset resolution mix and run conditions.
- `.tmp_opt_test/mmdit_old.py` holds the pre-change `mmdit.py` (from git HEAD) for equivalence
  checks; re-extract with `git show HEAD:extensions_built_in/diffusion_models/krea2/src/mmdit.py`
  after each commit.
- Never benchmark two copies of the real-size model side-by-side — they don't fit in 24 GB VRAM.
- **Do not commit or push** — user handles version control.
