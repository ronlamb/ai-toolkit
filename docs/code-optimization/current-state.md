# Krea2 Pipeline Optimization — Current State

**This file is the single source of truth for the current best metrics and pending work.**
Per-change details, benchmark tables, and verdicts live in `archive/krea2/set-N/` (one file per
change: `implementation-proposal-change-N.md`, plus `results-change-N.md` for kept changes).

## Current best metrics (as of change #16)

Short benchmark = 6 epochs × 30 steps, 4 images, same dataset throughout (mixes 512×512 and
1024×1024 sets, so compare only same-mix numbers).

| Metric | Value | Where measured |
|--------|-------|----------------|
| Training (short bench) | **~3.09–3.17 s/it** | epochs 4–6 avg 3.13; final cumulative (bottom-out) **3.09** |
| Samples (short bench, 1024-mix) | **~64.7 s/image** | epochs 4–6 avg |

> Kept change #16 (lean `ropeapply`, bf16): bottom-out 3.09 s/it vs #14 band 3.15–3.18; samples
> epochs 4–6 avg 64.7 vs ~65.8. User extended the run to 10 epochs — converged fine, only the
> first two (warm-up) epochs looked slower; decision: keep.

### Progress across sets

| Milestone | Training (s/it) | Samples (s/img) | Notes |
|-----------|-----------------|-----------------|-------|
| Set-1 baseline (change #5) | 3.25 | 65.64 | short bench |
| Change #10 state (set 2 end) | 3.22 | ~67.2 | short bench; set-2 kept #6, #9, #10 |
| Change #14 | ~3.15–3.18 | ~65.8 | padding `% 256` → `% 32` |
| **Change #16 (current best)** | **~3.09–3.13** | **~64.7** | lean `ropeapply` bf16 |
| Full-run bottom-out (#10 state) | 2.93 | 64.85 | long run (172 imgs, 9 samples/checkpoint) — different scale; per-checkpoint data in `archive/krea2/set-3/results-baseline-asof-change10.md` |

**Net vs set-1 baseline (short bench)**: training **−4%**, samples **−1.4%**.

## Kept changes (summary)

| # | Change | Impact | Archive |
|---|--------|--------|---------|
| 1–5 | Set 1 | — | `archive/krea2/set-1/` |
| 6 | Cache VAE norm constants (`latents_mean`/`std`) | samples −2.0% (short test) | `set-2/results-change-6.md` |
| 9 | Single dtype conversion in CFG loop | neutral; kept for cleanliness | `set-2/results-change-9.md` |
| 10 | Pre-compute text-fusion context in sampling loop | neutral; kept for cleanliness | `set-2/results-change-10.md` |
| 14 | Re-align sequence padding `% 256` → `% 32` | training −1.2…−2.5%, samples −2.1% | `set-4/results-change-14.md` |
| 16 | Lean `ropeapply` — bf16 instead of fp32 round-trip | training bottom-out −2.0%, samples −1.7% | `set-4/` (pending archive) |

Reverted (no measurable improvement or regression): #7, #8 (set 2); #11, #12, #13 (set 3);
#14 variant A — remove padding entirely (+10–17% training regression; see
`set-4/results-change-14.md`); #15 — lean RMSNorm (dead-even training, samples slower; user
decision).

## Benchmark results — Change #15 (lean RMSNorm), tested 2026-08-29

Short bench, same dataset mix. Cumulative `s/it` at epoch end; per-step avg from total-time deltas.

| Epoch | Cum s/it | Per-step avg s/it | Samples avg (s/img) |
|-------|----------|-------------------|---------------------|
| 1 | 3.84 | warm-up | 65.39 |
| 2 | 3.61 | 3.40 | 66.78 |
| 3 | 3.23 | 2.47 (anomalous fast epoch) | 66.52 |
| 4 | 3.23 | 3.23 | 66.65 |
| 5 | 3.22 | 3.17 | 66.65 |
| 6 | 3.19 | 3.03 | 66.30 |

| Metric | #14 best | #15 | Delta |
|--------|----------|-----|-------|
| Epochs 4–6 avg (s/it) | ~3.15–3.18 | 3.14 | −0.3…−1.2% (within variance) |
| Final cumulative (s/it) | 3.15–3.16 | 3.19 | +1% |
| Samples epochs 4–6 avg (s/img) | ~65.8 | 66.5 | +0.7% (slower) |

Plateau inside the #14 band; samples slightly slower; epoch-3 per-step 2.47 is a single
anomalous epoch (1:14 vs steady ~1:33–1:37), not sustained bottom-out. Micro-benchmark predicted
~2–3% training gain; plateau shows ≤1% at best → negligible/mixed.

**Decision (user): ⚠️ REVERTED** — sample times are the deciding metric and #15 was consistently
slower there (66.5 vs 65.8 s/img avg, slower in epochs 2–6); training was a dead heat. Code
restored via `git checkout -- extensions_built_in/diffusion_models/krea2/src/mmdit.py`.

## Benchmark results — Change #16 (lean `ropeapply`), tested 2026-08-29

Short bench, same dataset mix. Cumulative `s/it` at epoch end; per-step avg from total-time deltas.

| Epoch | Cum s/it | Per-step avg s/it | Samples avg (s/img) |
|-------|----------|-------------------|---------------------|
| 1 | 3.71 | warm-up | 63.77 |
| 2 | 3.30 | 2.90 | 64.74 |
| 3 | 3.29 | 3.27 | 65.02 |
| 4 | 3.14 | 2.70 | 65.22 |
| 5 | 3.17 | 3.27 | 65.61 |
| 6 | 3.09 | 2.70 | 63.22 |

| Metric | #14 best | #16 | Delta |
|--------|----------|-----|-------|
| Epochs 4–6 avg cum (s/it) | ~3.15–3.18 | 3.13 | −0.6…−1.5% |
| Final cumulative / bottom-out (s/it) | 3.15–3.16 | **3.09** | **−2.0…−2.2%** |
| Samples epochs 4–6 avg (s/img) | ~65.8 | 64.7 | **−1.7%** |
| Samples best epoch (s/img) | — | 63.2 (ep 6) | −4% vs 65.8 |

Both bottom-out metrics improved, in the same direction, matching the predicted ~1–2% from the
micro-bench (−65 % per `ropeapply` call × ~28 calls/fwd + checkpoint-recompute pass). The gain sits
at the upper edge of the ±1–2 % variance band — better than #15 (which was dead-even/slower), but
borderline. Per protocol, borderline → user decides.

User extended the run to **10 epochs**: converged fine; only the first two (warm-up) epochs looked
slower/less converged — consistent with bf16-rounded RoPE table noise at low step counts, not a
regression in steady state.

**Decision (user): ✅ KEEP** — both metrics improved and held up over 10 epochs. New current best:
bottom-out **3.09 s/it**, samples epochs 4–6 avg **64.7 s/img**.

## Pending work (Set 4)

Proposals in `implementation-proposal-change-15..17.md`.

| # | Change | Expected impact | Status |
|---|--------|-----------------|--------|
| 15 | Lean RMSNorm — drop per-call fp32 round-trip | ~1% both loops (block fwd+bwd −3.4% measured) | ⚠️ REVERTED — tested: training dead-even (epochs 4–6 per-step 3.14 vs 3.15), samples +0.7–1.1% slower; user decided to revert |
| 16 | Lean `ropeapply` — bf16 instead of fp32 round-trip | ~1–2% both loops | ✅ COMPLETED — bottom-out training **3.09 s/it** (−2.0% vs #14), samples epochs 4–6 avg **64.7 s/img** (−1.7%). User extended to 10 epochs: only warm-up (first 2 epochs) slower, converged fine; user decided to keep |
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
