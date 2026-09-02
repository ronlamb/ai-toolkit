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
| **Change #16 (current best)** | **~3.09–3.13** | **~64.7** | lean `ropeapply` bf16; **+20 pts visual accuracy, ~31% faster convergence vs #10 state (full runs)** |
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
decision); #17 — txtfusion gradient fix (correct but +11% s/it; design question open — see
Pending work).

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

## Change #17 — implemented 2026-08-29 (validation done; user benchmark pending)

Applied the minimal correctness fix from `implementation-proposal-change-17.md`: added
`use_reentrant=False` at both `checkpoint(...)` call sites in `mmdit.py`
(`TextFusionBlock.forward` line 284, `TextFusionTransformer.forward` line 323). 2 lines changed.

### Local validation (all passed)

1. **Component-level** (`TextFusionTransformer`, CUDA bf16, inputs with
   `requires_grad=False` — the cached-text-embeds scenario): output
   `requires_grad=True` / `grad_fn` present; **49/49 params got non-zero grad**; no
   checkpoint `UserWarning`. (Before the fix this path returned `grad_fn=None`.)
2. **Model-level** (`SingleStreamDiT`, gradient checkpointing on, cached non-grad context):
   **txtfusion 49/49** and all params **92/92** with non-zero grad — matches the proposal's
   expected restoration (was 0/33 pre-fix).
3. `pytest tests/` → **44 passed**.

*Test-config note*: a tiny model must use cuDNN-compatible head dims — default
`txtheads/txtkvheads=20` with `txtdim=1024` gives headdim 51 → "No available kernel".
Use `txtheads=8, txtkvheads=8` (headdim 128) in tests; real model is unaffected
(txtdim=2560 → headdim 128).

### Expected benchmark outcome

s/it may **rise** slightly (~tens of ms/step: txtfusion backward now actually runs);
sample times should be unchanged (sampling skips checkpointing via
`torch.is_grad_enabled()`). Judge on sample *quality* improving, since txtfusion LoRA
adapters finally train. Per protocol the keep/revert call is the user's.

### Benchmark results — user run 2026-08-29 (6 epochs × 30 steps, 4 images)

Short bench, same dataset mix. Cumulative `s/it` at epoch end; per-step avg from total-time deltas.

| Epoch | Cum s/it | Per-step avg s/it | Samples avg (s/img) |
|-------|----------|-------------------|---------------------|
| 1 | 4.81 | 4.79 (warm-up) | 68.49 |
| 2 | 4.41 | 4.03 | 67.68 |
| 3 | 4.05 | 3.33 | 67.61 |
| 4 | 3.82 | 3.13 | 66.46 |
| 5 | 3.71 | 3.30 | 64.20 |
| 6 | **3.63** | 3.20 | 63.95 |

| Metric | #16 best | #17 | Delta |
|--------|----------|-----|-------|
| Epochs 4–6 per-step avg (s/it) | 2.89 | 3.21 | **+11%** slower |
| Final cumulative / bottom-out (s/it) | 3.09 | 3.63 | **+17%** slower (still descending at ep 6) |
| Samples epochs 4–6 avg (s/img) | 64.7 | 64.9 | +0.3% — flat, as predicted |

Regression is ~3× the proposal's estimate (~320 ms/step measured vs ~100 ms predicted):
the skipped backward covered not just txtfusion's own recompute but the full gradient path
back through its 4 blocks × (B×12) batched sequence, which now executes every step.

**Decision (user): ⚠️ REVERTED 2026-08-29** — +11% steady-state training cost is not acceptable
as a pure correctness fix without first understanding the intended design. Code restored via
`git checkout -- extensions_built_in/diffusion_models/krea2/src/mmdit.py`.

### Open question to investigate later (why this was reverted, not closed)

The user's concern: **is the dropped-gradient behaviour actually a bug in the overall logic, or
is txtfusion intended to be frozen?** Two self-consistent designs exist and the codebase does
not say which one is meant:

- **(a) txtfusion should train.** Then #17 (or cheaper option C) is the right fix, and the cost
  is real compute that was previously skipped. Needs a VRAM check for option C (~+670 MB peak).
- **(b) txtfusion is intended frozen** (e.g. it's a pretrained text-fusion stage treated as a
  fixed feature extractor). Then the *real* bug is elsewhere: `lora_special.create_modules`
  matches on `"blocks"`, which accidentally attaches trainable LoRA adapters to
  `txtfusion.layerwise_blocks.*` / `txtfusion.refiner_blocks.*`. Those adapters get optimizer
  state and VRAM but never learn. Fixing that means **excluding** txtfusion from the LoRA match
  list — cheaper than today (no wasted optimizer state), same speed, no checkpoint change.

Related clarifications:
- `cache_text_embeddings: true` is a **VRAM/compute optimisation**, not part of the bug: it skips
  re-running the text encoder each step. It only *exposes* the reentrant-checkpoint quirk by
  feeding txtfusion inputs that never require grad.
- Activation checkpointing's purpose is memory, not speed: it discards activations in forward and
  **recomputes** them in backward to avoid storing them. It is always a compute-for-memory trade.
  The nested (double) checkpointing in txtfusion currently pays recompute twice for one
  memory saving — that part is redundant regardless of which design is intended.

Suggested next step when revisiting: ask the model author / check upstream Krea 2 training code
whether txtfusion + its LoRA are meant to train, then pick (a) or (b). If (b), also confirm no
existing checkpoints carry trained-looking txtfusion LoRA weights (they should be identical to
init, since they never received gradients).

## Full-run validation — state as of #16 (vs #10 full-run baseline), tested 2026-08-30

Same protocol as the set-3 baseline (`archive/krea2/set-3/results-baseline-asof-change10.md`):
172 images/epoch, checkpoint every epoch, 9 samples per checkpoint, run to step 3784. Code = #10 +
#14 + #16 (#15 and #17 reverted). 

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

### Key finding: run oscillates between two performance modes

Training per-step and sample times are **bimodal, and the two metrics track each other
checkpoint-by-checkpoint**: fast mode ≈ 2.74 s/it + ~62.7 s/img; slow mode ≈ 2.96 s/it +
~67.8 s/img. The #10 baseline run was flat (~2.91–2.95, ~64.8–65.1) with no fast mode at all —
so this is not new-run noise but a sustained multi-hour GPU state difference (thermal/power or
environmental load), affecting training and sampling together.

### Comparison vs #10 full-run baseline (2.93 s/it bottom-out, 64.85 s/img)

| Metric | #10 baseline | #16-state run | Delta |
|--------|--------------|---------------|-------|
| Bottom-out cumulative (s/it) | 2.93 | **2.86** | **−2.4%** |
| Mean per-step, steps 516–3784 | 2.90 | 2.85 | −1.7% |
| Fast-mode per-step | ~2.91 (flat) | **2.73–2.75** | **−5.9%** |
| Slow-mode per-step | ~2.91 (flat) | 2.94–2.98 | +1.5% |
| Mean samples (s/img) | 64.85 | 65.4 | +0.9% |
| Fast-mode samples | ~64.85 | **62.6–63.6** | **−2.5…−3.4%** |
| Slow-mode samples | ~64.85 | 67.7–67.9 | +4.5% |

Interpretation: in fast mode the #14+#16 code is clearly better than the #10 baseline on **both**
metrics (−6% training, −3 % samples); slow mode matches or slightly exceeds baseline. Averages are
muddied by the mode oscillation — if the slow mode is environmental (GPU clocks/thermals), the
true improvement is closer to the fast-mode numbers.

**User-reported qualitative results (pending): image quality and convergence improvements noted in
this run — details were cut off when pasting; to be appended here.**

## Qualitative validation — Change #16 state vs #10 state (full runs, same dataset)

Manual per-checkpoint review of the two full runs (small_run = #10 state, 
full_run = #16 state): 9 samples per epoch scored per aspect (count correct / 9;
tattoo categories scored against images that actually have that attribute). 
Visual accuracy = user's overall estimate.

### Epochs 1–10 — visual accuracy trajectory

| Epoch | #10 | #16 | Epoch | #10 | #16 |
|-------|-----|-----|-------|-----|-----|
| 1 | 20% | 25% | 6 | 40% | **50%** |
| 2 | 25% | 25% | 7 | 40% | 50% |
| 3 | 20% | 20% | 8 | 45% | **60%** |
| 4 | 35% | 35% | 9 | 50% | **65%** |
| 5 | 35% | 30% | 10 | 50% | **70%** |

Body/face/hair first "locked in" at epoch ~10 for both, but #16 pulled ahead immediately after.
Minor caveat: #16's *position* accuracy dipped in epochs 1–3 (5–6/9 vs 8–9/9) before catching up —
consistent with the warm-up wobble seen in the short bench.

### Peak window (epochs 15–17) — per-aspect at epoch 15

| Aspect | #10 | #16 | Aspect | #10 | #16 |
|--------|-----|-----|--------|-----|-----|
| Hair color / style | 9/9, 9/9 | 9/9, 9/9 | Chest tattoos (6) | **0** | **3** |
| Face / Body | 9/9, 9/9 | 9/9, 9/9 | Back tattoos (2) | 0 | **2** |
| Arm tattoos (8) | 7 | **9** | Leg tattoos (7) | 2 | **5** |
| Position | 9 | 9 | **Visual accuracy** | **70%** | **90–95%** |

The gap is concentrated in **fine/rare detail**: tattoo categories. Chest and back tattoos never
materialized in #10's early-mid epochs; they appear by epoch 9–15 in #16.

### Convergence point (the headline result)

| | #10 state | #16 state |
|---|-----------|-----------|
| Converged at | ~epoch 29 (chest tattoos not consistent until ~epoch 32) | **~epoch 20** |
| Steps to convergence | ~4,988–5,504 | **~3,440** |
| Quality at convergence | baseline | **better than #10's epoch 32** |

**#16 converges ~1,550 steps (~31%) earlier at higher final accuracy.** For production training
that is a larger real-world saving than the per-step speedups: same quality in ~⅔ the compute.

### Interpretation — why would "speed" changes improve quality?

Both runs have txtfusion frozen (reverted #17), so the difference comes from #14 + #16 alone.
#14 (padding `% 256` → `% 32`) is numerically neutral when masking is correct. #16 changes
numerics: RoPE tables are now bf16-rounded instead of fp32-computed-then-downcast — a small,
consistent perturbation to rotation embeddings at every attention call. Plausible mechanisms
(unverified): regularization-like noise improving generalization on rare features (tattoos appear
in few images), or simply a different-but-better training trajectory (numerics diverge from step 1).
Either way the effect is large (+20 pts visual accuracy at epoch 10 and 15, ~31% faster
convergence) and reproducible across the user's manual review — worth recording as #16's primary
benefit, with the caveat that trajectory variance cannot be fully excluded without a repeat run.

### Implication for change #17 (txtfusion design question)

The strong quality gains came from code where txtfusion was **frozen** — so they neither require
nor rule out training it. They do show the pipeline learns these fine details *without* txtfusion
gradients, which slightly favors design (b): if a future experiment wants to test (a), it should be
an explicit quality A/B (keep #16 code + gradient fix, compare convergence at epoch 20/30), not a
silent change.

## Pending work (Set 4)

Proposals in `implementation-proposal-change-15..17.md`.

| # | Change | Expected impact | Status |
|---|--------|-----------------|--------|
| 15 | Lean RMSNorm — drop per-call fp32 round-trip | ~1% both loops (block fwd+bwd −3.4% measured) | ⚠️ REVERTED — tested: training dead-even (epochs 4–6 per-step 3.14 vs 3.15), samples +0.7–1.1% slower; user decided to revert |
| 16 | Lean `ropeapply` — bf16 instead of fp32 round-trip | ~1–2% both loops | ✅ COMPLETED — bottom-out training **3.09 s/it** (−2.0% vs #14), samples epochs 4–6 avg **64.7 s/img** (−1.7%). User extended to 10 epochs: only warm-up (first 2 epochs) slower, converged fine; user decided to keep |
| 17 | Fix silently-dropped gradients in `txtfusion` (reentrant checkpoint) | **not a speedup** — correctness fix; measured +11% steady training cost, samples flat | ⚠️ REVERTED — tested: +11% s/it for no measurable speed benefit. Revisit only after confirming whether txtfusion is meant to train at all (see open question above) |

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
- **`pad_text_features`** — ~~optimized in set 1 (#3); current stack+slice version is fine.~~
  **CORRECTED 2026-08-30: this entry was wrong.** The #3 vectorized version crashes on ragged
  caption lengths (`torch.stack` before padding). See finding B / change #19 below.
- **`.to(device)` no-op removals in `predict_noise` / `get_noise_prediction`** — no-ops when
  already on device; not worth touching shared code. (Part B of #13, the `encode_images` `.to()`
  removal, was never benchmarked on its own.)
- **`calculate_loss` fp32 MSE accumulation** — intentional precision for bf16 training; keep.
- **Text encoder re-encode per step** — only when `cache_text_embeddings` is off; user's config
  has it on.

## Set 5 — audit of main…krea_5 (2026-08-30)

Detour audit of all changes since `main` (sets 1–4 + earlier MPS/CUDA work), restricted to code on
the Krea 2 path, looking for: calculation simplification, dead code, and extra CPU↔GPU transfers.
**No code changed yet.** All findings are being written up as proposals first; each will be
implemented in its own separate session.

Decisions taken during review:
- One change per fix (A, B, C, D benchmarked separately) — cleanest attribution.
- Change #18 kept **strictly minimal**; the scheduler cache-thrash is a separate task (#19).
- The unused `toolkit/util/torch_util.py` module + its test: **delete** (user decision).
- Non-performance changes are still benchmarked (short bench, no-regression confirmation).

### Verified findings

| # | Finding | File | Severity | Proposal |
|---|---------|------|----------|----------|
| A | `_get_step_indices` reverses sample↔index pairing for batch > 1 (assumes queries are sorted; they are random per sample) | `toolkit/samplers/custom_flowmatch_sampler.py` | Correctness — **dormant** under current configs (batch_size 1, `timestep_type: linear`) | #18 |
| B | `pad_text_features` crashes on ragged caption lengths (`torch.stack` before padding) | `krea2/src/pipeline.py` | Crash — any multi-prompt batch with differing token counts | #19 (see note) |
| C | `to_device_if_needed` device compare never matches on CUDA (`cuda` vs `cuda:0`) → always copies; also silently ignores `dtype` for `PromptEmbeds` already on device | `extensions_built_in/sd_trainer/SDTrainer.py` | Perf (defeats its own purpose) + dtype-skip edge | #20 |
| D | `flip_x` NameError — `deepcopy` line commented out, assignment left behind; plus duplicate `import copy` | `toolkit/data_loader.py` ~L568 | Crash when `flip_x: true` | #21 |
| E | `toolkit/util/torch_util.py` — 128 lines / 14 helpers, zero production call sites (only its own test + docs) | `toolkit/util/torch_util.py` | Dead code | delete module + test |

### Minor findings (folded into related changes or noted only)

- Scheduler weight-cache thrash: `_cached_device != timesteps.device` compares unindexed `cuda`
  vs `cuda:0`, so the 3 weight tensors are re-copied every call. Separate task from #18 by user
  decision. Only live when weighted timesteps are enabled.
- `pad_text_features`: `torch.tensor(lengths, device=device)` is a CPU→GPU copy per call — fold
  into the same change as B since that function is being rewritten anyway.
- `encode_images` per-image VAE loop (set-1 #1) loses batched encode on CUDA; MPS-motivated.
  Candidate for its own perf proposal if wanted.
- Dead import `precondition_model_outputs_flow_match` in SDTrainer (never invoked anywhere).
- Whitespace-only diff in `toolkit/data_transfer_object/data_loader.py` — diff noise.
- Per-step `hasattr(_cached_pipeline)` in `end_step_hook` — never true for Krea2Model; harmless.

### Open question carried from the audit

`set_train_timesteps` sigmoid/shift branches were rewritten (duplicate last element + argsort
descending) so `timesteps`/`sigmas` lengths align. Semantics differ from main's `ones`/`zeros`
append. Not yet checked for correctness — only matters for configs using
`timestep_type: sigmoid|shift|flux_shift|lognorm_blend`.

### Proposal index (set 5)

Written to `archive/krea2/set-5/`. Implementation happens one-per-session after all proposals are
written; nothing applied yet.

| # | Proposal | Finding | Status |
|---|----------|---------|--------|
| 18 | `implementation-proposal-change-18.md` | A — `_get_step_indices` reversal | implemented 2026-08-31, awaiting user benchmark |
| 19 | `implementation-proposal-change-19.md` | B — `pad_text_features` ragged crash (+ folded in the `lengths` CPU→GPU copy) | **implemented + benchmarked with control: KEEP** (control run slower than #19 → session drift, not regression) |
| 20 | `implementation-proposal-change-20.md` | C — `to_device_if_needed` device compare + dtype-skip | **implemented + benchmarked 2026-09-01: KEEP** (no regression) |
| 21 | `implementation-proposal-change-21.md` | D — `flip_x` UnboundLocalError + duplicate import | **implemented + benchmarked 2026-09-02: KEEP** (no regression) |
| — | *(no proposal)* | E — delete `toolkit/util/torch_util.py` + `tests/test_torch_util.py` | user decided: delete |
| 22? | *(to decide)* | Scheduler weight-cache thrash (`cuda` vs `cuda:0`) | separate task per user |

### Set 5 — implementation results

#### Change #18 — `_get_step_indices` reversal fix (implemented 2026-08-31, branch `krea_5`)

Applied the minimal fix from the proposal (18 → 12 lines, single function,
callers unchanged). Correctness-only: dormant under the benchmark config
(batch_size 1, `timestep_type: linear`, `linear_timesteps` false), so expected
benchmark result is **exactly neutral** — no-regression confirmation only.

Local validation (all passed):
- Repro vs main's equality-loop semantics (`.tmp_opt_test/repro_change18.py`): linear grid
  unsorted batch-3 `[923, 456, 781]` → `[77, 544, 219]` PASS (old branch code returned the
  reversed `[219, 544, 77]`); batch-1 and all-equal cases unchanged; random batch-8,
  sigmoid grid (exact + unsorted), and ascending-grid cases all match expected.
- `pytest tests/` → **44 passed**.

| Metric | Current best (#16) | #18 | Verdict |
|--------|--------------------|-----|---------|
| Bottom-out training (s/it), short bench | 3.09 | ~3.12 @ step 179 | neutral — no regression |
| Samples epochs 4–6 avg (s/img) | 64.7 | ~63.7–67.6 | neutral |
| Full run bottom-out (s/it) | 2.86 (`ut_2 - Copy`, #17 state, 22 ckpts) | **2.82** (36 ckpts) | faster + lower; see caveat |

Full-run note: the new run also bottomed out sooner. ~1.4% gain is **not attributable to #18**
(function never executes under this config) — likely longer run (cumulative avg over 6192 vs
3784 steps) + unseeded shuffle variance.

Run-comparability caveat: `training_seed` unset → training RNG differs every run; sample seeds
only pin generation latents. Cross-run quality comparisons are confounded by run-to-run
variance (see `set-5/implementation-proposal-change-18.md` benchmark section).

**Status: IMPLEMENTED — benchmarked, no regression. KEEP.**

#### Change #19 — `pad_text_features` ragged-caption crash fix (implemented 2026-08-31, branch `krea_5`)

Applied the proposed replacement verbatim (14+/13−, single function in
`extensions_built_in/diffusion_models/krea2/src/pipeline.py`). Fixes finding B
(`torch.stack` before padding crashed any ragged multi-prompt batch) and folds in the
minor finding: mask now built on CPU and transferred once, removing the per-call
implicit CPU→GPU sync from `torch.tensor(lengths, device=device)`.
Fast path preserved: equal-length lists still take a single vectorized stack+copy.

Local validation (all passed):
- Repro `.tmp_opt_test/repro_change19.py` vs reference per-row-loop semantics on **CPU and
  CUDA**: pre-fix the 3 ragged cases crash on both devices; post-fix all 5 cases × 2 devices
  PASS (features + mask exactly equal to reference, mask dtype long).
- `pytest tests/` → **44 passed**.

Expected benchmark: **exactly neutral** — benchmark config caches text embeddings (equal
lengths → fast path) and samples single prompts (B=1). No-regression confirmation only.

| Metric | Baseline (#16) | #18 run | #19 run |
|--------|----------------|---------|----------|
| Cumulative s/it @ step 179 | 3.09 (bottom-out) | ~3.12 | **3.22** (+4% vs #18) |
| Samples avg (s/img) | 64.7 | 63.7–67.6 band | **≈68.4** — every round ≥ top of #18 band |

**Slower in both phases**, but attribution analysis shows the slowdown cannot come from this
code: ~180 training calls + 48 sampling calls total under the bench, each on a tiny tensor —
explaining the deltas would need ~0.13 s/call in training and ~5 s/call in sampling. Features
and mask are bitwise-equal to the old code on the fast path (loss values unchanged step-for-step),
and both phases slowed together — signature of GPU thermal state / background load, not of this
function. Full analysis + options in `set-5/implementation-proposal-change-19.md`.

**Control experiment**: user ran the #19 bench twice in one session with identical results,
which excludes within-session noise but cannot compare against baselines measured in earlier
sessions. `pipeline.py` was reverted to HEAD for a same-session control bench (fix saved in
`.tmp_opt_test/change19.patch`).

Control result (2026-09-01, pre-#19 code): **3.36 s/it @179 / ≈69.2 s/img — slower than both
#19 runs** (3.22 / ≈68.4). Decisively the "session drift" outcome: today's machine state sits
below every historical band, so the apparent #19 regression vs history was never caused by
#19. Fix re-applied and re-validated (repro ALL PASS cpu+cuda, pytest 44).

**Status: IMPLEMENTED — control-benched, no regression vs same-code baseline. KEEP.**

#### Change #20 — `to_device_if_needed` device-compare + dtype-skip fix (implemented 2026-09-01, branch `krea_5`)

Applied the proposed change verbatim in `extensions_built_in/sd_trainer/SDTrainer.py`:
new `_devices_match` helper resolves unindexed vs indexed devices (`cuda` ≡ `cuda:0`,
`mps` ≡ `mps:0`), and the `PromptEmbeds` branch now honours a requested `dtype` cast when
the device already matches (was silently skipped). +21 lines net, one function + helper;
all ~59 call sites untouched.

Local validation (all passed):
- Repro on CUDA covering every row of the proposal's validation table plus extras —
  **ALL PASS**, incl. the previously-broken case: PromptEmbeds fp32 on `cuda:0` with an
  indexed target + bf16 now returns **bf16** (old code returned fp32).
- `pytest tests/` → **44 passed**.

Expected benchmark: **neutral** — `.to()` already short-circuits internally, so this removes
wasted Python work (~59 extra real `.to()` calls/step), not GPU copies. Per protocol,
non-performance changes are still benchmarked (short bench, no-regression check).
Given the #19 lesson: if the bench contradicts the neutral expectation, run a
**same-session control** before keeping/reverting.

Short bench (6 epochs × 30 steps, 4 images): bottom-out cumulative **3.08 s/it** @ step 179;
samples overall avg ≈65.9 s/img (epochs 4–6 avg ≈67.4; min per-image 62.4). Best bottom-out
ever recorded on this bench, but only +0.3% vs the historical best (3.09 = #16) and today's
machine state is in a faster regime than the #19 sessions anyway (their same-code control was
3.36 / ≈69.2). Bench agrees with mechanism analysis → **neutral, no regression**; protocol #5
control not triggered.

**Status: IMPLEMENTED — benchmarked, no regression. KEEP** (correctness fix; speed delta
negligible).

#### Change #21 — `flip_x` UnboundLocalError + duplicate import fix (implemented 2026-09-01, branch `krea_5`)

Applied as proposed in `toolkit/data_loader.py` (class **`AiToolkitDataset`** — the proposal
doc said `LoRADataset`, corrected there): restored the merge-swallowed
`new_file_item = copy.deepcopy(file_item)` assignment + comment in the flip_x block (now
matches the flip_y twin and `main`), deleted the duplicate `import copy`. Net −1 line.
**Hard crash fix**: any dataset with `flip_x: true` raised at construction; dormant under
the benchmark config (`flip_x: false`).

Local validation (all passed):
- Control-flow repros: flip_x-only (list doubles, originals unmutated) and flip_x+flip_y
  (×4 items, all combos once) — PASS; real `AiToolkitDataset.__init__` source check +
  module compile/import — PASS.
- `pytest tests/` → **44 passed**.

Expected benchmark: **byte-identical timing** to the #20 run (bottom-out 3.08 s/it) — the
fixed lines never execute under this config. No-regression confirmation only; protocol #5
control if it deviates beyond noise.

Short bench (6 epochs × 30 steps, 4 images): bottom-out cumulative **3.09 s/it** @ step 179
vs #20's 3.08 — flat (−0.3%, noise). Samples overall avg ≈63.2 s/img (epochs 4–6 avg ≈63.1),
≈6% faster than the #20 run — **not attributable** (code is dormant under `flip_x: false`);
same session-drift pattern as #18/#19/#20. Bench agrees with mechanism analysis → no control
needed (protocol #5 not triggered).

**Status: IMPLEMENTED — benchmarked, no regression. KEEP** (hard crash fix for
`flip_x: true` datasets; zero runtime cost elsewhere).

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
