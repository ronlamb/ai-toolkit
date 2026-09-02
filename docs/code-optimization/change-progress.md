# Krea2 Optimization — Change Progress

Historical record of all changes across sets. Current best metrics live in
`current-state.md`; per-change details and benchmark tables live in
`archive/krea2/set-N/implementation-proposal-change-N.md`.

---

## Progress across sets (short benchmark)

Short benchmark = 6 epochs × 30 steps, 4 images, same dataset mix (512×512 + 1024×1024).

| Milestone | Training (s/it) | Samples (s/img) | Notes |
|-----------|-----------------|-----------------|-------|
| Set-1 baseline (change #5) | 3.25 | 65.64 | short bench |
| Change #10 state (set 2 end) | 3.22 | ~67.2 | short bench; set-2 kept #6, #9, #10 |
| Change #14 | ~3.15–3.18 | ~65.8 | padding `% 256` → `% 32` |
| **Change #16 (current best)** | **~3.09–3.13** | **~64.7** | lean `ropeapply` bf16; **+20 pts visual accuracy, ~31% faster convergence vs #10 state (full runs)** |
| Full-run bottom-out (#10 state) | 2.93 | 64.85 | long run (172 imgs, 9 samples/checkpoint) — different scale; per-checkpoint data in `archive/krea2/set-3/results-baseline-asof-change10.md` |

**Net vs set-1 baseline (short bench)**: training **−4%**, samples **−1.4%**.

---

## Kept changes (summary)

| # | Change | Impact | Archive |
|---|--------|--------|---------|
| 1–5 | Set 1 | — | `archive/krea2/set-1/` |
| 6 | Cache VAE norm constants (`latents_mean`/`std`) | samples −2.0% (short test) | `set-2/results-change-6.md` |
| 9 | Single dtype conversion in CFG loop | neutral; kept for cleanliness | `set-2/results-change-9.md` |
| 10 | Pre-compute text-fusion context in sampling loop | neutral; kept for cleanliness | `set-2/results-change-10.md` |
| 14 | Re-align sequence padding `% 256` → `% 32` | training −1.2…−2.5%, samples −2.1% | `set-4/results-change-14.md` |
| 16 | Lean `ropeapply` — bf16 instead of fp32 round-trip | training bottom-out −2.0%, samples −1.7% | `set-4/implementation-proposal-change-16.md` |
| 18 | `_get_step_indices` reversal fix (batch > 1 correctness) | neutral (dormant under current config) | `set-5/implementation-proposal-change-18.md` |
| 19 | `pad_text_features` ragged-caption crash fix | neutral (correctness fix) | `set-5/implementation-proposal-change-19.md` |
| 20 | `to_device_if_needed` device-compare + dtype-skip fix | neutral (correctness fix) | `set-5/implementation-proposal-change-20.md` |
| 21 | `flip_x` UnboundLocalError + duplicate import fix | neutral (crash fix, dormant under current config) | `set-5/implementation-proposal-change-21.md` |

---

## Reverted changes

| # | Change | Reason | Archive |
|---|--------|--------|---------|
| 7 | `prepare()` per-step grid/mask rebuild | +8.6% training regression | `set-2/results-change-7.md` |
| 8 | (set 2) | no measurable improvement | `set-2/` |
| 11–13 | RoPE `omega`/freqs caching variants | reverted; do not re-propose | `set-3/` |
| 14-A | Remove padding entirely | +10–17% training regression | `set-4/results-change-14.md` |
| 15 | Lean RMSNorm — drop per-call fp32 round-trip | dead-even training, samples +0.7% slower; user decision | `set-4/implementation-proposal-change-15.md` |
| 17 | Fix silently-dropped gradients in `txtfusion` | correct but +11% s/it; design question open | `set-4/implementation-proposal-change-17.md` |

---

## Audited and rejected (no change proposed)

Items investigated during set-4/5 audits and found to have no benefit:

- **SDPA backend preference list** — forced cuDNN is already fastest for these shapes (GQA 48/12 heads, d=128).
- **`_mask` broadcast-view instead of materialized (B,1,L,L)** — ~1% of attention time; below threshold.
- **CFG cond+uncond batching** — B=2 costs 2.01× B=1; GPU already saturated at B=1.
- **RoPE `omega`/freqs caching** — set-3 #12/#13 reverted; do not re-propose.
- **`.to(device)` no-op removals in `predict_noise`/`get_noise_prediction`** — no-ops when already on device.
- **`calculate_loss` fp32 MSE accumulation** — intentional precision for bf16 training; keep.
- **Text encoder re-encode per step** — only when `cache_text_embeddings` is off; user's config has it on.
- **Dead import `precondition_model_outputs_flow_match`** in SDTrainer (never invoked).
- **Whitespace-only diff** in `toolkit/data_transfer_object/data_loader.py`.
- **Per-step `hasattr(_cached_pipeline)`** in `end_step_hook` — never true for Krea2Model; harmless.
- **`encode_images` per-image VAE loop** — MPS-motivated, candidate for separate perf proposal.

---

## Qualitative validation — Change #16 state vs #10 state (full runs, same dataset)

Manual per-checkpoint review of two full runs (small_run = #10 state, full_run = #16 state): 9 samples per epoch scored per aspect.

### Epochs 1–10 — visual accuracy trajectory

| Epoch | #10 | #16 | Epoch | #10 | #16 |
|-------|-----|-----|-------|-----|-----|
| 1 | 20% | 25% | 6 | 40% | **50%** |
| 2 | 25% | 25% | 7 | 40% | 50% |
| 3 | 20% | 20% | 8 | 45% | **60%** |
| 4 | 35% | 35% | 9 | 50% | **65%** |
| 5 | 35% | 30% | 10 | 50% | **70%** |

### Peak window (epochs 15–17) — per-aspect at epoch 15

| Aspect | #10 | #16 | Aspect | #10 | #16 |
|--------|-----|-----|--------|-----|-----|
| Hair color / style | 9/9, 9/9 | 9/9, 9/9 | Chest tattoos (6) | **0** | **3** |
| Face / Body | 9/9, 9/9 | 9/9, 9/9 | Back tattoos (2) | 0 | **2** |
| Arm tattoos (8) | 7 | **9** | Leg tattoos (7) | 2 | **5** |
| Position | 9 | 9 | **Visual accuracy** | **70%** | **90–95%** |

### Convergence point (the headline result)

| | #10 state | #16 state |
|---|-----------|-----------|
| Converged at | ~epoch 29 (chest tattoos not consistent until ~epoch 32) | **~epoch 20** |
| Steps to convergence | ~4,988–5,504 | **~3,440** |
| Quality at convergence | baseline | **better than #10's epoch 32** |

**#16 converges ~1,550 steps (~31%) earlier at higher final accuracy.**

### Interpretation — why would "speed" changes improve quality?

Both runs have txtfusion frozen (reverted #17), so the difference comes from #14 + #16 alone.
#14 (padding `% 256` → `% 32`) is numerically neutral when masking is correct. #16 changes
numerics: RoPE tables are now bf16-rounded instead of fp32-computed-then-downcast — a small,
consistent perturbation to rotation embeddings at every attention call. Plausible mechanisms
(unverified): regularization-like noise improving generalization on rare features, or simply a
different-but-better training trajectory.

### Implication for change #17 (txtfusion design question)

The strong quality gains came from code where txtfusion was **frozen** — so they neither require
nor rule out training it. They do show the pipeline learns these fine details *without* txtfusion
gradients, which slightly favors design (b): if a future experiment wants to test (a), it should be
an explicit quality A/B (keep #16 code + gradient fix, compare convergence at epoch 20/30).

---

## Open questions

### Change #17 — txtfusion design question (reverted, not closed)

**Is the dropped-gradient behaviour a bug, or is txtfusion intended to be frozen?**

- **(a) txtfusion should train.** Then #17 (or cheaper option C) is the right fix. Needs a VRAM check (~+670 MB peak).
- **(b) txtfusion is intended frozen.** Then the *real* bug is `lora_special.create_modules` matching on `"blocks"`, which accidentally attaches LoRA adapters to txtfusion. Fix: exclude txtfusion from the LoRA match list.

Suggested next step: ask the model author / check upstream Krea 2 training code whether txtfusion + its LoRA are meant to train, then pick (a) or (b).

### Scheduler weight-cache thrash

`_cached_device != timesteps.device` compares unindexed `cuda` vs `cuda:0`, so 3 weight tensors
are re-copied every call. Separate task per user decision. Only live when weighted timesteps
are enabled.

### `set_train_timesteps` sigmoid/shift branches

Rewritten (duplicate last element + argsort descending) so `timesteps`/`sigmas` lengths align.
Semantics differ from main's `ones`/`zeros` append. Not yet checked for correctness — only
matters for configs using `timestep_type: sigmoid|shift|flux_shift|lognorm_blend`.
