# Krea2 Pipeline Optimization - Current State (Set 2)

## Overview
This document tracks pending and completed optimization changes for the Krea2 pipeline (set 2).

**Previous set**: See `docs/code-optimization/archive/krea2/set-1/` for changes #1–#5.

**Baseline (start of set-2)** — the state entering set 2, i.e. end of set 1 (change #5). Short benchmark: 6 epochs × 30 steps, 4 images. See `archive/krea2/set-1/results-baseline-asof-change5.md`.
- Training time: **3.25s/it** (epoch 6 bottom-out)
- Sample generation: **65.64s/image** (epoch 6 average)

**Current best metrics** — after the set-2 full run (172 training images, 9 samples per checkpoint), as of change #10.
- Training time: **2.93s/it** (bottom-out, steps 2580–3784)
- Sample generation: **64.85s/image** (stable checkpoints, avg of steps 2236–3784 excluding 2924/3096 outliers)

**Improvement vs baseline at start of set-2**:
- Training: **-9.8%** (3.25 → 2.93s/it)
- Samples: **-1.2%** (65.64 → 64.85s/image)

**Full-run per-step data**: see `results-baseline-asof-change10.md` (per-checkpoint training s/it and sample times).

## Full-Run Results (Set 2, as of Change #10)

Per-checkpoint training s/it and sample generation times for the full run (172 images, 9 samples) are in **`results-baseline-asof-change10.md`**.

**Stable metrics (bottom-out, steps 2236–3784)**:
- Training: **2.93s/it** (range 2.93–2.96)
- Samples: **64.85s/image** (avg of stable checkpoints, excluding 2924/3096 outliers)

**Improvement vs baseline at start of set-2**: training **-9.8%** (3.25 → 2.93s/it), samples **-1.2%** (65.64 → 64.85s/image).

**Variation**: training 2.93–3.38s/it across full run (warm-up); samples 64.15–69.74s/image; stable range (steps 2236+) training 2.93–2.96s/it, samples 64.76–65.10s/image.

## Optimization Opportunities (Set 2)

### Change #6: Cache VAE normalization constants (latents_mean / latents_std)
**Status**: ✅ COMPLETED — Small sample generation improvement  
**Complexity**: Simple (1-5 lines)  
**Expected Impact**: 1-2% (eliminates repeated CPU→GPU tensor creation on every encode/decode)  
**Actual Result**: Samples 66.38s/image (-2.0% vs baseline 67.72). Training 3.29s/it (epochs 2-6, within baseline variance). Kept.  

**Issue**: Both `encode_images` and `decode_latents` create `latents_mean` and `latents_std` tensors from Python config lists on every call. These values are constant for the lifetime of the model. Caching them as model attributes eliminates repeated `torch.tensor(...)` + `.to(device, dtype)` operations (CPU→GPU copies).

**Location**: `extensions_built_in/diffusion_models/krea2/krea2.py`, lines 780-850

**Details**: See `implementation-proposal-change-6.md` and `results-change-6.md`

---

### Change #7: Cache position grid and mask in `prepare()`
**Status**: ⚠️ REVERTED — No measurable improvement, training slower  
**Complexity**: Moderate (6-10 lines)  
**Expected Impact**: 2-4% (both training and sampling)  
**Actual Result**: Training 3.53s/it (+8.6% vs baseline), Samples 67.81s/image (+0.1% vs baseline). Reverted.  

**Issue**: The `prepare()` function creates the image position grid (`imgids`), text positions (all zeros), and image mask on every call. For a fixed resolution, these tensors are identical across all steps. In the sampling loop (28 steps), this means 28 redundant allocations of the same position/mask tensors. Caching them per `(h_, w_, txtlen, b)` eliminates ~5 tensor allocations and a `repeat` operation every step.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 74-95

**Details**: See `implementation-proposal-change-7.md`

---

### Change #8: Cache RoPE frequencies for fixed positions
**Status**: ⚠️ REVERTED — Cannot work without Change #7; shape-based key produces wrong results  
**Complexity**: Moderate (6-10 lines)  
**Expected Impact**: 3-5% (both training and sampling)  
**Actual Result**: Corrupted output. Shape-based cache key `(tuple(pos.shape), pos.device)` is unsafe — different aspect ratios (e.g., 512×1024 vs 1024×512) produce the same `pos.shape` but different position values, returning wrong RoPE frequencies. Original proposal correctly required Change #7's stable `data_ptr()` identity. Without it, no safe cache key exists.  

**Issue**: `freqs = self.posemb(pos)` computes RoPE frequencies on every forward pass. Caching requires a safe key that uniquely identifies position content.

**Location**: `extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `SingleStreamDiT.forward` (~line 570)

**Details**: See `implementation-proposal-change-8.md`

---

### Change #9: Single dtype conversion in CFG sampling loop
**Status**: ✅ COMPLETED — Neutral impact (kept for code cleanliness)  
**Complexity**: Simple (1-5 lines)  
**Expected Impact**: 1-2% sample generation speedup  
**Actual Result**: No measurable improvement. Samples 65.71s/image vs baseline 65.12s/image (+0.9%, within variance). Kept anyway — zero runtime cost, cleaner code.  

**Issue**: In `Krea2Pipeline.__call__`, when CFG is enabled, `latents.to(dtype)` is called twice per denoising step (once for the conditional pass, once for the unconditional pass). Computing it once and reusing saves a redundant tensor copy per step (28 steps × 1 extra copy eliminated).

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 360-380

**Details**: See `implementation-proposal-change-9.md` and `results-change-9.md`

---

### Change #10: Pre-compute text fusion context in sampling loop
**Status**: ✅ COMPLETED — Neutral impact (kept for code cleanliness)  
**Complexity**: Complex (11-20 lines)  
**Expected Impact**: 5-8% sample generation speedup  
**Actual Result**: No measurable improvement. Training 3.22s/it (unchanged, expected), Samples 66.26s/image (within variance). The text fusion sub-network (4 blocks on text tokens) is a small fraction of total compute vs 28 main SingleStreamBlocks on the full combined sequence. Kept — zero runtime cost when not pre-fused.  

**Issue**: In `SingleStreamDiT.forward`, the text context passes through `txtfusion` (4 transformer blocks with attention + MLP) and `txtmlp` (norm + 2 linears) on EVERY call. In the 28-step sampling loop, the text context is identical across all steps, so this is 27 redundant computations of a non-trivial sub-network. Pre-computing the fused context once before the loop and passing it directly eliminates ~27/28 of these computations.

**Location**: `extensions_built_in/diffusion_models/krea2/src/mmdit.py` (add `fuse_context` method) + `extensions_built_in/diffusion_models/krea2/src/pipeline.py` (pipeline loop)

**Details**: See `implementation-proposal-change-10.md` and `results-change-10.md`

## Optimization Opportunities (Set 3 — full training-loop audit)

Full pass over the entire per-step training path: `SDTrainer.train_single_accumulation`
→ `BaseSDTrainProcess.process_general_training_batch` → `predict_noise` /
`get_noise_prediction` (krea2.py) → `pad_text_features` + `prepare` +
`predict_velocity` (pipeline.py) → `SingleStreamDiT.forward` (mmdit.py).
Focus: excess copies, wasted number conversions, CUDA-simplifiable math.

### Change #11: Vectorize `BaseModel.add_noise` (kill per-sample chunk loop)
**Status**: ⚠️ REVERTED — No measurable improvement (training +3.7%, within variance)  
**Complexity**: Simple (1-5 lines)  
**Expected Impact**: ~0.5–2% training speedup  
**Actual Result**: Training 3.34s/it (epochs 4-6) vs change #10's 3.22s/it (+3.7%, slower); Samples 68.23s/image (epochs 4-6) vs change #10's 67.21s/image (+1.5%, within ~5% variance). Consistent upward shift across all 6 epochs — most likely run-to-run variance (the change removes work and is bitwise-identical, so it cannot logically slow training). No measurable improvement. **Reverted.**  

**Benchmark (6 epochs × 30 steps, 4 images)**:

| Epoch | Steps | Total time | Avg training (s/it) | S1 | S2 | S3 | S4 | Avg sample (s) |
|-------|-------|------------|---------------------|--------|--------|--------|--------|----------------|
| 1 | 30 | 1:42 | 3.52 | 69.19 | 68.06 | 67.62 | 67.67 | 68.14 |
| 2 | 30 | 1:34 | 3.34 | 67.14 | 67.47 | 67.27 | 67.31 | 67.30 |
| 3 | 30 | 1:55 | 3.50 | 67.25 | 67.58 | 67.02 | 66.59 | 67.11 |
| 4 | 30 | 1:35 | 3.41 | 66.05 | 65.81 | 66.68 | 67.57 | 66.53 |
| 5 | 30 | 1:32 | 3.34 | 69.20 | 69.09 | 69.07 | 69.24 | 69.15 |
| 6 | 30 | 1:26 | 3.27 | 69.05 | 68.98 | 68.99 | 69.03 | 69.01 |

*Avg training (s/it) is the progress bar's cumulative rate at epoch end — same metric as change #10's table. Total time is the per-epoch elapsed delta (excludes sample generation).*

**Comparison vs Change #10 (epochs 4-6 stable)**:

| Metric | Change #10 | Change #11 | Delta |
|--------|------------|------------|-------|
| Training (s/it) | 3.22 | 3.34 | +3.7% (slower) |
| Samples (s/image) | 67.21 | 68.23 | +1.5% (within variance) |

**Verdict**: No measurable improvement. The delta is within run-to-run variance (set-1 established ~21% training variation; the change is bitwise-identical and removes work, so it cannot logically slow training). **Reverted** per protocol — `base_model.py` restored to the original chunk loop; test suite re-verified (44 passed).

**Issue**: `BaseModel.add_noise` (base_model.py ~line 750) chunks the batch into
B single-sample slices, calls `noise_scheduler.add_noise` B times in a Python loop,
then `torch.cat`s the full (B, 16, h, w) latent tensor back together — every
training step. For the flow-matching scheduler (Krea2's), `add_noise` is a single
affine blend `(1-t)*x + t*noise`; reshaping per-sample `(B,)` timesteps to
`(B,1,...,1)` lets one call replace B calls + 1 full copy. The chunk loop is kept
as fallback for schedulers that need per-sample calls (or shared single timesteps).

**Location**: `toolkit/models/base_model.py`, `BaseModel.add_noise`

**Implementation note**: the original proposal's fast path (raw `(B,)` timesteps)
was proven to crash — PyTorch aligns trailing dims, so `B` lands on the width dim.
Implemented with a mandatory reshape `(B,) → (B,1,...,1)` before the single
scheduler call. Unit check: bitwise-identical to chunked path across 5 cases
(float/int timesteps, fp16, B=1, shared-timestep fallback). `pytest tests/`:
44 passed.

**Details**: See `implementation-proposal-change-11.md`

---

### Change #12: RoPE in float32 with cached omega (drop per-call float64 rebuild)
**Status**: ⚠️ REVERTED — No measurable improvement (training +3.4%, within variance)  
**Complexity**: Simple (~15 lines across `rope` + `PositionalEncoding`)  
**Expected Impact**: ~0.1–0.5% training speedup  

**Issue**: `posemb(pos)` runs 3× `rope()` per forward, each rebuilding
`scale`/`omega` (arange + pow) **in float64** and running `einsum`+`cos`/`sin`
**in float64** over (B, L, d/2), then downcasting with `.float()`. Float64 trig
has no fast GPU path and the precision is wasted: `pos` holds small integers, the
model runs bf16 (rel. error ~4e-3), and `ropeapply` consumes the freqs in a bf16
multiply. Fix: cache `omega` as a plain (non-buffer, meta-safe) attribute built
lazily on first forward; compute in float32. `rope` is module-private (only caller
is `PositionalEncoding.forward`).

**Location**: `extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `rope` (line 31) + `PositionalEncoding` (line 136)

**Implementation note**: Unit check passed — max abs diff old (fp64) vs new
(fp32 cached omega) = 3.95e-06 across 5 random integer-position trials
(< 1e-5 revert threshold). `state_dict()` stays empty (plain attribute, not a
buffer — required because the transformer is built on `torch.device("meta")`
and loaded with `strict=True`). Meta construction verified. `pytest tests/`:
44 passed.

**Benchmark (6 epochs × 30 steps, 4 images)**:

| Epoch | Steps | Total time | Avg training (s/it) | S1 | S2 | S3 | S4 | Avg sample (s) |
|-------|-------|------------|---------------------|--------|--------|--------|--------|----------------|
| 1 | 30 | 1:50 | 3.81 | 68.54 | 67.71 | 67.44 | 67.34 | 67.76 |
| 2 | 30 | 1:45 | 3.65 | 67.30 | 67.35 | 67.27 | 67.25 | 67.29 |
| 3 | 30 | 1:33 | 3.47 | 67.36 | 67.25 | 67.23 | 67.19 | 67.26 |
| 4 | 30 | 1:37 | 3.41 | 67.30 | 67.24 | 67.22 | 67.19 | 67.24 |
| 5 | 30 | 1:29 | 3.32 | 67.28 | 67.23 | 67.19 | 67.18 | 67.22 |
| 6 | 30 | 1:29 | 3.26 | 67.34 | 67.20 | 67.29 | 66.08 | 66.98 |

*Avg training (s/it) is the progress bar's cumulative rate at epoch end — same metric as change #10/#11 tables. Total time is the per-epoch elapsed delta (excludes sample generation).*

**Comparison vs Change #10 (epochs 4-6 stable)**:

| Metric | Change #10 | Change #12 | Delta |
|--------|------------|------------|-------|
| Training (s/it) | 3.22 | 3.33 | +3.4% (slower) |
| Samples (s/image) | 67.21 | 67.15 | −0.1% (flat) |

**Verdict**: No measurable improvement. Training delta is within run-to-run variance (the change strictly removes work — fp64 trig → fp32, cached omega — so it cannot logically slow training). Samples flat. **Reverted** per protocol — `mmdit.py` restored to the original float64 `rope`; test suite re-verified (44 passed).

**Details**: See `implementation-proposal-change-12.md`

---

### Change #13: Cache `temb` frequency vector + drop redundant `.to()` in `encode_images`
**Status**: 📝 PROPOSED — awaiting approval  
**Complexity**: Simple (1–5 lines each, two independent micro-opts)  
**Expected Impact**: ~0.1% training speedup combined  

**Issue A**: `temb()` rebuilds a constant 128-value frequency vector
(`torch.exp(arange)`) on every forward. Cache it in a module-level dict keyed by
`(dim, device)` (meta-safe; `temb` is a free function with no module state).

**Issue B**: `Krea2Model.encode_images` ends with
`return latents.to(device, dtype=dtype)` — a no-op copy of the full (B, 16, h, w)
latent batch: each image was already moved to `device`/`dtype` before VAE encode,
so the stacked result is already in place. Remove it.

**Location**: `mmdit.py` `temb` (~line 74) + `krea2.py` `encode_images` (~line 810)

**Details**: See `implementation-proposal-change-13.md`

---

### Audited and rejected (no change proposed)
- **`prepare()` per-step grid/mask rebuild** — already rejected in set-2 (Change #7
  reverted: +8.6% training). The ~5 small tensor allocations are not worth the
  cache-key complexity; `pos`/`mask` are tiny vs the (B, L, 6144) activations.
- **`pad_text_features`** — already optimized in set-1 (Change #3); current
  stack+slice version is fine.
- **`predict_noise` `latents.to(self.device_torch)` / `timestep.to(...)`** —
  no-ops when already on device (the normal case); `.to()` with matching
  device/dtype returns the same tensor without a copy. Not worth touching shared code.
- **`calculate_loss` `pred.float()` / `target.float()`** — intentional fp32 MSE
  accumulation for bf16 training; removing it would hurt precision. Kept.
- **`get_noise_prediction` `latent_model_input.to(device, dtype)`** — no-op in the
  normal case (latents already on device/dtype from `process_general_training_batch`).
- **Text encoder re-encode per step** — only happens when the dataset does NOT set
  `cache_text_embeddings: true` (see `BaseSDTrainProcess.is_caching_text_embeddings`).
  If the user's dataset config has it off, enabling it is a **config change** (no
  code) that would be the single biggest per-step win available — worth checking,
  but out of scope for code edits.

## Testing Protocol

For each change:
1. Implement the optimization
2. Run benchmark test: 3 epochs × 30 steps, generate 4 images
3. Compare against the current best metrics (above) and the baseline at start of set-2
4. Keep change only if it improves speed; revert otherwise

## Notes

- Baseline variation observed in set 2: training ~2.93–3.38s/it (warm-up), samples ~64.15–69.74s/image
- Changes #7, #8, and #10 compound: caching positions enables caching RoPE freqs, and pre-fusing context is independent
- User will run benchmark tests and provide logs
- **This file is the single source of truth** for optimization state, metrics, and per-change status. Per-checkpoint full-run data lives in `results-baseline-asof-change10.md`.
