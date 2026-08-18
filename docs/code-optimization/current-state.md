# Krea2 Pipeline Optimization - Current State (Set 2)

## Overview
This document tracks pending and completed optimization changes for the Krea2 pipeline (set 2).

**Previous set**: See `docs/code-optimization/archive/krea2/set-1/` for changes #1–#5.

**Current best metrics** (after set-2 full-run validation — 172 training images, 9 samples):
- Training time: **2.93s/it** (bottom-out from full run, steps 2580–3784)
- Sample generation: **64.85s/image** (stable checkpoints, avg of steps 2236–3784 excluding 2924/3096 outliers)

**Set-1 best metrics** (for comparison):
- Training time: **3.03s/it**
- Sample generation: **65.12s/image**

**Cumulative improvement (set-1 → set-2)**:
- Training: **-3.3%** (3.03 → 2.93s/it)
- Samples: **-0.4%** (65.12 → 64.85s/image)

**Benchmark baseline** (6 epochs × 30 steps, 4 images — see `results-baseline-asof-change5.md`):
- Training time: **3.25s/it** (epoch 6 bottom-out)
- Sample generation: **65.64s/image** (epoch 6 average)

## Full-Run Benchmark Results (172 images, 9 samples — Set 2 as of Change #10)

### Training Time (s/it)

| Steps | Time | s/it | Notes |
|-------|------|------|-------|
| 172 | 9:37 | 3.38 | Warm-up |
| 344 | 18:31 | 3.24 | |
| 516 | 26:50 | 3.13 | |
| 688 | 35:10 | 3.07 | |
| 860 | 43:32 | 3.04 | |
| 1032 | 51:48 | 3.02 | |
| 1204 | 1:00:11 | 3.00 | |
| 1376 | 1:08:29 | 2.99 | |
| 1548 | 1:16:51 | 2.98 | |
| 1720 | 1:25:10 | 2.97 | |
| 1892 | 1:33:30 | 2.97 | |
| 2064 | 1:41:45 | 2.96 | |
| 2236 | 1:50:08 | 2.96 | |
| 2408 | 1:58:25 | 2.95 | |
| 2580 | 2:06:47 | 2.95 | |
| 2752 | 2:15:03 | 2.95 | |
| 2924 | 2:23:22 | 2.94 | |
| 3096 | 2:31:43 | 2.94 | |
| 3268 | 2:39:57 | 2.94 | |
| 3440 | 2:48:17 | 2.94 | |
| 3612 | 2:56:39 | 2.94 | |
| 3784 | 3:04:59 | 2.93 | Bottom-out |

### Sample Generation (s/image)

| Steps | Avg Sample (s) | Notes |
|-------|----------------|-------|
| 172 | 69.74 | Warm-up |
| 344 | 67.18 | |
| 516 | 65.01 | |
| 688 | 64.99 | |
| 860 | 64.81 | |
| 1032 | 64.85 | |
| 1204 | 64.87 | |
| 1376 | 65.05 | |
| 1548 | 64.86 | |
| 1720 | 64.84 | |
| 1892 | 64.89 | |
| 2064 | 65.05 | |
| 2236 | 64.76 | |
| 2408 | 64.90 | |
| 2580 | 64.78 | |
| 2752 | 65.10 | |
| 2924 | 64.15 | Outlier (lower) |
| 3096 | 64.15 | Outlier (lower) |
| 3268 | 64.90 | |
| 3440 | 65.04 | |
| 3612 | 64.98 | |
| 3784 | 64.92 | |

### Stable Metrics (Bottom-out, steps 2236–3784)

- **Training**: **2.93s/it** (range 2.93–2.96)
- **Samples**: **64.85s/image** (avg of stable checkpoints, excluding 2924/3096 outliers)

### Comparison vs Set-1 Best

| Metric | Set-1 Best | Set-2 (Change #10) | Delta |
|--------|------------|--------------------|-------|
| Training (s/it) | 3.03 | 2.93 | **-3.3%** |
| Samples (s/image) | 65.12 | 64.85 | **-0.4%** |

### Comparison vs Original Baseline (asof change 5)

| Metric | Baseline | Set-2 (Change #10) | Delta |
|--------|----------|--------------------|-------|
| Training (s/it) | 3.25 | 2.93 | **-9.8%** |
| Samples (s/image) | 65.64 | 64.85 | **-1.2%** |

### Baseline Variation Analysis

- Training time varies from 2.93s to 3.38s across full run (warm-up effect)
- Sample generation varies from 64.15s to 69.74s across full run (warm-up + variance)
- Stable range (steps 2236+): training 2.93–2.96s/it, samples 64.76–65.10s/image

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
**Status**: 📝 PROPOSED — awaiting approval  
**Complexity**: Simple (1-5 lines)  
**Expected Impact**: ~0.5–2% training speedup  

**Issue**: `BaseModel.add_noise` (base_model.py ~line 750) chunks the batch into
B single-sample slices, calls `noise_scheduler.add_noise` B times in a Python loop,
then `torch.cat`s the full (B, 16, h, w) latent tensor back together — every
training step. For the flow-matching scheduler (Krea2's), `add_noise` is a single
affine blend `(1-t)*x + t*noise` that broadcasts per-sample `(B,)` timesteps
correctly over the whole batch, so one call replaces B calls + 1 full copy. The
chunk loop is kept as fallback for schedulers that need per-sample calls.

**Location**: `toolkit/models/base_model.py`, `BaseModel.add_noise`

**Details**: See `implementation-proposal-change-11.md`

---

### Change #12: RoPE in float32 with cached omega (drop per-call float64 rebuild)
**Status**: 📝 PROPOSED — awaiting approval  
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
3. Compare against current best (set-1 results)
4. Keep change only if improvement >5%

## Notes

- Set-1 established that baseline variation is ~21% training, ~5% samples
- Only changes with >5% improvement should be kept (per set-1 verdict criteria)
- Changes #7, #8, and #10 compound: caching positions enables caching RoPE freqs, and pre-fusing context is independent
- User will run benchmark tests and provide logs
- **This file is the single source of truth** for optimization state, metrics, and benchmark results. All data from `results-baseline-asof-change10.md` has been consolidated here.
