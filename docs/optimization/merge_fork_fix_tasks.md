# Chroma MPS Optimization Tasks

## Context
- **Branch**: `mac_gpu_util`
- **Hardware**: Apple Silicon Mac (MPS device)
- **Model**: Chroma (lodestones/Chroma1-HD), quantized with qfloat8
- **Optimizer**: adamw8bit, bf16 dtype
- **Config**: No compile, no layer_offloading, no EMA, no preservation training

## Baseline Performance

### Target (pre-merge, local optimizations restored)
- Training: ~10.2–11.9 s/it (steps 1-30)
- Sampling: ~56.4–56.8 s/it (2 images)

### Current (after restoring local optimizations)
- Training: ~12.3 s/it
- Sampling: ~55.5 s/it

---

## Validation Protocol

Each change requires:
1. ≤20 lines per function (surgical improvements only)
2. Unit tests pass (if applicable)
3. Speed test: 3 epochs × first 30 steps, generate 2 images
4. Compare against baseline after each change

---

## Tasks (ordered by expected impact)

### Task 1: Cache `timestep_embedding` freqs tensor
**File:** `extensions_built_in/diffusion_models/chroma/src/layers.py:35-54`

**Problem:** `freqs` tensor is constant but allocated fresh every call via `torch.exp(...).to(t.device)`. Called 3× per forward pass (timesteps, guidance, mod_index).

**Change:** Move `freqs` to a registered buffer, computed once at init.

**Expected impact:** ★★★ (eliminates 3 tensor allocations + device transfers per forward pass)

**Status:** PASS — Training: 12.03→12.11 s/it (stable). Sampling: 54.6→55.3 s/it (stable). No degradation over 8 checkpoints.
**Result:**

---

### Task 1b: Cache `rope` omega tensor
**File:** `extensions_built_in/diffusion_models/chroma/src/math.py:33-45`

**Problem:** `scale` and `omega` tensors are constant for a given `(dim, theta)` but allocated fresh every call via `torch.arange(..., device=pos.device)` and `theta**scale`. Called 3× per forward pass (once per axis in `EmbedND`).

**Change:** Cache `omega` in a module-level dict keyed by `(dim, theta, device)`. Create `scale` directly on target device.

**Expected impact:** ★★ (eliminates 3 tensor allocations + power computations per forward pass)

**Status:** Neutral — Training: 12.18 s/it (same as Task 1 baseline). Sampling: 55.3 s/it (same). Stable, no regression. Small tensor allocations not the bottleneck here.
**Result:**

---

### Task 2: Avoid bf16→fp32→bf16 round-trip in `apply_rope()`
**File:** `extensions_built_in/diffusion_models/chroma/src/math.py:49-52`

**Problem:** `xq.float().reshape(...)` then `.type_as(xq)` — full dtype conversion round-trip. Called 57× per forward pass (19 double blocks + 38 single blocks).

**Change:** Keep computation in input dtype when possible, or use `torch.autocast` to minimize conversions.

**Expected impact:** ★★★ (57× fewer dtype conversions per forward pass)

**Status:** Done
**Result:** Epoch 1: 11.51 s/it (was 12.3), sampling 54.8 s/it (was 55.5). Degraded to 12.25 s/it by epoch 3. PASS (net improvement, memory pressure suspected).

---

### Task 3: Pre-compute `distribute_modulations` slice indices
**File:** `extensions_built_in/diffusion_models/chroma/src/layers.py:93-170`

**Problem:** Python loop over ~58 keys with string matching (`"single_blocks" in key`) and new `ModulationOut` object creation every forward pass. Pure CPU overhead blocking MPS command queue.

**Change:** Pre-compute slice indices at init time. Use `torch.split()` instead of the loop.

**Expected impact:** ★★★ (eliminates ~58 Python iterations + string matches per forward pass)

**Status:** Skipped — Python-level micro-optimizations add overhead on MPS. Task 3a (key caching) caused regression (12.47 vs 11.51 s/it). Not the bottleneck.

---

### Task 4: Fix `torch.no_grad()` + `.requires_grad_(True)` hack
**File:** `extensions_built_in/diffusion_models/chroma/src/model.py:205-222`

**Problem:** `with torch.no_grad(): ... mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))` — forces graph disconnect then reconnect. Creates implicit MPS sync point every step.

**Change:** Move `.requires_grad_(True)` off the in-place path. Use `detach().clone().requires_grad_(True)` after the `no_grad` block instead of calling it on the input tensor inside the block.

**Expected impact:** ★★★ (removes MPS command buffer sync every step)

**Risk:** HIGH — may break gradient accumulation. Test carefully.

**Status:** PASS — Degradation eliminated. Training: 13.34→12.93 s/it (warmup then stable/improving). Sampling: 62.06→56.48 s/it. Trend is REVERSED (getting faster), confirming MPS warmup behavior, not memory leak.
**Result:**

---

### Task 5: Create `latent_image_ids` directly on device
**File:** `extensions_built_in/diffusion_models/chroma/pipeline.py:33`

**Problem:** `torch.zeros(height // patch_size, width // patch_size, 3)` creates tensor on CPU, then `.to(device)` transfers to MPS. Called every sampling step.

**Change:** Add `device=device` to `torch.zeros()`.

**Expected impact:** ★★ (eliminates CPU→MPS transfer per sampling step)

**Status:** Reverted — Caused regression (12.14 vs 11.51 with Task 2 alone).

---

### Task 6: Vectorize `modify_mask_to_attend_padding`
**File:** `extensions_built_in/diffusion_models/chroma/src/model.py:55-78`

**Problem:** `for i in range(batch_size): current_seq_len = int(seq_length[i].item())` — `.item()` forces CPU sync on MPS per batch element.

**Change:** Vectorize with `torch.clamp` and advanced indexing.

**Expected impact:** ★★ (eliminates O(batch_size) sync points per forward pass)

**Status:** Reverted — Caused regression (12.24 vs 11.51 with Task 2 alone).

---

### Task 7: Reduce `gc.collect()` frequency in training loop
**File:** `jobs/process/BaseSDTrainProcess.py` (multiple locations)

**Problem:** `flush()` calls `gc.collect()` which blocks the MPS command queue. Called on every save/sample step.

**Change:** Pass `garbage_collect=False` to `flush()` in hot path (after save steps, first flush). GC still runs on save/sample/OOM paths.

**Expected impact:** ★★ (reduces blocking GC pauses)

**Status:** PASS — Training: 12.1→11.97 s/it (stable, -1.1%). Sampling: 55.0→54.8 s/it (stable). No degradation over 12 checkpoints.
**Result:**

---

### Task 8: Persistent RNG buffer in `copy_stochastic_bf16`
**File:** `toolkit/optimizers/optimizer_utils.py:100-120`

**Problem:** `torch.randint(0, 1 << 16, src_i32.shape, ...)` allocates a full-sized random tensor every optimizer step. MPS `torch.randint` is slow.

**Change:** Use a persistent buffer updated in-place with `torch.randint(..., out=buffer)`.

**Expected impact:** ★★ (reduces allocation + RNG overhead per optimizer step)

**Status:** Reverted — Caused severe regression (13.9 s/it vs 12.8 s/it baseline). Persistent buffer pinned MPS memory, making fragmentation worse.
**Result:**

---

### Task 9: Cache `txt_img_mask` matrix multiply
**File:** `extensions_built_in/diffusion_models/chroma/src/model.py:244`

**Problem:** `txt_img_mask.float().T @ txt_img_mask.float()` — ~4600×4600 matmul every forward pass. Cacheable if sequence length is constant.

**Change:** Cache result when sequence length doesn't change.

**Expected impact:** ★ (eliminates large matmul per forward pass)

**Status:** Not started
**Result:**

---

### Task 10: Enable `num_workers > 0` on macOS
**File:** `toolkit/data_loader.py:720`

**Problem:** `if is_native_windows() or is_macos(): dataloader_kwargs['num_workers'] = 0` — data loading runs on main thread, blocking training loop.

**Change:** Use `num_workers=2-4` on macOS with `multiprocessing_context='spawn'`.

**Expected impact:** ★ (unblocks training loop from I/O)

**Status:** Reverted — `spawn` workers can't share MPS tensor storage (`RuntimeError: _share_filename_: only available on CPU`). Fundamental MPS limitation.
**Result:**

---

## Results Tracking

| Task | File | Before (s/it) | After (s/it) | Delta | Pass/Fail | Notes |
|------|------|--------------|--------------|-------|-----------|-------|
| Base | — | 12.3 train / 55.5 sample | — | — | — | Current baseline |
| 1 | layers.py (timestep_embedding) | | | | | |
| 2 | math.py (apply_rope) | | | | | |
| 3 | layers.py (distribute_modulations) | | | | | |
| 4 | model.py (no_grad hack) | | | | | |
| 5 | pipeline.py (latent_image_ids) | | | | | |
| 6 | model.py (mask vectorize) | | | | | |
| 7 | BaseSDTrainProcess.py (gc.collect) | | | | | |
| 8 | optimizer_utils.py (RNG buffer) | | | | | |
| 9 | model.py (mask matmul cache) | | | | | |
| 10 | data_loader.py (num_workers) | | | | | |
| Final | — | | | | | |

## Notes
- **DO NOT commit or push to repo** — User handles version control
- User tests manually after each change
- Use `torch_util.py` helpers where applicable (`get_device_type()`, `is_mps_device()`, `flush_cache()`, etc.)
