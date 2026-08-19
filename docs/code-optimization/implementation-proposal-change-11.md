# Implementation Proposal #11: Vectorize `BaseModel.add_noise` (kill per-sample chunk loop)

## Status
⚠️ REVERTED — No measurable improvement (training +3.7%, within variance)

> **⚠️ Design correction found during implementation:** the original fast path
> (`self.noise_scheduler.add_noise(original_samples, noise, timesteps)` with raw
> `(B,)` timesteps) **crashes**: PyTorch aligns trailing dims, so the `B` dim lands
> on the latent *width* dim → `RuntimeError: The size of tensor a (4) must match
> the size of tensor b (32) at non-singleton dimension 3`. The implemented fast
> path reshapes timesteps `(B,)` → `(B, 1, ..., 1)` before the single scheduler
> call. Verified bitwise-identical to the chunked loop (see Validation Results).

## Complexity
Simple (1-5 lines changed)

## Expected Impact
~0.5–2% training speedup (removes a per-sample Python loop + B-1 tiny kernel launches
+ one `torch.cat` of the full latent tensor, every training step)

## Issue Description

There are **two** functions named `add_noise`; only the second is modified.

1. **The scheduler's** `CustomFlowMatchEulerDiscreteScheduler.add_noise`
   (`toolkit/samplers/custom_flowmatch_sampler.py`) — the actual math, **unchanged**:

   ```python
   def add_noise(self, original_samples, noise, timesteps):
       t_01 = (timesteps / 1000).to(original_samples.device)
       noisy_model_input = (1.0 - t_01) * original_samples + t_01 * noise
       return noisy_model_input
   ```

2. **The wrapper** `BaseModel.add_noise` (`toolkit/models/base_model.py`, ~line 750)
   — what the training loop calls via `self.sd.add_noise(...)`. It runs the
   scheduler's math **once per sample** in a Python loop:

   ```python
   def add_noise(self, original_samples, noise, timesteps, **kwargs):
       original_samples_chunks = torch.chunk(original_samples, original_samples.shape[0], dim=0)
       noise_chunks = torch.chunk(noise, noise.shape[0], dim=0)
       timesteps_chunks = torch.chunk(timesteps, timesteps.shape[0], dim=0)

       if len(timesteps_chunks) == 1 and len(timesteps_chunks) != len(original_samples_chunks):
           timesteps_chunks = [timesteps_chunks[0]] * len(original_samples_chunks)

       noisy_latents_chunks = []
       for idx in rgit s          original_samples_chunks[idx], noise_chunks[idx], timesteps_chunks[idx])
           noisy_latents_chunks.append(noisy_latents)

       noisy_latents = torch.cat(noisy_latents_chunks, dim=0)
       return noisy_latents
   ```

So the wrapper runs B separate `(1-t)*x + t*noise` ops on 1-sample slices and
re-concatenates them — B-1 extra kernel launches, a Python loop, and one full
`torch.cat` copy of the (B, 16, h, w) latent tensor per step. The chunking exists
only to support schedulers whose `add_noise` is not batch-broadcastable (e.g.
DDPM with per-sample sigma handling). For the flow-matching scheduler it is pure
overhead.

## Proposed Change

Add a batch-aware fast path in `BaseModel.add_noise` that calls the scheduler
**once on the full batch** when the timestep tensor already matches the latent
batch (the normal training case), and keeps the chunk loop as fallback:

```python
def add_noise(self, original_samples, noise, timesteps, **kwargs):
    # Fast path (Change #11): flow-matching / DDPM-style schedulers broadcast
    # per-sample timesteps over the whole batch. Reshape (B,) -> (B,1,...,1)
    # so one scheduler call replaces the per-sample chunk loop + full torch.cat.
    if isinstance(timesteps, torch.Tensor) and timesteps.dim() == 1 \
            and timesteps.shape[0] == original_samples.shape[0]:
        t = timesteps.reshape(original_samples.shape[0], *([1] * (original_samples.dim() - 1)))
        return self.noise_scheduler.add_noise(original_samples, noise, t)

    original_samples_chunks = torch.chunk(...)
    # ...existing chunk loop unchanged...
```

**Why this is safe for Krea2 (and other flow-matching archs):**
- `timesteps` arrives as `(B,)` float (from `noise_scheduler.timesteps[timestep_indices]`).
- **The reshape is mandatory, not optional.** PyTorch broadcasting aligns *trailing*
  dims: raw `(B,)` against `(B, C, h, w)` aligns `B` with the width dim and raises
  (verified empirically). Reshaping to `(B, 1, ..., 1)` makes the scheduler's
  `(t / 1000) * x` scale each sample by its own `t_i`.
- The chunked path produces exactly `(1 - t_i) * x_i + t_i * n_i` per sample —
  bit-identical math to the batched op (same elementwise ops, same order).
- Non-flow-matching schedulers (DDPM etc.) with `timesteps.shape[0] == B` would also
  take the fast path; DDPM's `add_noise` is itself batch-broadcastable in diffusers
  (`sqrt_alpha * sample + sqrt_one_minus_alpha * noise` with sigma broadcast), so the
  result is identical there too (verified bitwise for index-based math). If any exotic
  scheduler breaks, the fallback loop remains for `timesteps.shape[0] != B` (e.g.
  single shared timestep) or non-1-D timesteps.

**Precision:** identical — same elementwise arithmetic, no dtype change
(`t_01` is float32 in both paths; the blend upcasts to the sample dtype as before).

## Location
`toolkit/models/base_model.py`, `BaseModel.add_noise` (~line 750)

## Lines Changed
~6 (one `if` + reshape + early return inserted at top of method)

## Validation Plan
1. Unit check: for a random (B=4, 16, 32, 32) latent + noise + (B,) timesteps,
   assert `torch.allclose` between old chunked result and new batched result
   (expect exact equality).
2. Speed test per protocol: 8 epochs × 30 steps, generate 2 images.
   Expect small but consistent s/it reduction; sample time unchanged (this path
   is training-only).

## Validation Results (pre-benchmark)

**Unit check — PASSED.** Ran old chunked path vs new fast path through the real
`CustomFlowMatchEulerDiscreteScheduler.add_noise`; all cases **bitwise equal**
(`torch.equal`):

| Case | Setup | Result |
|------|-------|--------|
| 1 | B=4, (B,) float timesteps — normal training case | ✅ bitwise equal |
| 2 | B=4, shared `(1,)` timestep — fallback path | ✅ bitwise equal (unchanged) |
| 3 | B=1, `(1,)` timestep — fast path (B==1) | ✅ bitwise equal |
| 4 | B=4, fp16 latents — training dtype | ✅ bitwise equal |
| 5 | B=4, int timesteps (IntTensor) | ✅ bitwise equal |

Also verified DDPM-style index-based `add_noise` math is bitwise identical with
reshaped timesteps (fast path is safe for non-flow-matching schedulers too).

**Scheduler blast-radius check — all families bitwise-identical.** The fast path
is scheduler-agnostic (triggers for any 1-D per-sample timesteps), so every
`add_noise` family in the repo was tested against the chunked path:

| Family | `add_noise` math | Training? | Result |
|--------|------------------|-----------|--------|
| Flow-matching (`CustomFlowMatchEulerDiscreteScheduler`, `HidreamO1FlowmatchScheduler`) | `(1-t)*x + t*noise` | ✅ all 23 models | bitwise equal |
| DDPM/LCM direct-indexing (`CustomLCMScheduler`) | `sqrt(α)[t]*x + sqrt(1-α)[t]*noise` | ❌ inference-only | bitwise equal |
| DPMSolver `index_for_timestep` (omnigen2, fm_solvers_unipc) | index loop → sigma broadcast | ❌ inference-only | bitwise equal (maxdiff 0.0) |

`FlashFlowMatchEulerDiscreteScheduler` has no `add_noise` (only `scale_noise`) —
never reaches this path. The DPMSolver family iterates timesteps via
`index_for_timestep`; the reshape makes each element `(1,1,...,1)`, which still
resolves to the same index via `(schedule == t).nonzero().item()`. **Only
flow-matching is used in training, so that's the only family with practical impact.**

**Test suite — PASSED.** `pytest tests/` → 44 passed (2.76s).

**Note:** the original proposal's fast path (raw `(B,)` timesteps, no reshape)
was proven to raise `RuntimeError` before implementation; the reshape fix above
is what was implemented.

**Benchmark — COMPLETED (6 epochs × 30 steps, 4 images).** No measurable improvement.

| Epoch | Steps | Total time | Avg training (s/it) | S1 | S2 | S3 | S4 | Avg sample (s) |
|-------|-------|------------|---------------------|--------|--------|--------|--------|----------------|
| 1 | 30 | 1:42 | 3.52 | 69.19 | 68.06 | 67.62 | 67.67 | 68.14 |
| 2 | 30 | 1:34 | 3.34 | 67.14 | 67.47 | 67.27 | 67.31 | 67.30 |
| 3 | 30 | 1:55 | 3.50 | 67.25 | 67.58 | 67.02 | 66.59 | 67.11 |
| 4 | 30 | 1:35 | 3.41 | 66.05 | 65.81 | 66.68 | 67.57 | 66.53 |
| 5 | 30 | 1:32 | 3.34 | 69.20 | 69.09 | 69.07 | 69.24 | 69.15 |
| 6 | 30 | 1:26 | 3.27 | 69.05 | 68.98 | 68.99 | 69.03 | 69.01 |

**Stable (epochs 4-6)**: training **3.34s/it**, samples **68.23s/image**.

**Comparison vs Change #10 (epochs 4-6)**: training 3.22 → 3.34s/it (**+3.7%, slower**); samples 67.21 → 68.23s/image (**+1.5%**, within ~5% variance).

**Verdict**: No measurable improvement — the delta is within run-to-run variance (set-1 established ~21% training variation). The change is bitwise-identical and *removes* work (B kernel launches + a full `torch.cat`), so it cannot logically slow training; the measured regression is variance. **Reverted** per protocol — `base_model.py` restored to the original chunk loop; test suite re-verified (44 passed).

## Revert Criteria
- Any test failure or non-bitwise result mismatch in the unit check.
- No measurable improvement after full benchmark (keep only if ≥ ~0.5% or for
  code cleanliness — it removes a real per-step copy).

## Notes
- This is the single clearest "excess copies" item in the training loop: one full
  latent `torch.cat` + B-1 redundant kernel launches per step.
- Does not touch the scheduler, so no risk to timestep sampling or weighting.
