# Implementation Proposal #11: Vectorize `BaseModel.add_noise` (kill per-sample chunk loop)

## Status
📝 PROPOSED — awaiting approval

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
       for idx in range(original_samples.shape[0]):
           # scheduler's add_noise called once PER SAMPLE:
           noisy_latents = self.noise_scheduler.add_noise(
               original_samples_chunks[idx], noise_chunks[idx], timesteps_chunks[idx])
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
    # Fast path: flow-matching style schedulers broadcast per-sample timesteps
    # (B,) -> (B,1,1,1) and handle the whole batch in one op. The chunk loop
    # below only exists for schedulers that need per-sample calls.
    if isinstance(timesteps, torch.Tensor) and timesteps.shape[0] == original_samples.shape[0]:
        return self.noise_scheduler.add_noise(original_samples, noise, timesteps)

    original_samples_chunks = torch.chunk(...)
    # ...existing chunk loop unchanged...
```

**Why this is safe for Krea2 (and other flow-matching archs):**
- `timesteps` arrives as `(B,)` float (from `noise_scheduler.timesteps[timestep_indices]`).
- In the scheduler's `add_noise`, `(timesteps / 1000)` is `(B,)`; multiplying with
  `(B, C, h, w)` broadcasts as `(1, B, 1, 1) * (B, C, h, w)` → **correct** per-sample
  scaling (verified: PyTorch aligns trailing dims; the B dim lands on C).
- The chunked path produces exactly `(1 - t_i) * x_i + t_i * n_i` per sample —
  bit-identical math to the batched op (same elementwise ops, same order).
- Non-flow-matching schedulers (DDPM etc.) with `timesteps.shape[0] == B` would also
  take the fast path; DDPM's `add_noise` is itself batch-broadcastable in diffusers
  (`sqrt_alpha * sample + sqrt_one_minus_alpha * noise` with sigma broadcast), so the
  result is identical there too. If any exotic scheduler breaks, the fallback loop
  remains for `timesteps.shape[0] != B` (e.g. single shared timestep).

**Precision:** identical — same elementwise arithmetic, no dtype change
(`t_01` is float32 in both paths; the blend upcasts to the sample dtype as before).

## Location
`toolkit/models/base_model.py`, `BaseModel.add_noise` (~line 750)

## Lines Changed
~4 (one `if` + early return inserted at top of method)

## Validation Plan
1. Unit check: for a random (B=4, 16, 32, 32) latent + noise + (B,) timesteps,
   assert `torch.allclose` between old chunked result and new batched result
   (expect exact equality).
2. Speed test per protocol: 8 epochs × 30 steps, generate 2 images.
   Expect small but consistent s/it reduction; sample time unchanged (this path
   is training-only).

## Revert Criteria
- Any test failure or non-bitwise result mismatch in the unit check.
- No measurable improvement after full benchmark (keep only if ≥ ~0.5% or for
  code cleanliness — it removes a real per-step copy).

## Notes
- This is the single clearest "excess copies" item in the training loop: one full
  latent `torch.cat` + B-1 redundant kernel launches per step.
- Does not touch the scheduler, so no risk to timestep sampling or weighting.
