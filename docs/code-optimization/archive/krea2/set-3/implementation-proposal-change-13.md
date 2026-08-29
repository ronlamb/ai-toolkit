# Implementation Proposal #13: Cache `temb` frequency vector + drop redundant `.to()` in `encode_images`

## Status
⚠️ REVERTED — Slower bottom-out (3.16 vs 3.02 s/it), flat total times, no end-of-run improvement

**Benchmark summary**: 6 epochs × 30 steps, 4 images. Training bottom-out 3.16 s/it
vs current best's 3.02 s/it (+4.6%); samples ~69.5s/image (1024-mix epochs) vs
~67.2s baseline (+3.4%). Dataset mixes 512×512 + 1024×1024 sets, so absolute
sample times shift with the mix.

> **Baseline calibration note (added in set 4)**: the 3.02 s/it figure is from a longer
> (~179+ step) run and is not comparable to the 6×30 short benchmark, which bottoms out at
> ~3.22–3.26 s/it in the change #10 state. Against that corrected baseline this change's
> training was neutral-to-faster (3.16 vs 3.22); samples were slower (+3.4%), so it was still
> reverted — but the "+4.6% slower" framing overstated the regression.

| Epoch | Steps | Total time | Avg training (s/it) | S1 | S2 | S3 | S4 | Avg sample (s) |
|-------|-------|------------|---------------------|--------|--------|--------|--------|----------------|
| 1 | 30 | 1:59 | 4.12 | 70.62 | 70.00 | 69.79 | 69.77 | 70.05 |
| 2 | 30 | 1:42 | 3.76 | 69.58 | 69.58 | 69.56 | 69.55 | 69.57 |
| 3 | 30 | 1:35 | 3.56 | 69.57 | 69.51 | 69.50 | 69.49 | 69.52 |
| 4 | 30 | 1:39 | 3.49 | 69.65 | 69.57 | 69.51 | 69.50 | 69.56 |
| 5 | 30 | 1:44 | 3.49 | 69.61 | 69.60 | 69.57 | 69.07 | 69.46 |
| 6 | 30 | 1:21 | 3.36 | 64.31 | 64.22 | 64.19 | 64.17 | 64.22 |

*Avg training (s/it) is the progress bar's cumulative rate at epoch end. Total time is the
per-epoch elapsed delta (excludes sample generation). Epoch 6's ~64s samples reflect a
lighter-resolution batch in the mixed dataset, not an improvement.*

**Verdict**: slower samples (+3.4%) with no training win — **reverted** per protocol.
`mmdit.py` restored to the original per-call `temb` freqs; `krea2.py` restored to the original
`return latents.to(device, dtype=dtype)`. Test suite re-verified (44 passed). Part B (the no-op
`.to()` removal in `encode_images`) was never benchmarked on its own.

**Implementation notes**:
- Part A: cache key is `(dim, period, device)` (added `period` vs. the original
  proposal's `(dim, device)`) so a non-default `period` can't silently return wrong
  freqs. Unit check: cached `temb` is bitwise-identical (`torch.equal`) to the
  original across dim ∈ {256, 1024}, B ∈ {1, 3, 8}, dtype ∈ {fp32, bf16};
  `period=1e3` does not collide with the cached default.
- Part B: no-op removal (identical values). `pytest tests/`: 44 passed.

## Complexity
Simple (1–5 lines each)

## Expected Impact
~0.1% training speedup combined (tiny, but both are pure waste: a per-call
`torch.exp(arange)` rebuild and a no-op device/dtype copy of the full latent batch)

## Part A: Cache `temb` frequencies (mmdit.py)

### Issue
`SingleStreamDiT.forward` calls `temb(t, self.config.tdim, ...)` once per forward.
Inside:

```python
def temb(t, dim, period=1e4, tfactor=1e3, device=None, dtype=None):
    half = dim // 2
    freqs = torch.exp(
        -math.log(period) * torch.arange(half, dtype=torch.float32, device=device) / half
    )
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)
```

`freqs` is a constant (dim=256 → 128 values) rebuilt with `arange` + `exp` on every
forward. It is a free function (no module state), so cache it in a module-level dict
keyed by `(dim, device)` — meta-safe (built on first real call), survives device
moves (per-device key).

```python
_TEMB_FREQS = {}

def _temb_freqs(dim: int, device) -> Tensor:
    key = (dim, device)
    f = _TEMB_FREQS.get(key)
    if f is None:
        half = dim // 2
        f = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=device) / half)
        _TEMB_FREQS[key] = f
    return f

def temb(t, dim, period=1e4, tfactor=1e3, device=None, dtype=None):
    freqs = _temb_freqs(dim, device)
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)
```

(If `period` is ever made configurable, add it to the cache key; today it's a
hardcoded default and `SingleStreamDiT` never overrides it.)

### Location / lines
`mmdit.py`, `temb` (~line 74). ~6 lines.

## Part B: Drop redundant `.to()` in `Krea2Model.encode_images` (krea2.py)

### Issue
```python
latents = torch.stack(latents)  # (B, 16, h, w), already on `device` in `dtype`
mean = self._vae_latents_mean.to(dtype=dtype)
std_inv = self._vae_latents_std_inv.to(dtype=dtype)
latents = (latents - mean) * std_inv
return latents.to(device, dtype=dtype)   # <-- no-op copy of the full batch
```

Each `img` was already moved with `.to(device, dtype=dtype)` before VAE encode, so
the stacked latents are on `device` in `dtype`. The final `.to(device, dtype=dtype)`
is a no-op that still forces a full (B, 16, h, w) copy on some paths. Remove it
(return `latents` directly).

### Location / lines
`krea2.py`, `encode_images` (~line 810). 1 line.

## Precision
- Part A: identical math (same fp32 `exp` values, just not recomputed).
- Part B: identical (no-op removal).

## Validation Plan
1. Unit check for A: `temb` output before/after on a random (B,) t — expect exact
   equality.
2. Speed test per protocol: 8 epochs × 30 steps, generate 2 images.
   Expect negligible-to-small s/it reduction; keep for cleanliness if neutral.

## Revert Criteria
- Any mismatch in the unit check, or test failures.
