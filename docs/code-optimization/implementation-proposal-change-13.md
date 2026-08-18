# Implementation Proposal #13: Cache `temb` frequency vector + drop redundant `.to()` in `encode_images`

## Status
📝 PROPOSED — awaiting approval (two independent micro-opts, each ≤ 5 lines)

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
