# Implementation Proposal #8: Cache RoPE Frequencies for Fixed Positions

## Status
⬜ PROPOSED - Awaiting implementation and user testing  
**Depends on**: Change #7 (stable `pos` object identity). **Do not apply standalone** — see Safety Notes.

## Complexity
Moderate (6-10 lines changed in `SingleStreamDiT.forward` + a small cache field)

## Expected Impact
1-3% sample generation speedup (skips RoPE freq recomputation across the 28-step loop); ~0-1% training (within a fixed-resolution bucket)

## Focus Area
- **Calls that don't change and can be moved outside the loop**: `freqs = self.posemb(pos)` is a pure function of `pos`. In the 28-step sampling loop, `pos` (and therefore `freqs`) is identical on every step, yet the RoPE frequencies are recomputed 28×.
- **Memory usage waste**: a ~5MB float32 freqs tensor is allocated and freed every forward pass.

## Issue Description

In `SingleStreamDiT.forward`, the RoPE frequencies are computed on every call:

```python
freqs = self.posemb(pos)   # 3x rope(): arange + einsum + cos/sin over (B, L, d)
```

`posemb(pos)` runs `rope()` once per position axis (3 axes), each doing an
`einsum` plus `cos`/`sin`/`stack` over the full sequence `(B, L, d)`. For a
1024×1024 image the sequence is ~4600 tokens, so this builds a
`(B, L, headdim/2, 2, 2)` float32 tensor (~5MB) on **every** forward pass.

In the sampling loop (`Krea2Pipeline.__call__`), `pos` is constant across all
28 denoising steps (same resolution, same text length), so `freqs` is identical
28 times in a row. Recomputing it 27 extra times per image is pure waste.

## Current Code

### Location: `extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `SingleStreamDiT.forward` (~line 570)

```python
        blockcaches = [None] * len(self.blocks)
        if ref_kv_cache is not None:
            blockcaches, refmask = ref_kv_cache
            extra = padmask.unsqueeze(1).unsqueeze(3) & refmask.unsqueeze(1).unsqueeze(2)
            mask = torch.cat((mask, extra), dim=3)

        freqs = self.posemb(pos)

        for block, blockkv in zip(self.blocks, blockcaches):
            ...
```

## Optimized Code

### Step 1: Add a freqs cache field in `__init__`

```python
    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        # Cache of RoPE freqs keyed by (pos.data_ptr(), pos.shape). Safe only
        # when the caller reuses a stable `pos` object across calls (see
        # pipeline.prepare's geometry cache, Change #7). A miss simply
        # recomputes; a hit skips the posemb() trig/einsum work.
        self._freqs_cache: tuple = (None, None)   # (key, freqs)
```

### Step 2: Replace the `freqs = self.posemb(pos)` line with a cached lookup

```python
        # RoPE freqs are a pure function of `pos`. When the caller reuses a
        # stable pos object (fixed resolution, e.g. the 28-step sampling loop),
        # its data_ptr + shape are unchanged, so we reuse the cached freqs and
        # skip the posemb() einsum/trig. Keyed on data_ptr+shape; a mismatch
        # (new pos object) falls through to a recompute, so this degrades
        # gracefully and is never incorrect for stable-pos callers.
        fkey = (pos.data_ptr(), tuple(pos.shape))
        if self._freqs_cache[0] != fkey:
            self._freqs_cache = (fkey, self.posemb(pos))
        freqs = self._freqs_cache[1]

        for block, blockkv in zip(self.blocks, blockcaches):
            ...
```

## Changes Summary

- **`__init__`**: one added line initializing `self._freqs_cache`.
- **`forward`**: the single line `freqs = self.posemb(pos)` → a 5-line cached lookup.

**Net modified lines in `forward`**: ~6 (1 removed, 5 added). Within the ≤20-line limit.

## Reasoning

1. **Moves a constant call out of the 28-step loop**: `freqs` depends only on `pos`, which is constant per sampling run. Caching removes 27/28 recomputations of the RoPE trig/einsum per image.
2. **Numerically identical**: On a cache hit we return the exact tensor that `posemb(pos)` would have produced (same input, deterministic op). On a miss we compute it normally.
3. **No API breakage**: `forward` signature and return value are unchanged; the cache is internal.

## Safety Notes (IMPORTANT)

The cache key is `(pos.data_ptr(), pos.shape)`. This is **only safe when the
caller reuses a stable `pos` object across calls** — i.e. **Change #7 must be
applied first**, because Change #7's `_prepare_geom` cache holds the `pos`
tensor alive in a module-level dict, so its `data_ptr()` is stable for a given
geometry and cannot be recycled by the allocator while cached.

- **With Change #7**: `pos` objects are long-lived and unique per geometry →
  `data_ptr()` is a reliable identity → cache hits are correct.
- **Without Change #7**: `pos` is a fresh tensor each call and may be freed
  then reallocated at the same address with *different* content (same shape) →
  a stale cache hit would return wrong freqs. **Therefore do not apply this
  change unless Change #7 is in place.**

Additional safety properties:
- **Device moves**: if the model/pos move device, `data_ptr()` changes → cache
  miss → recompute. Correct.
- **Ref latents (edit mode)**: adding refs changes `pos` content and shape →
  different key → recompute. Correct.
- **Memory**: one ~5MB tensor held per active geometry; negligible and bounded
  by the number of distinct resolutions in flight (typically 1).

## Validation Protocol

Run benchmark test:
- 3 epochs × 30 steps, generate 4 images
- Compare against set-1 best (training ~3.03s/it, samples ~65.12s/image)
- **Prerequisite**: Change #7 must already be applied and validated.

## Expected Results

- **Sample generation**: 1-3% improvement (27/28 RoPE recomputes removed per image)
- **Training time**: ~0-1% (within a fixed-resolution bucket the pos is stable; across buckets it recomputes)
- **VRAM**: +~5MB (one cached freqs tensor per active geometry)

## Known Limitations

1. **Hard dependency on Change #7** for correctness (see Safety Notes).
2. **Modest impact**: RoPE freq computation is a small fraction of the forward pass (the 28 attention blocks dominate), so even eliminating it entirely yields only a few percent on sampling.
3. **Single-entry cache**: Only the most recent geometry's freqs are kept. If training alternates between two resolutions every step, this thrashes (recompute each time) and provides no benefit — but it also never hurts. A small dict keyed by geometry (like Change #7) could be used instead if multi-resolution thrashing is observed.

## User Action Required

1. **Confirm Change #7 is applied first.**
2. Implement the change
3. Run benchmark test with the protocol above
4. If improvement >5% keep; otherwise evaluate as cumulative benefit (likely small)
