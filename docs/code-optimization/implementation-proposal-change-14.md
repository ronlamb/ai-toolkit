# Change #14: Remove dead 256-alignment padding in `SingleStreamDiT.forward`

**Status**: PROPOSED (not yet implemented)
**Complexity**: Simple (5 lines removed)
**Expected Impact**: 2–8% training **and** sampling (largest at short captions / mixed resolutions)
**Applies to**: both training loop and image generation loop

## Issue

`extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `SingleStreamDiT.forward` (~line 519):

```python
        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        # Pad combined sequence to a multiple of 256 to stabilize compiled kernel shapes.
        fulllen = combined.shape[1]
        _padlen = (-fulllen) % 256
        if _padlen > 0:
            combined = F.pad(combined, (0, 0, 0, _padlen))
            mask = F.pad(mask, (0, _padlen), value=False)
            pos = F.pad(pos, (0, 0, 0, _padlen))
```

This padding was added for `torch.compile` ("stabilize compiled kernel shapes"), but compile was **reverted in set-1 (Change #2)** — no code path in Krea 2 uses `torch.compile` anymore. What remains is pure overhead on *every* forward pass:

1. **A full-sequence copy**: `F.pad(combined, ...)` allocates a new `(B, L_pad, 6144)` bf16 tensor and copies the whole sequence (~56 MB per sample at 1024×1024) just to append masked-out pad tokens. Same for `pos`.
2. **Wasted compute on pad tokens**: every block runs attention + MLP over the padded length. Because dense (masked) attention cost scales ~L², the waste exceeds the token delta:
   - 1024×1024, txtlen=512: 4608 → 4864 = **+5.5% tokens, ~+11% attention**
   - 512×512, short caption (txtlen≈47): 1071 → 1280 = **+19.5% tokens, ~+43% attention** ← the common case for short prompts!
   - Only exact multiples of 256 pay nothing.

The user's dataset mixes 512×512 and 1024×1024 with natural caption lengths, so most steps pay this tax.

## Proposed change

Delete the padding block entirely:

```python
        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)
```

Downstream code never depends on the padded length:
- `final[:, txtlen : txtlen + imglen - reflen, :]` slices by real lengths.
- The `isolate_refs` asymmetric mask uses `combined.shape[1]` — consistent either way.
- The ref-K/V `extra` mask concat uses `padmask` (B, L) — consistent either way.

**Numerics**: padded query rows produced outputs that were sliced off; pad *keys* were excluded from softmax by the bool key-padding mask (`_mask`). Removing them is mathematically identical for real tokens (not bitwise-identical — different kernel shapes can reorder reductions — so visually verify one sample image).

## Fallback variant B (only if A regresses)

If cuDNN SDPA turns out to prefer aligned sequence lengths, keep alignment but shrink the target:

```python
        _padlen = (-combined.shape[1]) % 16   # was % 256
```

Test A first; only try B if A is slower. (Same file, same lines.)

## Validation plan

- `pytest tests/` (44 passed baseline).
- Visual check: one preview image looks normal (no structure change vs pre-change at same seed/steps).
- Benchmark: 6+ epochs × 30 steps, 4 images; compare **bottom-out s/it** and sample times vs current best (#10 state: 3.02 s/it bottom-out user-measured; samples ~67.2s in 1024-mix epochs) with the same dataset mix.
- Keep if bottom-out improves beyond variance; negligible → user decides; slower → revert (`git checkout -- extensions_built_in/diffusion_models/krea2/src/mmdit.py`).
