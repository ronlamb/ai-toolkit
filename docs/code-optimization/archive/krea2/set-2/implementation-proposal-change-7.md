# Implementation Proposal #7: Cache Position Grid and Mask in `prepare()`

## Status
⬜ PROPOSED - Awaiting implementation and user testing

## Complexity
Moderate (6-10 lines changed in `prepare()` + a small new helper)

## Expected Impact
2-4% (both training and sampling)

## Focus Area
- **Inefficient repeated work / calls that don't change**: The image position grid, text positions (all zeros), and the image portion of the attention mask are **constant for a fixed resolution**, yet `prepare()` rebuilds them on every call. In the 28-step sampling loop this is 27 redundant rebuilds of identical tensors; in training it's one redundant rebuild per step.
- **Memory usage waste**: ~5 small GPU tensors (zeros, aranges, a `repeat`, two `cat`s) are allocated and freed every call.

## Issue Description

`prepare()` in `pipeline.py` builds the RoPE position grid and key-padding mask from scratch on every call:

```python
def prepare(img, txtlen, patch, txtmask):
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)          # alloc
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None] # alloc + write
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :] # alloc + write
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)  # alloc
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool) # alloc
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)         # alloc
    mask = torch.cat((txtmask.to(img.device).bool(), imgmask), dim=1)  # alloc
    pos = torch.cat((txtpos, imgpos), dim=1)                      # alloc
    return img, pos, mask
```

For a **fixed resolution** (`h`, `w`) and fixed text length (`txtlen`), the following are identical on every call:
- `imgpos` — the image RoPE grid (depends only on `h_`, `w_`, `b`)
- `txtpos` — all zeros (depends only on `txtlen`, `b`)
- therefore the full `pos = cat(txtpos, imgpos)` (depends only on `b`, `txtlen`, `h_`, `w_`)
- `imgmask` — all True (depends only on `b`, `h_*w_`)

Only two things actually change per call:
- the rearranged image tokens (`img` — the real latents)
- the **text** portion of `mask` (from `txtmask`, which varies with the prompt)

So we can cache `(pos, imgmask)` keyed by `(b, txtlen, h_, w_)` and rebuild only the text part of `mask` each call.

## Current Code

### Location: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 74-95

```python
def prepare(
    img: torch.Tensor, txtlen: int, patch: int, txtmask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patchify the latent and build the combined text+image position / mask.

    in:  img      (B, C, h, w) image latent
         txtlen   number of text tokens
         patch    transformer patch size
         txtmask  (B, txtlen) long/bool mask, 1 for real text tokens
    out: (img_tokens (B, h/p*w/p, C*p*p), pos (B, txtlen+imglen, 3),
          mask (B, txtlen+imglen))
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask.to(img.device).bool(), imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask
```

## Optimized Code

### Step 1: Add a small module-level cache helper (new function)

```python
# Cache of resolution-invariant position/mask tensors built by prepare().
# Keyed by (b, txtlen, h_, w_); values are (pos, imgmask) on the build device.
# Entries are tiny (a few KB each) and only a handful of resolutions are ever
# active, so the dict stays small. pos is read-only downstream (F.pad / rope),
# so sharing the cached tensor across calls is safe.
_PREPARE_CACHE: dict = {}


def _prepare_geom(b, txtlen, h_, w_, device):
    key = (b, txtlen, h_, w_)
    entry = _PREPARE_CACHE.get(key)
    if entry is None or entry[0].device != device:
        imgids = torch.zeros((h_, w_, 3), device=device)
        imgids[..., 1] = torch.arange(h_, device=device)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device)[None, :]
        imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
        txtpos = torch.zeros(b, txtlen, 3, device=device)
        pos = torch.cat((txtpos, imgpos), dim=1)
        imgmask = torch.ones(b, h_ * w_, device=device, dtype=torch.bool)
        entry = (pos, imgmask)
        _PREPARE_CACHE[key] = entry
    return entry
```

### Step 2: Rewrite `prepare()` to use the cache

```python
def prepare(
    img: torch.Tensor, txtlen: int, patch: int, txtmask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patchify the latent and build the combined text+image position / mask.

    in:  img      (B, C, h, w) image latent
         txtlen   number of text tokens
         patch    transformer patch size
         txtmask  (B, txtlen) long/bool mask, 1 for real text tokens
    out: (img_tokens (B, h/p*w/p, C*p*p), pos (B, txtlen+imglen, 3),
          mask (B, txtlen+imglen))

    The resolution-invariant position grid and image mask are cached per
    (b, txtlen, h_, w_) so the 28-step sampling loop (and every training step)
    does not rebuild them; only the text portion of the mask is rebuilt here.
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    pos, imgmask = _prepare_geom(b, txtlen, h_, w_, img.device)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    mask = torch.cat((txtmask.to(img.device).bool(), imgmask), dim=1)
    return img, pos, mask
```

## Changes Summary

- **New helper** `_prepare_geom()` + module-level `_PREPARE_CACHE` dict: builds `(pos, imgmask)` once per unique `(b, txtlen, h_, w_)` and reuses it.
- **`prepare()`**: removed the 6 lines that build `imgids`/`arange`s/`repeat`/`txtpos`/`pos`; replaced with a single cache lookup. The `img` rearrange and the text-part of `mask` are still computed per call (they genuinely change).

**Net modified lines in `prepare()`**: ~6 removed, ~2 added. Well within the ≤20-line limit. The helper is a new function (does not count against an existing function's budget).

## Reasoning

1. **Eliminates repeated work**: The position grid and image mask are pure functions of the resolution, which is constant within a sampling run (28 steps) and usually constant across consecutive training steps. Caching removes 5 tensor allocations + a `repeat` per call.
2. **Sampling loop benefit is largest**: In `Krea2Pipeline.__call__`, `prepare()` runs 28× per image with identical geometry. Caching removes 27/28 of the rebuilds.
3. **Numerically identical**: The cached tensors are built with exactly the same operations as before; only *when* they're built changes (once vs every call). `pos` is returned by reference but is only ever read downstream (`F.pad`, `rope`), so sharing is safe.
4. **No API breakage**: `prepare()` signature and return values are unchanged.

## Validation Protocol

Run benchmark test:
- 3 epochs × 30 steps, generate 4 images
- Compare against set-1 best (training ~3.03s/it, samples ~65.12s/image)

## Expected Results

- **Training time**: small improvement (one fewer geometry rebuild per step)
- **Sample generation**: larger relative improvement (27/28 redundant geometry rebuilds removed per image)
- **VRAM**: negligible (a few KB per cached resolution; dict stays tiny in practice)

## Known Limitations

1. **Cache growth**: The dict is keyed by `(b, txtlen, h_, w_)`. In practice only a handful of resolutions are active (one per sampling run; bucketed but few distinct sizes in training). Each entry is a few KB, so total memory is negligible. If unbounded growth were ever a concern (e.g. wildly varying txtlen), the key could be reduced to `(b, h_, w_)` by rebuilding `txtpos` per call — but that's not expected to be needed.
2. **Device moves**: The cache stores tensors on the build device and checks `entry[0].device != device` to rebuild if the model moved devices. This handles low-VRAM offload scenarios.
3. **`txtlen` in the key**: Including `txtlen` means a new cache entry per distinct text length. Since `pos`'s text portion is all zeros, this is slightly conservative (a `(b, h_, w_)` key would suffice for `imgpos`, but the full `pos` includes the zero text block whose length depends on `txtlen`). Keeping `txtlen` in the key is simplest and correct.

## User Action Required

1. Implement the change
2. Run benchmark test with the protocol above
3. Report training time, sample generation times
4. If improvement >5% keep; otherwise evaluate as cumulative benefit
