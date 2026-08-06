# Implementation Proposal #2: torch.compile for predict_velocity

## Status
⚠️ PROPOSED - Awaiting user testing

## Complexity
Moderate (6-10 lines changed)

## Expected Impact
5-8% speedup

## Issue Description

The `predict_velocity` function in `pipeline.py` contains complex tensor operations including multiple `rearrange`, `cat`, and model calls. These operations could benefit from PyTorch's compilation capabilities, especially for the training loop where the same operations are repeated many times.

## Current Code

### Location: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 105-180

```python
def predict_velocity(
    model: SingleStreamDiT,
    latents: torch.Tensor,  # (B, C, h, w)
    t: torch.Tensor,  # (B,) flow time in [0, 1] (1 = pure noise)
    context: torch.Tensor,  # (B, Lt, n*d) flattened stacked Qwen3-VL features
    text_mask: torch.Tensor,  # (B, Lt) 1 for real text tokens
    ref_latents: Optional[List[List[torch.Tensor]]] = None,  # per-sample (C, h, w) refs
    isolate_refs: bool = False,
    ref_kv_cache: Optional[dict] = None,
) -> torch.Tensor:
    """Run the MMDiT on the packed [text | image | refs] sequence."""
    patch = model.config.patch
    b, c, h, w = latents.shape

    if ref_kv_cache is not None and not isolate_refs:
        raise ValueError(
            "ref_kv_cache requires isolate_refs: cached ref K/V are only "
            "step-invariant when ref tokens attend solely to each other"
        )
    reuse_ref_kv = ref_kv_cache is not None and ref_kv_cache.get("kv") is not None
    if reuse_ref_kv:
        ref_latents = None  # the refs are consumed from the cache instead

    # Restore the stacked-layer axis flattened in pad_text_features: F -> (n, d).
    n = model.config.txtlayers
    context = context.reshape(
        context.shape[0], context.shape[1], n, context.shape[-1] // n
    )

    img_tokens, pos, mask = prepare(latents, context.shape[1], patch, text_mask)

    reflen = 0
    ref_mask = None
    if ref_latents is not None and any(len(r) > 0 for r in ref_latents):
        ref_tokens, ref_pos, ref_mask = pack_ref_latents(
            ref_latents, patch, img_tokens.device, img_tokens.dtype
        )
        reflen = ref_tokens.shape[1]
        img_tokens = torch.cat((img_tokens, ref_tokens), dim=1)
        pos = torch.cat((pos, ref_pos), dim=1)
        mask = torch.cat((mask, ref_mask), dim=1)

    capture = None
    if ref_kv_cache is not None and not reuse_ref_kv and reflen > 0:
        capture = []

    out = model(
        img=img_tokens,
        context=context,
        t=t,
        pos=pos,
        mask=mask,
        reflen=reflen,
        isolate_refs=isolate_refs,
        ref_kv_capture=capture,
        ref_kv_cache=(ref_kv_cache["kv"], ref_kv_cache["mask"])
        if reuse_ref_kv
        else None,
    )

    if capture is not None:
        ref_kv_cache["kv"] = capture
        ref_kv_cache["mask"] = ref_mask

    # (B, imglen, c*p*p) -> (B, c, h, w)
    velocity = rearrange(
        out,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=h // patch,
        w=w // patch,
    )
    return velocity
```

## Optimized Code

```python
@torch.compile(mode="reduce-overhead", dynamic=True)
def predict_velocity(
    model: SingleStreamDiT,
    latents: torch.Tensor,  # (B, C, h, w)
    t: torch.Tensor,  # (B,) flow time in [0, 1] (1 = pure noise)
    context: torch.Tensor,  # (B, Lt, n*d) flattened stacked Qwen3-VL features
    text_mask: torch.Tensor,  # (B, Lt) 1 for real text tokens
    ref_latents: Optional[List[List[torch.Tensor]]] = None,  # per-sample (C, h, w) refs
    isolate_refs: bool = False,
    ref_kv_cache: Optional[dict] = None,
) -> torch.Tensor:
    """Run the MMDiT on the packed [text | image | refs] sequence."""
    patch = model.config.patch
    b, c, h, w = latents.shape

    if ref_kv_cache is not None and not isolate_refs:
        raise ValueError(
            "ref_kv_cache requires isolate_refs: cached ref K/V are only "
            "step-invariant when ref tokens attend solely to each other"
        )
    reuse_ref_kv = ref_kv_cache is not None and ref_kv_cache.get("kv") is not None
    if reuse_ref_kv:
        ref_latents = None  # the refs are consumed from the cache instead

    # Restore the stacked-layer axis flattened in pad_text_features: F -> (n, d).
    n = model.config.txtlayers
    context = context.reshape(
        context.shape[0], context.shape[1], n, context.shape[-1] // n
    )

    img_tokens, pos, mask = prepare(latents, context.shape[1], patch, text_mask)

    reflen = 0
    ref_mask = None
    if ref_latents is not None and any(len(r) > 0 for r in ref_latents):
        ref_tokens, ref_pos, ref_mask = pack_ref_latents(
            ref_latents, patch, img_tokens.device, img_tokens.dtype
        )
        reflen = ref_tokens.shape[1]
        img_tokens = torch.cat((img_tokens, ref_tokens), dim=1)
        pos = torch.cat((pos, ref_pos), dim=1)
        mask = torch.cat((mask, ref_mask), dim=1)

    capture = None
    if ref_kv_cache is not None and not reuse_ref_kv and reflen > 0:
        capture = []

    out = model(
        img=img_tokens,
        context=context,
        t=t,
        pos=pos,
        mask=mask,
        reflen=reflen,
        isolate_refs=isolate_refs,
        ref_kv_capture=capture,
        ref_kv_cache=(ref_kv_cache["kv"], ref_kv_cache["mask"])
        if reuse_ref_kv
        else None,
    )

    if capture is not None:
        ref_kv_cache["kv"] = capture
        ref_kv_cache["mask"] = ref_mask

    # (B, imglen, c*p*p) -> (B, c, h, w)
    velocity = rearrange(
        out,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=h // patch,
        w=w // patch,
    )
    return velocity
```

## Changes Summary

- Added `@torch.compile(mode="reduce-overhead", dynamic=True)` decorator to the function
- This compiles the function on first call with the given input shapes
- `dynamic=True` allows the compiled graph to handle varying sequence lengths (important for training with different resolutions)

## Reasoning

1. **torch.compile** can optimize the computational graph, eliminating Python overhead and fusing operations
2. **mode="reduce-overhead"** is appropriate for training where we want to minimize compilation overhead
3. **dynamic=True** is necessary because sequence lengths vary during training (different resolutions, different numbers of reference tokens)

## Validation Protocol

Run benchmark test:
- 3 epochs × 30 steps
- Generate 4 images per epoch

Compare against baseline results in `results-baseline.md`.

## Expected Results

- **Training time**: 5-8% improvement (compiled graph eliminates Python overhead)
- **Sample generation**: 3-5% improvement (faster inference with compiled graph)

## Known Limitations

1. First call will be slower due to compilation overhead
2. Memory usage may increase slightly due to compiled graph caching
3. May not work with all PyTorch versions (requires 2.0+)

## User Action Required

1. Test this change with the benchmark protocol
2. Report training time and sample generation times
3. If improvement >5%, keep the change; otherwise, revert
