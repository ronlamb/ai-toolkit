# Implementation Proposal #10: Pre-compute Text Fusion Context in Sampling Loop

## Status
⬜ PROPOSED - Awaiting implementation and user testing  
**Highest-impact change in set 2.**

## Complexity
Complex (11-20 lines total across `mmdit.py` + `pipeline.py`)

## Expected Impact
5-8% sample generation speedup (eliminates 27/28 redundant text-fusion computations per image)

## Focus Area
- **Calls that don't change and can be moved outside the loop**: The text context passes through `txtfusion` (4 transformer blocks: 2 layerwise + 2 refiner, each with attention + SwiGLU MLP) and `txtmlp` (RMSNorm + 2 linears) on **every** forward pass. In the 28-step sampling loop, the text context and its mask are identical on every step, so this sub-network is computed 28× with the same input. Pre-computing it once before the loop removes 27/28 of those computations.
- **Memory usage waste**: intermediate activations from the text-fusion blocks are allocated and freed every step.

## Issue Description

In `SingleStreamDiT.forward`, the first operations on the text context are:

```python
txtmask = _mask(mask[:, : context.shape[1]])
context = self.txtfusion(context, mask=txtmask)   # 4 transformer blocks (attn + MLP each)
context = self.txtmlp(context)                     # RMSNorm + 2 linears
```

`self.txtfusion` is a `TextFusionTransformer`: **4** `TextFusionBlock`s (2 layerwise + 2 refiner), each running an `Attention` (QKV projections + SDPA + output proj) and a `SwiGLU` MLP over the text sequence. This is a non-trivial sub-network — roughly 4/28 ≈ 14% of the transformer's block count, run on the text tokens.

In `Krea2Pipeline.__call__`, the 28-step Euler loop calls `predict_velocity` (→ `model.forward`) every step. The **text context is constant** across all 28 steps (same prompt, same mask). So `txtfusion` + `txtmlp` run 28× per image (56× with CFG) on identical input. Only the noisy `img` tokens and the timestep `t` change per step.

Pre-computing the fused text context **once** before the loop and passing it into `forward` (which then skips `txtfusion`/`txtmlp`) removes 27/28 of these computations per image.

## Current Code

### Location A: `extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `SingleStreamDiT.forward` (~line 490)

```python
    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        reflen: int = 0,
        isolate_refs: bool = False,
        ref_kv_capture: list | None = None,
        ref_kv_cache: tuple[list, Tensor] | None = None,
    ) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        txtmask = _mask(mask[:, : context.shape[1]])

        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)
        ...
```

### Location B: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, `predict_velocity` (~line 150)

```python
def predict_velocity(
    model: SingleStreamDiT,
    latents: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latents=None,
    isolate_refs: bool = False,
    ref_kv_cache: Optional[dict] = None,
) -> torch.Tensor:
    ...
    n = model.config.txtlayers
    context = context.reshape(
        context.shape[0], context.shape[1], n, context.shape[-1] // n
    )
    img_tokens, pos, mask = prepare(latents, context.shape[1], patch, text_mask)
    ...
    out = model(
        img=img_tokens, context=context, t=t, pos=pos, mask=mask,
        reflen=reflen, isolate_refs=isolate_refs, ref_kv_capture=capture,
        ref_kv_cache=(...) if reuse_ref_kv else None,
    )
```

### Location C: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, `Krea2Pipeline.__call__` (~line 340)

```python
        cond_feats, cond_mask = pad_text_features(conditional_embeds.text_embeds, device, dtype)
        if do_cfg:
            uncond_feats, uncond_mask = pad_text_features(unconditional_embeds.text_embeds, device, dtype)
        ...
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((latents.shape[0],), tcurr, dtype=dtype, device=device)
            v_cond = predict_velocity(transformer, latents.to(dtype), t, cond_feats, cond_mask, ...)
            if do_cfg:
                v_uncond = predict_velocity(transformer, latents.to(dtype), t, uncond_feats, uncond_mask, ...)
                v = v_cond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            latents = latents + (tprev - tcurr) * v.to(torch.float32)
```

## Optimized Code

### Step 1: Add a `fuse_context` method to `SingleStreamDiT` (new function)

```python
    def fuse_context(self, context: Tensor, text_mask: Tensor) -> Tensor:
        """Pre-compute the fused text context (txtfusion + txtmlp).

        ``context`` is the 4D stacked-layer features ``(B, Lt, n, d)``;
        ``text_mask`` is the ``(B, Lt)`` key-padding mask (1 for real text
        tokens). Returns the fused ``(B, Lt, features)`` context. Callers that
        run many steps with identical text (the 28-step sampling loop) compute
        this once and pass the result to ``forward`` via ``fused_context``,
        skipping the per-step txtfusion/txtmlp recompute. Numerically identical
        to the inline fusion in ``forward`` -- only *when* it runs changes.
        """
        txtmask = _mask(text_mask)
        return self.txtmlp(self.txtfusion(context, mask=txtmask))
```

### Step 2: Add an optional `fused_context` param to `forward` and skip fusion when provided

```python
    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        reflen: int = 0,
        isolate_refs: bool = False,
        ref_kv_capture: list | None = None,
        ref_kv_cache: tuple[list, Tensor] | None = None,
        fused_context: Tensor | None = None,   # <-- new
    ) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        if fused_context is not None:
            # Pre-fused by the caller (sampling loop): skip txtfusion/txtmlp.
            context = fused_context
        else:
            txtmask = _mask(mask[:, : context.shape[1]])
            context = self.txtfusion(context, mask=txtmask)
            context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)
        ...
```

### Step 3: Thread `fused_context` through `predict_velocity`

Add the param and pass it to the model call:
```python
def predict_velocity(
    model: SingleStreamDiT,
    latents: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latents=None,
    isolate_refs: bool = False,
    ref_kv_cache: Optional[dict] = None,
    fused_context: Tensor | None = None,   # <-- new
) -> torch.Tensor:
    ...
    out = model(
        img=img_tokens, context=context, t=t, pos=pos, mask=mask,
        reflen=reflen, isolate_refs=isolate_refs, ref_kv_capture=capture,
        ref_kv_cache=(...) if reuse_ref_kv else None,
        fused_context=fused_context,   # <-- new
    )
```

### Step 4: Pre-fuse once before the sampling loop in `Krea2Pipeline.__call__`

```python
        cond_feats, cond_mask = pad_text_features(conditional_embeds.text_embeds, device, dtype)
        if do_cfg:
            uncond_feats, uncond_mask = pad_text_features(unconditional_embeds.text_embeds, device, dtype)

        # Pre-fuse the text context once: it is identical across all denoising
        # steps, so computing txtfusion+txtmlp per step (28x) is pure waste.
        n = transformer.config.txtlayers
        def _fuse(feats, tmask):
            c4 = feats.reshape(feats.shape[0], feats.shape[1], n, feats.shape[-1] // n)
            return transformer.fuse_context(c4, tmask)
        fused_cond = _fuse(cond_feats, cond_mask)
        if do_cfg:
            fused_uncond = _fuse(uncond_feats, uncond_mask)

        ...
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((latents.shape[0],), tcurr, dtype=dtype, device=device)
            lat_d = latents.to(dtype)   # (see Change #9)
            v_cond = predict_velocity(
                transformer, lat_d, t, cond_feats, cond_mask,
                ref_latents=ref_latents, isolate_refs=isolate, ref_kv_cache=ref_cache,
                fused_context=fused_cond,   # <-- new
            )
            if do_cfg:
                v_uncond = predict_velocity(
                    transformer, lat_d, t, uncond_feats, uncond_mask,
                    ref_latents=ref_latents, isolate_refs=isolate, ref_kv_cache=ref_cache,
                    fused_context=fused_uncond,   # <-- new
                )
                v = v_cond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            latents = latents + (tprev - tcurr) * v.to(torch.float32)
```

## Changes Summary

- **`mmdit.py`**: new `fuse_context()` method (~8 lines, new function); `forward` gains one param and a 5-line if/else replacing the original 3 fusion lines (~4 net).
- **`pipeline.py` `predict_velocity`**: one new param + one new arg in the model call (2 lines).
- **`pipeline.py` `Krea2Pipeline.__call__`**: ~6 lines added before the loop (pre-fuse) + 2 new args in the loop calls.

**Per-function modified lines**: `forward` ~6, `predict_velocity` ~2, `__call__` ~8. All within the ≤20-line limit.

## Reasoning

1. **Moves a constant, expensive call out of the 28-step loop**: `txtfusion` (4 transformer blocks) + `txtmlp` depend only on the text context and mask, both constant per sampling run. Pre-fusing removes 27/28 recomputations (54/56 with CFG) per image.
2. **Numerically identical**: `fuse_context` runs the exact same ops (`_mask` → `txtfusion` → `txtmlp`) as the inline path. The text context and mask are bit-identical across steps, so the pre-fused result equals what each step would have computed.
3. **No API breakage**: `fused_context` is optional and defaults to `None`, so all existing callers (training via `get_noise_prediction`) are unaffected — they take the normal inline-fusion path.
4. **Training untouched**: Training calls `predict_velocity` without `fused_context`, so it still fuses inline (correct, since training text varies per step). Only the sampling loop opts in.

## Safety Notes

- **Gradient checkpointing**: `txtfusion` uses `checkpoint(...)` only when `torch.is_grad_enabled()`. The sampling loop runs under `@torch.no_grad()` (the pipeline's `__call__` is decorated), so no checkpointing occurs in either the inline or pre-fused path — same code path, safe.
- **Edit mode / ref latents**: Pre-fusing the text context is independent of reference-image handling (refs are appended to the *image* tokens, not the text). `fused_context` only replaces the text fusion; ref logic in `forward` is unchanged.
- **CFG**: Both cond and uncond contexts are pre-fused once each (they differ). Each step reuses its respective fused context.
- **Memory**: holds one extra `(B, Lt, features)` tensor per active prompt (cond + uncond). For a single preview image this is small relative to the model.

## Validation Protocol

Run benchmark test:
- 3 epochs × 30 steps, generate 4 images (CFG enabled — default for krea2 previews)
- Compare against set-1 best (training ~3.03s/it, samples ~65.12s/image)

## Expected Results

- **Sample generation**: 5-8% improvement (27/28 text-fusion recomputes removed per image; the fusion is ~14% of block compute, so removing 27/28 of it across the loop is a meaningful fraction of total sampling time)
- **Training time**: no change (training uses the inline path; `fused_context` is not passed)
- **VRAM**: +small (one fused context tensor per prompt held during the loop)

## Known Limitations

1. **Sampling-only**: No effect on training time. The win is entirely in the 28-step preview loop.
2. **Requires CFG or single-pass**: With `do_cfg` False, only one context is pre-fused (still a win — 27/28 removed). With CFG, both are pre-fused.
3. **Impact depends on text length**: The fusion cost scales with the text token count `Lt`. For short prompts the absolute saving is smaller; for long prompts (up to 512 tokens) it's larger. Expect the high end of 5-8% for typical prompt lengths.

## User Action Required

1. Implement the change (all 4 steps)
2. Run benchmark test with the protocol above (ensure CFG / guidance_scale > 0)
3. Report sample generation times and training time (training should be unchanged — a good correctness signal)
4. If improvement >5% keep; this is the highest-impact set-2 change
