# Implementation Proposal #9: Single Dtype Conversion in CFG Sampling Loop

## Status
⬜ PROPOSED - Awaiting implementation and user testing

## Complexity
Simple (1-5 lines changed)

## Expected Impact
1-2% sample generation speedup

## Focus Area
- **Excessive memory copies**: `latents.to(dtype)` is called twice per denoising step when CFG is enabled (once for the conditional pass, once for the unconditional pass). The second call converts an already-converted tensor — a redundant GPU copy.
- **Calls that don't change and can be moved outside the loop**: The dtype-cast of `latents` is identical for both passes in a step, so it should be computed once.

## Issue Description

In `Krea2Pipeline.__call__`, the Euler integration loop runs two model passes per step when classifier-free guidance is on:

```python
for tcurr, tprev in zip(ts[:-1], ts[1:]):
    t = torch.full((latents.shape[0],), tcurr, dtype=dtype, device=device)
    v_cond = predict_velocity(
        transformer,
        latents.to(dtype),          # <-- conversion 1 (float32 -> model dtype)
        t, cond_feats, cond_mask, ...
    )
    if do_cfg:
        v_uncond = predict_velocity(
            transformer,
            latents.to(dtype),      # <-- conversion 2 (redundant: same input)
            t, uncond_feats, uncond_mask, ...
        )
        v = v_cond + guidance_scale * (v_cond - v_uncond)
    else:
        v = v_cond
    latents = latents + (tprev - tcurr) * v.to(torch.float32)
```

`latents` is held in float32 (the ODE accumulator). Each `predict_velocity` call casts it to the model dtype (`bf16`). With CFG, that cast happens **twice per step** on the same tensor. Over 28 steps × 4 images, that's a redundant copy of the full latent tensor (16×h×w) 28 extra times per image.

## Current Code

### Location: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines ~360-380

```python
        # Euler integration of the flow ODE (with optional CFG).
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((latents.shape[0],), tcurr, dtype=dtype, device=device)
            v_cond = predict_velocity(
                transformer,
                latents.to(dtype),
                t,
                cond_feats,
                cond_mask,
                ref_latents=ref_latents,
                isolate_refs=isolate,
                ref_kv_cache=ref_cache,
            )
            if do_cfg:
                v_uncond = predict_velocity(
                    transformer,
                    latents.to(dtype),
                    t,
                    uncond_feats,
                    uncond_mask,
                    ref_latents=ref_latents,
                    isolate_refs=isolate,
                    ref_kv_cache=ref_cache,
                )
                v = v_cond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            latents = latents + (tprev - tcurr) * v.to(torch.float32)
```

## Optimized Code

Hoist the dtype cast to a single variable at the top of each step:

```python
        # Euler integration of the flow ODE (with optional CFG).
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((latents.shape[0],), tcurr, dtype=dtype, device=device)
            lat_d = latents.to(dtype)   # cast once; shared by cond + uncond passes
            v_cond = predict_velocity(
                transformer,
                lat_d,
                t,
                cond_feats,
                cond_mask,
                ref_latents=ref_latents,
                isolate_refs=isolate,
                ref_kv_cache=ref_cache,
            )
            if do_cfg:
                v_uncond = predict_velocity(
                    transformer,
                    lat_d,
                    t,
                    uncond_feats,
                    uncond_mask,
                    ref_latents=ref_latents,
                    isolate_refs=isolate,
                    ref_kv_cache=ref_cache,
                )
                v = v_cond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            latents = latents + (tprev - tcurr) * v.to(torch.float32)
```

## Changes Summary

- **`Krea2Pipeline.__call__`**: added one line `lat_d = latents.to(dtype)` at the top of the loop body; replaced both `latents.to(dtype)` call arguments with `lat_d`.

**Net modified lines**: 3 (1 added, 2 changed). Well within the ≤20-line limit.

## Reasoning

1. **Eliminates a redundant GPU copy**: The float32→bf16 cast of the full latent tensor is done once per step instead of twice. `predict_velocity` does not mutate its latents argument (it only reads it to build tokens), so sharing `lat_d` between the two passes is safe.
2. **Numerically identical**: Both passes receive exactly the same tensor value as before (`latents.to(dtype)`); we just compute it once instead of twice.
3. **No API breakage**: No signatures change; this is a local refactor inside the loop.

## Validation Protocol

Run benchmark test:
- 3 epochs × 30 steps, generate 4 images (with CFG enabled — the default `guidance_scale` for krea2 previews)
- Compare against set-1 best (training ~3.03s/it, samples ~65.12s/image)

## Expected Results

- **Sample generation**: 1-2% improvement (one fewer full-latent cast per step × 28 steps)
- **Training time**: no change (this is sampling-only code; training uses `get_noise_prediction`, not this loop)
- **VRAM**: negligible (one extra short-lived `lat_d` reference per step, freed at loop end)

## Known Limitations

1. **Sampling-only**: No effect on training time. The win is purely in the 28-step preview loop.
2. **Modest impact**: A single tensor cast is small relative to the two full MMDiT forward passes it feeds, so expect a low-single-digit percent at most.
3. **Only helps when CFG is on**: If `do_cfg` is False, there was only ever one cast, so this change is a no-op (still correct).

## User Action Required

1. Implement the change
2. Run benchmark test with the protocol above (ensure CFG / guidance_scale > 0 so both passes run)
3. Report sample generation times
4. If improvement >5% keep; otherwise evaluate as cumulative benefit (likely small)
