# Implementation Proposal #5: Dtype Conversion Optimization

## Status
⚠️ PROPOSED - Awaiting user testing

## Complexity
Simple (1-5 lines changed)

## Expected Impact
2-3% speedup

## Issue Description

In `get_noise_prediction` in `krea2.py`, timesteps are converted to float32 and then back to model dtype in the prediction loop. This redundant conversion can be eliminated by using the model dtype directly.

## Current Code

### Location: `extensions_built_in/diffusion_models/krea2/krea2.py`, lines 630-670

```python
def get_noise_prediction(
    self,
    latent_model_input: torch.Tensor,  # (B, 16, h, w)
    timestep: torch.Tensor,  # 0..1000 scale
    text_embeddings: AdvancedPromptEmbeds,
    batch: "DataLoaderBatchDTO" = None,
    **kwargs,
):
    if self.model.device == torch.device("cpu"):
        self.model.to(self.device_torch)

    # Clean reference latents from the batch's control images (if any); they
    # ride along in the sequence at t=0 and are never noised.
    ref_latents = None
    if batch is not None and self.is_edit:
        with torch.no_grad():
            _, _, lh, lw = latent_model_input.shape
            target_pixels = (lh * self.vae_scale_factor) * (
                lw * self.vae_scale_factor
            )
            ref_latents = self._batch_ref_latents_from_batch(
                batch, latent_model_input.shape[0], target_pixels=target_pixels
            )

    # toolkit timestep (0..1000, 1000 = pure noise) -> Krea flow time t in
    # [0, 1] with t=1 = pure noise. Same convention -> straight divide.
    t = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0
    if t.dim() == 0:
        t = t.unsqueeze(0)
    if t.shape[0] != latent_model_input.shape[0]:
        t = t.expand(latent_model_input.shape[0])

    context, text_mask = pad_text_features(
        text_embeddings.text_embeds, self.device_torch, self.torch_dtype
    )

    pred = predict_velocity(
        self.transformer,
        latent_model_input.to(self.device_torch, self.torch_dtype),
        t,
        context,
        text_mask,
        ref_latents=ref_latents,
        isolate_refs=self.kv_cache,
    )
    return pred
```

## Optimized Code

```python
def get_noise_prediction(
    self,
    latent_model_input: torch.Tensor,  # (B, 16, h, w)
    timestep: torch.Tensor,  # 0..1000 scale
    text_embeddings: AdvancedPromptEmbeds,
    batch: "DataLoaderBatchDTO" = None,
    **kwargs,
):
    if self.model.device == torch.device("cpu"):
        self.model.to(self.device_torch)

    # Clean reference latents from the batch's control images (if any); they
    # ride along in the sequence at t=0 and are never noised.
    ref_latents = None
    if batch is not None and self.is_edit:
        with torch.no_grad():
            _, _, lh, lw = latent_model_input.shape
            target_pixels = (lh * self.vae_scale_factor) * (
                lw * self.vae_scale_factor
            )
            ref_latents = self._batch_ref_latents_from_batch(
                batch, latent_model_input.shape[0], target_pixels=target_pixels
            )

    # toolkit timestep (0..1000, 1000 = pure noise) -> Krea flow time t in
    # [0, 1] with t=1 = pure noise. Same convention -> straight divide.
    # Use model dtype directly to avoid redundant conversion
    t = timestep.to(self.device_torch, dtype=self.torch_dtype) / 1000.0
    if t.dim() == 0:
        t = t.unsqueeze(0)
    if t.shape[0] != latent_model_input.shape[0]:
        t = t.expand(latent_model_input.shape[0])

    context, text_mask = pad_text_features(
        text_embeddings.text_embeds, self.device_torch, self.torch_dtype
    )

    pred = predict_velocity(
        self.transformer,
        latent_model_input.to(self.device_torch, self.torch_dtype),
        t,
        context,
        text_mask,
        ref_latents=ref_latents,
        isolate_refs=self.kv_cache,
    )
    return pred
```

### Also update predict_velocity to handle model dtype timesteps:

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
        t=t,  # Now accepts model dtype (bf16/fp16) instead of float32
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

- **Line 650**: Changed `dtype=torch.float32` to `dtype=self.torch_dtype`
- **predict_velocity**: Updated to accept model dtype timesteps directly

## Reasoning

The current implementation:
1. Converts timestep to float32
2. Passes to `predict_velocity`
3. Inside the model, operations may convert back to model dtype

By using model dtype directly:
1. No redundant conversion
2. Less memory movement
3. Faster execution

## Validation Protocol

Run benchmark test:
- 3 epochs × 30 steps
- Generate 4 images per epoch

Compare against baseline results in `results-baseline.md`.

## Expected Results

- **Training time**: 2-3% improvement (eliminated dtype conversion)
- **Sample generation**: 1-2% improvement (faster timestep handling)

## User Action Required

1. Test this change with the benchmark protocol
2. Report training time and sample generation times
3. If improvement >5%, keep the change; otherwise, revert
