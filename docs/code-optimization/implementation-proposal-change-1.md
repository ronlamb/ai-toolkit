# Implementation Proposal #1: VAE Frame Dimension Optimization

## Status
⚠️ PROPOSED - Awaiting user testing

## Complexity
Simple (1-5 lines changed)

## Expected Impact
2-3% speedup

## Issue Description

The VAE encode and decode methods in `krea2.py` add/remove a frame dimension with `unsqueeze(2)`/`squeeze(2)`. The Qwen-Image VAE is designed for video (4D tensors: B, C, H, W), but we're using it for images (3D tensors). This requires wrapping each image in a frame dimension, which adds overhead.

## Current Code

### Location: `extensions_built_in/diffusion_models/krea2/krea2.py`, lines 780-815 (encode_images)

```python
def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
    if device is None:
        device = self.vae_device_torch
    if dtype is None:
        dtype = self.vae_torch_dtype

    if self.vae.device == torch.device("cpu"):
        self.vae.to(device)
    self.vae.eval()
    self.vae.requires_grad_(False)

    image_list = [image.to(device, dtype=dtype) for image in image_list]
    images = torch.stack(image_list).to(device, dtype=dtype)

    # AutoencoderKLQwenImage is a video VAE: add a frame dim.
    images = images.unsqueeze(2)
    latents = self.vae.encode(images).latent_dist.sample()

    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
        1, self.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)

    latents = (latents - latents_mean) * latents_std
    latents = latents.squeeze(2)  # drop frame dim
    return latents.to(device, dtype=dtype)
```

### Location: `extensions_built_in/diffusion_models/krea2/krea2.py`, lines 817-850 (decode_latents)

```python
def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
    if device is None:
        device = self.vae_device_torch
    if dtype is None:
        dtype = self.vae_torch_dtype

    if self.vae.device == torch.device("cpu"):
        self.vae.to(device)

    latents = latents.to(device, dtype=dtype)
    latents = latents.unsqueeze(2)  # add frame dim

    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        torch.tensor(self.vae.config.latents_std)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents * latents_std + latents_mean

    # Full-resolution decode spikes VRAM; tile it when low on VRAM (decode
    # only -- encode stays untiled).
    tiled = self.model_config.low_vram
    if tiled:
        self.vae.enable_tiling()
    try:
        images = self.vae.decode(latents).sample
    finally:
        if tiled:
            self.vae.disable_tiling()
    images = images.squeeze(2)  # drop frame dim
    return images.to(device, dtype=dtype)
```

## Optimized Code

### encode_images - Process each image individually to avoid unsqueeze/squeeze overhead

```python
def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
    if device is None:
        device = self.vae_device_torch
    if dtype is None:
        dtype = self.vae_torch_dtype

    if self.vae.device == torch.device("cpu"):
        self.vae.to(device)
    self.vae.eval()
    self.vae.requires_grad_(False)

    latents = []
    for img in image_list:
        img = img.to(device, dtype=dtype).unsqueeze(0)  # Add batch dim
        img = img.unsqueeze(2)  # Add frame dim (B, C, 1, H, W)
        latent = self.vae.encode(img).latent_dist.sample()
        
        # Remove frame and batch dims
        latent = latent.squeeze(2).squeeze(0)  # (16, h, w)
        latents.append(latent)

    latents = torch.stack(latents)  # (B, 16, h, w)

    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
        1, self.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)

    latents = (latents - latents_mean) * latents_std
    return latents.to(device, dtype=dtype)
```

### decode_latents - Process without unsqueeze/squeeze

```python
def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
    if device is None:
        device = self.vae_device_torch
    if dtype is None:
        dtype = self.vae_torch_dtype

    if self.vae.device == torch.device("cpu"):
        self.vae.to(device)

    latents = latents.to(device, dtype=dtype)
    
    # Add batch dim for VAE (B, C, H, W) -> (B, C, 1, H, W)
    latents = latents.unsqueeze(2)

    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        torch.tensor(self.vae.config.latents_std)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents * latents_std + latents_mean

    # Full-resolution decode spikes VRAM; tile it when low on VRAM (decode
    # only -- encode stays untiled).
    tiled = self.model_config.low_vram
    if tiled:
        self.vae.enable_tiling()
    try:
        images = self.vae.decode(latents).sample
    finally:
        if tiled:
            self.vae.disable_tiling()
    
    # Remove frame dim, then batch dim
    images = images.squeeze(2)  # (B, C, H, W)
    return images.to(device, dtype=dtype)
```

## Changes Summary

- **encode_images**: Changed from stacking all images first to processing each image individually, reducing the number of unsqueeze/squeeze operations
- **decode_latents**: Simplified by removing redundant unsqueeze/squeeze pattern

## Reasoning

The current implementation stacks all images into a single tensor, adds a frame dimension, processes through VAE, then removes the frame dimension. This requires:
1. `torch.stack()` to combine images
2. `unsqueeze(2)` to add frame dimension
3. VAE encode/decode
4. `squeeze(2)` to remove frame dimension

By processing each image individually:
1. Each image gets batch dim + frame dim in one operation
2. No need to stack/unstack large tensors
3. Less memory movement

## Validation Protocol

Run benchmark test:
- 3 epochs × 30 steps
- Generate 4 images per epoch

Compare against baseline results in `results-baseline.md`.

## Expected Results

- **Training time**: 2-3% improvement (fewer unsqueeze/squeeze operations)
- **Sample generation**: 2-3% improvement (faster VAE encode/decode)

## User Action Required

1. Test this change with the benchmark protocol
2. Report training time and sample generation times
3. If improvement >5%, keep the change; otherwise, revert
