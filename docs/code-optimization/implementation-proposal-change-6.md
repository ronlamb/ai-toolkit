# Implementation Proposal #6: Cache VAE Normalization Constants

## Status
⬜ PROPOSED - Awaiting implementation and user testing

## Complexity
Simple (1-5 lines changed)

## Expected Impact
1-2% speedup (eliminates repeated CPU→GPU tensor creation on every VAE encode/decode)

## Focus Area
- **Excessive CPU→GPU copies**: `torch.tensor(python_list)` creates a CPU tensor, then `.to(device, dtype)` copies it to GPU. This happens on every `encode_images` / `decode_latents` call.
- **Memory usage waste**: Two small tensors are allocated and freed on every call instead of being held once.

## Issue Description

Both `encode_images` and `decode_latents` in `krea2.py` build the VAE latent normalization constants from Python config lists on **every call**:

```python
latents_mean = (
    torch.tensor(self.vae.config.latents_mean)      # CPU tensor from Python list
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)          # CPU -> GPU copy
)
latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
    1, self.vae.config.z_dim, 1, 1, 1
).to(latents.device, latents.dtype)                 # CPU -> GPU copy
```

`self.vae.config.latents_mean` and `latents_std` are fixed per-VAE (16 values each for the Qwen-Image VAE). They never change after load. Yet:

- **Training**: `encode_images` is called once per training step (to encode the clean image into latents). That's 2 CPU→GPU copies + 2 allocations per step.
- **Sampling**: `decode_latents` is called once per generated image (28-step loop already done, so 1 decode).

The values are constant for the lifetime of the model. They should be computed **once** and cached as model attributes, then reused on every call.

## Current Code

### Location: `extensions_built_in/diffusion_models/krea2/krea2.py`

#### `encode_images` (lines ~780-815)
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

#### `decode_latents` (lines ~817-850)
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

## Optimized Code

### Step 1: Add a cached-constants helper (called once, e.g. in `load_model` after VAE load)

Add a small method that builds and stores the constants once. The constants are
stored in **float32** (the source-of-truth precision) and cast to the call's
`dtype` at use time — a cheap 16-element cast, not a CPU→GPU copy:

```python
def _cache_vae_norm_constants(self):
    """Build the VAE latent normalization constants once and cache them.

    ``latents_mean`` / ``latents_std`` are fixed per-VAE (16 values for the
    Qwen-Image VAE) and never change after load. Building them from Python
    config lists on every encode/decode caused a CPU->GPU copy each time;
    cache them as (1, z_dim, 1, 1, 1) float32 tensors on the VAE device
    instead and cast to the call dtype at use time.
    """
    z = self.vae.config.z_dim
    mean = (
        torch.tensor(self.vae.config.latents_mean, dtype=torch.float32)
        .view(1, z, 1, 1, 1)
        .to(self.vae_device_torch)
    )
    std = (
        torch.tensor(self.vae.config.latents_std, dtype=torch.float32)
        .view(1, z, 1, 1, 1)
        .to(self.vae_device_torch)
    )
    # encode uses (x - mean) * (1/std); decode uses x * std + mean.
    self._vae_latents_mean = mean
    self._vae_latents_std_inv = 1.0 / std   # for encode
    self._vae_latents_std = std             # for decode
```

Call it at the end of `load_model`, right after `vae.to(...)`:
```python
        vae = self._load_vae()
        vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)
        self._cache_vae_norm_constants()   # <-- add this line
```

### Step 2: Use the cached constants in `encode_images`

Replace the two inline tensor builds with a single lookup (cast to call dtype):
```python
    latents = torch.stack(latents)  # (B, 16, h, w)

    mean = self._vae_latents_mean.to(dtype=dtype)
    std_inv = self._vae_latents_std_inv.to(dtype=dtype)
    latents = (latents - mean) * std_inv
    return latents.to(device, dtype=dtype)
```

### Step 3: Use the cached constants in `decode_latents`

Replace the two inline tensor builds with a single lookup (cast to call dtype):
```python
    latents = latents.unsqueeze(2)

    mean = self._vae_latents_mean.to(dtype=dtype)
    std = self._vae_latents_std.to(dtype=dtype)
    latents = latents * std + mean

    # Full-resolution decode spikes VRAM; tile it when low on VRAM (decode
    # only -- encode stays untiled).
    tiled = self.model_config.low_vram
```

## Changes Summary

- **New method** `_cache_vae_norm_constants()`: builds the 3 constant tensors once (mean, std_inv for encode, std for decode) in float32 on the VAE device and stores them as attributes.
- **`load_model`**: one added line to call the cache builder after VAE load.
- **`encode_images`**: 2 inline `torch.tensor(...).to(...)` blocks (6 lines) → 3 lines using cached attributes + a cheap dtype cast.
- **`decode_latents`**: 2 inline `torch.tensor(...).to(...)` blocks (8 lines) → 3 lines using cached attributes + a cheap dtype cast.

**Net modified lines per function**: `encode_images` ~5, `decode_latents` ~6. All within the ≤20-line limit.

## Reasoning

1. **Eliminates CPU→GPU copies**: `torch.tensor(python_list)` allocates on CPU; `.to(device, dtype)` then copies to GPU. This happened every encode/decode call. Now it happens exactly once at load time.
2. **Eliminates repeated allocation**: The 3 small tensors are allocated once and reused, instead of being created and garbage-collected on every call.
3. **Numerically identical**: The math is unchanged — same values, same shapes `(1, z_dim, 1, 1, 1)`, same dtype/device. Only the *when* they're built changes (once vs every call).
4. **No API breakage**: `encode_images` / `decode_latents` signatures and return values are unchanged.

## Validation Protocol

Run benchmark test:
- 3 epochs × 30 steps, generate 4 images
- Compare against set-1 best (training ~3.03s/it, samples ~65.12s/image)

## Expected Results

- **Training time**: small improvement (encode called once/step; 2 fewer CPU→GPU copies + allocations per step)
- **Sample generation**: small improvement (decode called once/image)
- **VRAM**: negligible change (3 tiny 16-element tensors held permanently)

## Known Limitations

1. The constants are cached on `self.vae_device_torch` / `self.vae_torch_dtype`. If the VAE is later moved to a different device/dtype (e.g. low-VRAM offload), the cache would need rebuilding. In practice `encode_images`/`decode_latents` already move the VAE to `device` before use, and the constants are tiny — if a device mismatch is ever observed, rebuild via `_cache_vae_norm_constants()`.
2. Impact is small (1-2%) because the tensors are tiny; the win is removing redundant work, not a large compute reduction.

## User Action Required

1. Implement the change
2. Run benchmark test with the protocol above
3. Report training time, sample generation times
4. If improvement >5% keep; otherwise evaluate as cumulative benefit
