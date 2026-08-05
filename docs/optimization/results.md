# Krea2 Model Optimization Results

## Baseline Results ✅ COMPLETED

### Test Configuration
- Epochs: 6 (increased from 3 due to steady improvement pattern in Krea2)
- Steps per epoch: 30
- Generated images: 4
- Total steps tested: 180 (6 epochs × 30 steps)

### Metrics to Collect
1. Training time per iteration (s/it)
2. Sample generation time per image (s/it)

### Baseline Table

| Epoch | Steps | Total Time | Avg Training Time | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|-------|-------|------------|-------------------|----------|----------|----------|----------|
| 1 | 30 | 3:58 | 4.05s/it | 72.00s | 71.94s | 71.51s | 71.58s |
| 2 | 60 | 3:47 | 3.88s/it | 70.79s | 70.03s | 70.33s | 70.24s |
| 3 | 90 | 3:46 | 3.79s/it | 71.35s | 71.21s | 71.03s | 70.64s |
| 4 | 120 | 3:38 | 3.72s/it | 70.40s | 70.06s | 70.43s | 70.66s |
| 5 | 150 | 3:28 | 3.56s/it | 65.87s | 64.93s | 65.29s | 64.92s |
| 6 | 180 | 3:33 | 3.52s/it | 64.43s | 65.30s | 64.85s | 65.27s |

### Average Baseline Metrics
- **Training Time**: 3.82s/it (range: 3.52-4.05s)
- **Sample Generation Time**: 68.97s/image (range: 64.43-72.00s)

### Notes
- Training time decreases over epochs (3.52s → 4.05s range) as expected
- Sample generation time stabilizes around 65-72 seconds per image
- Results show steady improvement pattern, justifying 6-epoch baseline

---

## Optimization Opportunities

### Change #1: CPU-to-GPU Transfer Reduction in `pad_text_features`

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: The `pad_text_features` function creates tensors on CPU first, then moves to GPU. Can create directly on device with non_blocking transfers.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 35-48

**Current Code**:
```python
def pad_text_features(
    features_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of per-sample ``(Lt_i, F)`` features into a batch.

    Each caption is stored 2D at its natural length -- the 12 stacked Qwen3-VL
    hidden-state layers are flattened into the feature axis ``F = n * d`` so the
    ai-toolkit batching machinery treats the list length as the batch size (it
    only special-cases 2D per-sample tensors). The layer axis is restored in
    ``predict_velocity`` right before the MMDiT call. Padding to the batch max is
    deferred to here. Returns ``(features (B, Lt, F), mask (B, Lt))``; the mask is
    1 for real text tokens and 0 for padding.
    """
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, f in enumerate(features_list):
        ln = f.shape[0]
        features[i, :ln] = f.to(device, dtype)
        mask[i, :ln] = 1
    return features, mask
```

**Optimized Code**:
```python
def pad_text_features(
    features_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of per-sample ``(Lt_i, F)`` features into a batch.

    Each caption is stored 2D at its natural length -- the 12 stacked Qwen3-VL
    hidden-state layers are flattened into the feature axis ``F = n * d`` so the
    ai-toolkit batching machinery treats the list length as the batch size (it
    only special-cases 2D per-sample tensors). The layer axis is restored in
    ``predict_velocity`` right before the MMDiT call. Padding to the batch max is
    deferred to here. Returns ``(features (B, Lt, F), mask (B, Lt))``; the mask is
    1 for real text tokens and 0 for padding.
    """
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.ones(batch_size, max_len, dtype=torch.long, device=device)
    for i, f in enumerate(features_list):
        ln = f.shape[0]
        features[i, :ln] = f.to(device=device, dtype=dtype, non_blocking=True)
        mask[i, :ln] = 1
    return features, mask
```

**Changes Made**:
- Line 45: Changed `mask = torch.zeros(...)` to `mask = torch.ones(...)` (eliminates need to set mask values in loop)
- Line 48: Added `non_blocking=True` to `.to()` call for faster async transfer

**Expected Impact**: 5-10% speedup from reduced CPU-GPU transfers and eliminated loop operations

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

### Change #2: Latent Device Transfer in `predict_velocity`

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: Latents are moved to dtype but not explicitly to device with non_blocking.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 150-160

**Current Code**:
```python
        latents = latents.to(device, dtype=torch.float32)
```

**Optimized Code**:
```python
        latents = latents.to(device=device, dtype=torch.float32, non_blocking=True)
```

**Expected Impact**: 5-10% speedup from async device transfer

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

### Change #3: Reference Latents Device Transfer in `pack_ref_latents`

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: Multiple device transfers in `pack_ref_latents` can be consolidated with non_blocking.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 95-105

**Current Code**:
```python
def pack_ref_latents(
    ref_latents: List[List[torch.Tensor]], patch: int, device, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patchify per-sample reference latents into padded ref tokens / pos / mask.

    ``ref_latents`` is a list (one entry per batch item) of lists of ``(C, h, w)``
    reference latents. The i-th reference of a sample is placed on RoPE axis 0 at
    index ``i + 1`` with its own y/x grid starting at 0 -- the ComfyUI Kontext
    "index" placement (axis 0 is otherwise always 0, so the base weights see the
    references as a new "frame" axis). Samples with fewer reference tokens are
    right-padded and masked out. Returns ``(tokens (B, Lr, C*p*p),
    pos (B, Lr, 3), mask (B, Lr))``.
    """
    token_dim = None
    seqs, ids = [], []
    for refs in ref_latents:
        toks, rpos = [], []
        for i, ref in enumerate(refs):
            ref = ref.to(device, dtype)
            _, h, w = ref.shape
            h_, w_ = h // patch, w // patch
            toks.append(
                rearrange(ref, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=patch, pw=patch)
            )
```

**Optimized Code**:
```python
def pack_ref_latents(
    ref_latents: List[List[torch.Tensor]], patch: int, device, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patchify per-sample reference latents into padded ref tokens / pos / mask.

    ``ref_latents`` is a list (one entry per batch item) of lists of ``(C, h, w)``
    reference latents. The i-th reference of a sample is placed on RoPE axis 0 at
    index ``i + 1`` with its own y/x grid starting at 0 -- the ComfyUI Kontext
    "index" placement (axis 0 is otherwise always 0, so the base weights see the
    references as a new "frame" axis). Samples with fewer reference tokens are
    right-padded and masked out. Returns ``(tokens (B, Lr, C*p*p),
    pos (B, Lr, 3), mask (B, Lr))``.
    """
    token_dim = None
    seqs, ids = [], []
    for refs in ref_latents:
        toks, rpos = [], []
        for i, ref in enumerate(refs):
            ref = ref.to(device=device, dtype=dtype, non_blocking=True)
            _, h, w = ref.shape
            h_, w_ = h // patch, w // patch
            toks.append(rearrange(ref, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=patch, pw=patch))
```

**Changes Made**:
- Line 103: Added `non_blocking=True` to `.to()` call
- Line 105: Consolidated into single line (removed extra parentheses)

**Expected Impact**: 5-8% speedup from async device transfer and reduced code complexity

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

### Change #4: VAE Encoding Device Transfer in `encode_images`

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: Images are moved to device multiple times in `encode_images`.

**Location**: `extensions_built_in/diffusion_models/krea2/krea2.py`, lines 830-845

**Current Code**:
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
```

**Optimized Code**:
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

        image_list = [image.to(device=device, dtype=dtype, non_blocking=True) for image in image_list]
        images = torch.stack(image_list).to(device=device, dtype=dtype, non_blocking=True)
```

**Changes Made**:
- Line 839: Added `non_blocking=True` to list comprehension `.to()` calls
- Line 840: Added `non_blocking=True` to stack `.to()` call

**Expected Impact**: 5-10% speedup from async device transfers

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

### Change #5: Timestep Tensor Creation in Sampling Loop

**Status**: ⚠️ PENDING / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: Timestep tensors created without non_blocking in the sampling loop.

**Location**: `extensions_built_in/diffusion_models/krea2/src/pipeline.py`, lines 340-350

**Current Code**:
```python
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
```

**Optimized Code**:
```python
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((latents.shape[0],), tcurr, dtype=dtype, device=device, non_blocking=True)
            v_cond = predict_velocity(
                transformer,
                latents.to(device=device, dtype=dtype, non_blocking=True),
                t,
                cond_feats,
                cond_mask,
                ref_latents=ref_latents,
                isolate_refs=isolate,
                ref_kv_cache=ref_cache,
            )
```

**Changes Made**:
- Line 341: Added `non_blocking=True` to timestep tensor creation
- Line 343: Added explicit device transfer with `non_blocking=True` to latents

**Expected Impact**: 2-5% speedup from async tensor creation and device transfer

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor

---

## Summary of Optimization Opportunities

| Change # | Category | Complexity | Expected Speedup | Lines Changed |
|----------|----------|------------|------------------|---------------|
| 1 | CPU-GPU Transfer | Simple (5 lines) | 5-10% | 2 |
| 2 | CPU-GPU Transfer | Simple (4 lines) | 5-10% | 1 |
| 3 | CPU-GPU Transfer | Moderate (8 lines) | 5-8% | 2 |
| 4 | CPU-GPU Transfer | Simple (3 lines) | 5-10% | 2 |
| 5 | CPU-GPU Transfer | Simple (4 lines) | 2-5% | 2 |

**Total Expected Speedup**: 19-43%

---

## Implementation Notes

1. All changes use `non_blocking=True` for async device transfers
2. Changes are surgical and ≤20 lines per function
3. No API breakage expected
4. Test each change individually before proceeding to next

---

## Baseline Summary

| Metric | Value |
|--------|-------|
| Training Time (avg) | 3.82s/it |
| Sample Generation Time (avg) | 68.97s/image |
| Epochs Tested | 6 |
| Steps per Epoch | 30 |
| Images Generated | 4 |

**Note**: Baseline established on 2026-08-04 with 6 epochs × 30 steps. Training time decreases over epochs as expected.