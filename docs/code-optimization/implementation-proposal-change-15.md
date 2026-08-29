# Change #15: Lean `RMSNorm` — drop the per-call fp32 round-trip

**Status**: PROPOSED (not yet implemented)
**Complexity**: Simple (~6 lines in one class)
**Expected Impact**: ~1–3% training, small sample improvement (block-level measurement below)
**Applies to**: both loops — `RMSNorm` runs 58× per forward in the main blocks (prenorm+postnorm × 28 + `LastLayer`), plus 4× inside txtfusion when not pre-fused

## Issue

`extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `RMSNorm.forward` (~line 172):

```python
    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0)
        )
        return t.to(dtype)
```

Every call allocates a **full fp32 copy of the activation** (at 1024²: 4864×6144×4 B ≈ **120 MB per call**), upcasts the weight, runs `F.rms_norm` in fp32, then downcasts. RMS is memory-bound; the fp32 round-trip multiplies bytes moved by ~1.5–2×. The extra precision is wasted anyway: the result is immediately downcast to bf16 by `.to(dtype)`.

**Measured on this machine (RTX 4090, torch 2.9.1, bf16):**

| micro-bench (L=4864, features=6144) | current | lean |
|---|---|---|
| one RMSNorm forward | 0.80 ms | **0.10 ms** |
| one RMSNorm forward+backward | 2.20 ms | **0.80 ms** |

| full `SingleStreamBlock` fwd+bwd (L=4352) | current | lean RMS only |
|---|---|---|
| per block | 105.2 ms | **101.6 ms** (−3.4%) |

≈ 56 calls/forward × ~1.4 ms saved fwd+bwd ≈ **~80 ms/step** with checkpointing recompute → ~2–3% of a ~3 s step. Sampling (forward-only) also benefits: the fp32 copies disappear from every norm in all 28 blocks + last layer, twice per step under CFG.

## Proposed change

```python
    def forward(self, x: Tensor) -> Tensor:
        # F.rms_norm requires weight.dtype == x.dtype; compute directly in x's
        # dtype instead of materializing a full fp32 copy of the activation.
        w = (self.scale + 1.0).to(x.dtype)
        return F.rms_norm(x, (self.features,), eps=self.eps, weight=w)
```

**Numerics (measured with identical fixed upstream gradients)**: mean relative gradient error **0.15%** on both `grad_x` and `grad_scale` — pure bf16 rounding noise (bf16 epsilon ≈ 0.4%), same class as set-1 #5's timestep-dtype change which *improved* convergence metrics.

Note: `self.scale` stays a fp32 `Parameter` (optimizer state / master weights unchanged); only the per-call weight cast changes. No `state_dict` change, meta-device safe (no buffers). QK-norm (`QKNorm` wraps two `RMSNorm`s on q/k) uses the same class — covered by this change.

## Rejected alternative (measured first — recorded here so it is not re-proposed)

Replaced `with sdpa_kernel(SDPBackend.CUDNN_ATTENTION)` in `attention()` with a backend preference list `[FLASH, CUDNN, EFFICIENT]`. Micro-benchmarks on this machine (torch 2.9.1, bf16, GQA 48/12 heads): **forced cuDNN is already fastest** — masked fwd 3.4 ms vs 5–6 ms for the list; unmasked 3.1 ms vs ~4 ms; gqa shapes 0.2 ms vs 0.3 ms at short L. A preference list would *regress* every attention call. Keep the cuDNN pin. (The module docstring claiming forcing was removed is stale — code still forces it.)

## Validation plan

- Unit check: max abs / rel diff old-vs-new forward and gradients on random bf16 tensors (< 1% rel expected; measured 0.15%).
- `pytest tests/` (44 passed baseline).
- Benchmark: 6+ epochs × 30 steps, 4 images; compare **bottom-out s/it** + sample times vs current best (#10 state) with the same dataset mix. Test after #14 so deltas are attributable per change.
- Keep if beyond variance; negligible → user decides; slower → revert (`git checkout -- extensions_built_in/diffusion_models/krea2/src/mmdit.py`).
