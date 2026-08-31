# Change #16: Lean `ropeapply` — apply RoPE in bf16 instead of a full fp32 round-trip

**Status**: ✅ COMPLETED (kept) — tested 2026-08-29, extended to 10 epochs by user
**Complexity**: Simple (~5 lines in one function)
**Expected Impact**: ~1–2% training, ~1–2% sampling (measured micro-bench below)
**Applies to**: both loops — `ropeapply` runs once per block (28×/forward in the main blocks)

## Issue

`extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `ropeapply()` (~line 42):

```python
def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    # Apply the rotation directly in x's dtype. The fp32 round-trip materializes
    # ~134 MB of upcast q/k per call only to be downcast again at the end.
    freqs = freqs.to(xq.dtype)[:, None, :, :, :]
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape), xk_.reshape(*xk.shape)
```

(`freqs.to(xq.dtype)` is a ~0.5 MB cast of the frequency table, not the activations.)

**Numerics (measured with identical fixed upstream gradients)**: mean relative error **0.63%** on outputs and on `grad_q`/`grad_k`. The current path computes `round_bf16(fp32(f·x))`; the lean path computes `round_bf16(bf16(f)·x)` — the difference is one bf16 rounding of the cos/sin table (bf16 epsilon ≈ 0.4%, so this is same-order noise). q/k are already bf16 outputs of `QKNorm`/`RMSNorm`, and downstream SDPA (cuDNN) accumulates in fp32 regardless. Same class as set-1 #5 (timestep dtype), which *improved* convergence metrics.

**Risk note**: this touches q/k before attention, i.e. the model's positional signal — more sensitive than #15. Validation therefore includes a fixed-seed visual sample comparison in addition to the unit check. If loss curves or samples look off, revert (one function, `git checkout`).

## Proposed change

```python
def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    # Apply the rotation directly in x's dtype. The fp32 round-trip materializes
    # ~134 MB of upcast q/k per call only to be downcast again at the end.
    freqs = freqs.to(xq.dtype)[:, None, :, :, :]
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape), xk_.reshape(*xk.shape)
```

(`freqs.to(xq.dtype)` is a ~0.5 MB cast of the frequency table, not the activations.)

**Numerics (measured with identical fixed upstream gradients)**: mean relative error **0.63%** on outputs and on `grad_q`/`grad_k`. The current path computes `round_bf16(fp32(f·x))`; the lean path computes `round_bf16(bf16(f)·x)` — the difference is one bf16 rounding of the cos/sin table (bf16 epsilon ≈ 0.4%, so this is same-order noise). q/k are already bf16 outputs of `QKNorm`/`RMSNorm`, and downstream SDPA (cuDNN) accumulates in fp32 regardless. Same class as set-1 #5 (timestep dtype), which *improved* convergence metrics.

**Risk note**: this touches q/k before attention, i.e. the model's positional signal — more sensitive than #15. Validation therefore includes a fixed-seed visual sample comparison in addition to the unit check. If loss curves or samples look off, revert (one function, `git checkout`).

## Validation plan

- Unit check: max abs / mean rel diff old-vs-new forward and gradients on random bf16 tensors (< 1% rel expected; measured 0.63%).
- Fixed-seed sample comparison: one preview image before/after at the same seed/steps — structure must match (differences only at bf16 noise level).
- `pytest tests/` (44 passed baseline).
- Benchmark: test **after** #15 (so the delta is attributable); 6+ epochs × 30 steps, 4 images; compare bottom-out s/it + samples vs current best (#10 state), same dataset mix.
- Keep if beyond variance; negligible → user decides; slower → revert (`git checkout -- extensions_built_in/diffusion_models/krea2/src/mmdit.py`).

## Validation results (2026-08-29, pre-benchmark)

Implemented in `extensions_built_in/diffusion_models/krea2/src/mmdit.py` exactly as proposed
(8 lines changed, one function). Equivalence script: `.tmp_opt_test/check_change16.py`
(old = `git show HEAD:...` → `.tmp_opt_test/mmdit_old.py`; realistic shapes
q=(1,48,4352,128), k=(1,12,4352,128) bf16, identical fixed upstream grads).

| Check | Result |
|---|---|
| Forward mean rel diff (q / k) | **0.208 %** / 0.208 % (< 1 % threshold; better than the 0.63 % projected) |
| Grad mean rel diff (grad_q / grad_k) | 0.208 % / 0.208 % |
| Output dtype | bf16, gradients flow through new path ✔ |
| Micro-bench fwd (L=4352) | old **1.538 ms** → new **0.532 ms** (−65 %, matches proposal's 1.49→0.51) |
| `pytest tests/` | **44 passed** |

Awaiting user benchmark: 6 epochs × 30 steps, 4 images; compare bottom-out s/it and sample
times vs #14 best (training ~3.15–3.18 s/it, samples ~65.8 s/img), same dataset mix.
Also do the fixed-seed preview-image comparison before/after (risk note above).

## Benchmark results (tested 2026-08-29)

Short bench, same dataset mix. Cumulative `s/it` at epoch end; per-step avg from total-time deltas.

| Epoch | Cum s/it | Per-step avg s/it | Samples avg (s/img) |
|-------|----------|-------------------|---------------------|
| 1 | 3.71 | warm-up | 63.77 |
| 2 | 3.30 | 2.90 | 64.74 |
| 3 | 3.29 | 3.27 | 65.02 |
| 4 | 3.14 | 2.70 | 65.22 |
| 5 | 3.17 | 3.27 | 65.61 |
| 6 | 3.09 | 2.70 | 63.22 |

| Metric | #14 best | #16 | Delta |
|--------|----------|-----|-------|
| Epochs 4–6 avg cum (s/it) | ~3.15–3.18 | 3.13 | −0.6…−1.5% |
| Final cumulative / bottom-out (s/it) | 3.15–3.16 | **3.09** | **−2.0…−2.2%** |
| Samples epochs 4–6 avg (s/img) | ~65.8 | 64.7 | **−1.7%** |

Both bottom-out metrics improved and the direction matches the micro-bench prediction
(~28 ms/step forward + recompute pass ≈ 1–2 %). Gain is at the upper edge of the ±1–2 %
variance band — clearly better than #15 (dead-even/slower), but borderline. **User decides.**
