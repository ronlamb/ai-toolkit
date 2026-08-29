# Change #16: Lean `ropeapply` — apply RoPE in bf16 instead of a full fp32 round-trip

**Status**: PROPOSED (not yet implemented)
**Complexity**: Simple (~5 lines in one function)
**Expected Impact**: ~1–2% training, ~1–2% sampling (measured micro-bench below)
**Applies to**: both loops — `ropeapply` runs once per block (28×/forward in the main blocks)

## Issue

`extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `ropeapply()` (~line 42):

```python
def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)
```

`xq.float()` / `xk.float()` materialize **full fp32 copies of q and k** (at L=4352: 60 heads × 4352 × 128 × 4 B ≈ **134 MB per call**), the element-wise multiply/add runs in fp32, then `.to(xq.dtype)` downcasts back to bf16. Same pattern as #15: memory-bound elementwise work paying ~2× the bytes, and the result is rounded to bf16 anyway.

**Measured on this machine (RTX 4090, torch 2.9.1, bf16, L=4352, q=48 heads + k=12 heads):**

| micro-bench | current | lean (bf16) |
|---|---|---|
| one `ropeapply` forward | 1.49 ms | **0.51 ms** |
| full `SingleStreamBlock` fwd+bwd (on top of #15) | 101.6 ms | **99.4 ms** |

≈ 28 calls/forward × ~1 ms saved ≈ **~28 ms/step forward-only**, plus the same again in the checkpoint-recompute pass → combined with #15, block fwd+bwd drops **105.2 → 99.4 ms (−5.5% per block)**. Sampling benefits too: ~1 ms × 28 blocks ≈ 28 ms off every forward (×2 under CFG).

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
