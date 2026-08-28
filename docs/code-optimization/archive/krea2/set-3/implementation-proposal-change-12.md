# Implementation Proposal #12: RoPE in float32 with cached omega (drop per-call float64 rebuild)

## Status
⚠️ REVERTED — No measurable improvement (training +3.4% vs change #10, within
run-to-run variance; samples flat at −0.1%). Code restored to original float64
`rope`; 44/44 tests re-verified after revert.

**Unit check result (pre-revert)**: max abs diff old (fp64) vs new
(fp32 cached omega) = **3.95e-06** across 5 random integer-position trials
(b∈1–3, n∈1–3, pos<128) — well under the 1e-5 revert threshold.
`state_dict()` stays empty (no new buffers), meta-device construction verified
safe, output dtype float32.

**Benchmark result (6 epochs × 30 steps, 4 images)**: training 3.26–3.81 s/it
(bottom-out 3.26, cumulative-rate metric), samples 66.98–67.76 s/image
(epochs 4-6 avg 67.15). Vs change #10 stable (3.22 s/it, 67.21 s/image):
training +3.4% (variance — the change removes work, cannot logically slow),
samples −0.1% (flat). Full table in `current-state.md`.

## Complexity
Simple (~15 lines across `rope` + `PositionalEncoding`)

## Expected Impact
~0.1–0.5% training speedup (removes float64 trig on GPU + 3 per-call frequency
rebuilds every forward). Small, but it is exactly the "wasted number conversion"
class: float64 has no fast path on most GPUs and buys nothing here.

## Issue Description

`SingleStreamDiT.forward` calls `freqs = self.posemb(pos)` once per forward
(outside the block loop). `PositionalEncoding.forward` calls `rope()` three times
(one per position axis, axdims = [32, 48, 48]):

```python
def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()
```

Per forward this:
1. Rebuilds `scale`/`omega` (arange + pow) **in float64** — 3× per step, although
   the values are constant for the module's lifetime (theta/ntk/dims never change).
2. Runs `einsum` + `cos`/`sin` **in float64** over (B, L, d/2) — for a 1024px
   image that's ~4608 tokens × 12 freqs per axis. Float64 trig on GPU is slow
   (no fast SFU path) and the result is immediately downcast with `.float()`.
3. The float64 precision buys nothing: `pos` holds small integers (≤ ~128), the
   downstream model runs in bf16 (relative error ~4e-3), and `ropeapply` consumes
   the freqs in a bf16 q/k multiply. fp32 trig error (~1e-5 absolute on the
   argument) is ~1000× smaller than the model's own rounding.

## Proposed Change

Cache `omega` lazily as a plain (non-buffer) attribute — **not** a registered
buffer, because the transformer is built on `torch.device("meta")` and loaded with
`load_state_dict(..., strict=True)`, so any new buffer would break the strict load.
A plain attribute initialized to `None` and built on first forward is meta-safe,
survives device moves (rebuilt if the device changes), and stays out of the state dict.

```python
def rope(pos: Tensor, omega: Tensor) -> Tensor:
    # pos (B, N) float32 integer positions; omega (d/2,) cached float32 freqs.
    out = pos.unsqueeze(-1) * omega          # (B, N, d/2), float32
    c = torch.cos(out)
    s = torch.sin(out)
    out = torch.stack([c, -s, s, c], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims
        # Built lazily on first forward (module is constructed on the meta device,
        # so nothing tensor-valued may be created in __init__). Plain attribute on
        # purpose: a registered buffer would break strict state-dict loading.
        self._omega = None

    def forward(self, pos: Tensor) -> Tensor:
        if self._omega is None or self._omega.device != pos.device:
            self._omega = torch.cat([
                1.0 / ((self.theta * self.ntk) ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
                for d in self.axdims
            ]).to(pos.device)
        parts, off = [], 0
        for i, d in enumerate(self.axdims):
            half = d // 2
            parts.append(rope(pos[..., i], self._omega[off:off + half]))
            off += half
        return torch.cat(parts, dim=-3)
```

Notes:
- `pos` is already float32 (built from `torch.zeros`/`arange` in `prepare()`), so
  the multiply stays fp32 end-to-end; no cast needed.
- `rope`'s signature changes, but it is module-private (only called from
  `PositionalEncoding.forward` — verified by search). No API breakage.
- The final `.float()` in the old `rope` disappears (output is already fp32).

## Location
`extensions_built_in/diffusion_models/krea2/src/mmdit.py`, `rope` (line 31) and
`PositionalEncoding` (line 136)

## Lines Changed
~15 total (`rope`: ~7, `__init__`: +2, `forward`: ~8) — all ≤ 20 per function

## Precision
fp64 → fp32 on the frequency computation. Worst-case argument error ~1e-5 absolute
vs ~1e-14 before; the freqs feed bf16 q/k (relative error 4e-3). Net effect on the
model is far below its own dtype noise. Does not hinder precision in any practical sense.

## Validation Plan
1. Unit check: build a small `PositionalEncoding`, compare old vs new output on a
   random integer `pos` — expect max abs diff < 1e-5 (fp32 vs fp64 trig only).
2. Speed test per protocol: 8 epochs × 30 steps, generate 2 images.
   Expect a small consistent s/it reduction; samples visually identical.

## Revert Criteria
- Unit check diff > 1e-4 (would indicate an algebra mistake, not fp32 noise).
- Test failures or no measurable improvement.
