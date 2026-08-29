# Results — Change #14: Re-align sequence padding (% 256 → % 32)

**Status**: ✅ KEPT (variant B) — new short-benchmark best as of this change.

## Test Configuration
- Short benchmark: 6 epochs × 30 steps, 4 images (same dataset mix as all Set-2/3/4 short tests).
- Baseline for comparison: change #10 state, short benchmark bottom-out **~3.22 s/it**
  (epochs 4–6: 3.23 / 3.22 / 3.22), samples **~67.2 s/img** (epochs 4–6 avg).

> **Baseline calibration note**: the **3.02 s/it** figure quoted in earlier docs comes from a
> longer (~179+ step) run; it is not comparable to the 6×30 short benchmark. Use ~3.22 s/it for
> short-benchmark comparisons.

## Variant A — remove padding entirely: ❌ REVERTED (regression)

Per-epoch deltas: 4.43 / 4.20 / 3.83 / 3.60 / 3.67 / **3.80 s/it** — plateaued ~3.6–3.8 instead
of bottoming out (~+10–17% vs the 3.22 short-benchmark baseline). Samples 69.8 → 66.2 s/img,
improving epoch-over-epoch while training stalled: consistent with a backward-only slowdown at
unaligned (odd) sequence lengths.

Micro-benchmarks (`attention()` fwd+bwd, real head counts 48q/12kv, d=128, bf16, RTX 4090,
torch 2.9.1 — per-token ns):

| pad target | short bucket L=1071 | long bucket L=4141 |
|---|---|---|
| odd (variant A) | 1.171 | **3.725** ← slow |
| % 16 | 1.193 | 3.467 |
| **% 32** | **1.098** | **3.177** ← best |
| % 64 | 1.119 | 3.162 (tie) |
| % 256 (old code) | 1.246 | 3.289 |

Odd lengths hurt the attention backward at long sequences; `% 256` over-pads up to 8× more tokens
than needed. `% 32` wins at both buckets — better than old code AND better than no padding.

## Variant B — pad to `% 32`: ✅ KEPT

`_padlen = (-combined.shape[1]) % 32` (was `% 256`) in `SingleStreamDiT.forward`. ≤31 pad tokens
instead of ≤255, aligned enough for cuDNN's fast kernels.

### Benchmark (6 epochs × 30 steps, 4 images; cumulative s/it at epoch end)

| Epoch | Cumulative s/it | Avg sample (s) |
|-------|-----------------|----------------|
| 1 | 3.23 | 66.31 |
| 2 | **2.96** | 65.55 |
| 3 | 3.15 | 66.08 |
| 4 | 3.22 | 65.50 |
| 5 | 3.17 | 65.60 |
| 6 | 3.15 (final 3.16) | 66.23 |

### Comparison vs baseline (#10 state, short benchmark)

| Metric | Baseline | Variant B | Delta |
|--------|----------|-----------|-------|
| Training epochs 4-6 avg (s/it) | ~3.22–3.26 | 3.18 | −1.2% to −2.5% |
| Final cumulative (epoch 6, s/it) | 3.22 | 3.15–3.16 | −1.9% |
| Samples epochs 4-6 avg (s/img) | ~67.2 | 65.8 | −2.1% |

## Verification
- `pytest tests/`: 44 passed.
- Numerical equivalence vs pre-change code (old loaded from git, same weights): **bitwise-identical
  outputs (max|diff|=0)** at L=1071/4141/4608; tiny seq L=36 differs ≤1 bf16 ulp (kernel-shape
  reduction order — expected, below bf16 noise).

## Verdict
Small but consistent improvement on both metrics (~−2%), below the 5% threshold — **kept by user
decision**. Sample times notably stable across all epochs (65.5–66.3) vs baseline drift to 69.
Mechanistically strictly less padding work than old `% 256` code; micro-benchmarks favor `% 32`
at both buckets.

**Details**: See `../../implementation-proposal-change-14.md`.
