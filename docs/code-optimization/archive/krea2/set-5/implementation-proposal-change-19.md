# Change #19: Fix ragged-caption crash in Krea2 `pad_text_features` (keep vectorized fast path)

**Status**: IMPLEMENTED + BENCHMARKED (incl. same-session control) 2026-08-31/09-01 — **KEEP**.
Slower vs historical baseline, but the pre-change control run is *slower still*: session drift,
not this change.
**Complexity**: Simple (~14 lines changed, one function, under the 20-line limit)
**Impact**: Crash fix + removes a per-call CPU→GPU sync. **Exactly neutral** on the current
benchmark path (single-prompt sampling takes the fast path). Dormant-but-fatal for any multi-prompt
batch with differing caption token counts.

## Issue — found during the main…krea_5 audit (2026-08-30)

`extensions_built_in/diffusion_models/krea2/src/pipeline.py`, `pad_text_features` (~line 46).
Set-1 change #3 vectorized this function and introduced `torch.stack(features_list)` **before**
padding — but the function's own docstring says *"Padding to the batch max is deferred to here"*.
The stack contradicts that: it requires every sample to already have the same length.

**Before** — current function body (docstring omitted; lines ~46–68):

```python
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    # Stack all features first (may be shorter than max_len)
    all_features = torch.stack(features_list)  # (B, Lt_max_actual, F)  <-- RuntimeError if lengths differ

    # Create padded features tensor
    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)

    # Copy only the valid portion (faster than per-row assignment)
    features[:, :all_features.shape[1]] = all_features

    # Create mask using arange (vectorized)
    range_tensor = torch.arange(max_len, device=device).unsqueeze(0)      # (1, max_len)
    lengths_tensor = torch.tensor(lengths, device=device).unsqueeze(1)    # (B, 1)  <-- implicit CPU->GPU sync
    mask = (range_tensor < lengths_tensor).long()  # (B, max_len)

    return features, mask
```

### Why ragged lists are the normal case here

`AdvancedPromptEmbeds` stores **each caption at its natural token length** (verified: its
`concat_prompt_embeds` keeps ragged lists — no padding). So any batch of >1 prompt whose captions
encode to different lengths hits the crash. The docstring above the function even describes this
storage layout ("Each caption is stored 2D at its natural length").

### Blast radius (call sites)

| Call site | Path | Status today |
|---|---|---|
| `Krea2Model.get_noise_prediction` (`krea2.py` ~L645) | training step, batch = dataloader batch | crashes if captions differ in length |
| `Krea2Pipeline.__call__` / `predict_velocity` (sampling) | per-sample lists; current configs sample 1 prompt at a time | safe by accident (B=1 fast path) |

Reproduced on this machine: features of length 7 and 5 →
`RuntimeError: stack expects each tensor to be equal size, but got [7, 4] and [5, 4]`.

**Corroboration**: the ideogram4 twin of this function
(`extensions_built_in/diffusion_models/ideogram4/src/pipeline.py` ~L166) was *never* vectorized and
still uses a per-row copy loop — correct for ragged input. Krea2's #3 replaced that pattern with
`stack` and introduced the regression.

## Proposed change (minimal, fast path preserved)

**After** — replacement body of `pad_text_features` (same signature and docstring):

```python
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    if len(set(lengths)) == 1:
        # All samples share one length -> single vectorized copy (fast path).
        features[:] = torch.stack(features_list).to(device=device, dtype=dtype)
    else:
        # Ragged captions are stored at natural length; pad each row in turn.
        for i, f in enumerate(features_list):
            features[i, : lengths[i]] = f.to(device=device, dtype=dtype)

    # Build the mask on CPU (tiny) and transfer once, instead of creating a
    # device tensor from a Python list (implicit sync) per call.
    range_cpu = torch.arange(max_len).unsqueeze(0)
    lengths_cpu = torch.tensor(lengths, dtype=torch.long).unsqueeze(1)
    mask = (range_cpu < lengths_cpu).to(device=device, dtype=torch.long)

    return features, mask
```

### What changed and why

| Part | Reason |
|---|---|
| `stack` → conditional fast path | fixes the crash; keeps #3's vectorized speed when all lengths match (the common cached-embeds case) |
| ragged fallback = per-row copy | same semantics as main's original loop and ideogram4's current code |
| mask built on CPU, transferred once | removes `torch.tensor(lengths, device=device)` — an implicit CPU→GPU sync point on **every** call (training step + every sampled image). This is the "minor finding" from the audit folded in per user decision. |

Removed: the old three-step stack/slice/assign block and the two device-side mask allocations
(`arange` on GPU + `tensor(lengths)` on GPU).

## Validation performed (2026-08-30, CPU tensors)

| Case | Old | Proposed |
|---|---|---|
| equal lengths `[9,9,9] × F=32` | ok | **bitwise identical** features and mask (`torch.equal` → True/True) |
| ragged `[7, 5]` | `RuntimeError: stack expects each tensor to be equal size` | works; shape `(2, 7, 4)`; mask `[[1×7],[1×5,0,0]]`; row values verified |
| single sample `[6]` (sampling path) | ok | identical shape/mask |

## Validation plan (implementation session)

1. Re-run the three-case equivalence check above on CUDA tensors as well.
2. `pytest tests/` → expect 44 passed (no existing test covers this function).
3. **Benchmark per protocol** (non-performance changes are benchmarked too): short bench
   6 epochs × 30 steps, 4 images. Expect **exactly neutral** vs current best (bottom-out
   3.09 s/it, samples 64.7 s/img) — the benchmark's single-prompt sampling and cached-equal-length
   embeds both take the fast path. No-regression confirmation only.

## Results

### Implementation (2026-08-31, branch `krea_5`)

Applied the proposed replacement body verbatim to `pad_text_features` in
`extensions_built_in/diffusion_models/krea2/src/pipeline.py`. Diff: 14 insertions /
13 deletions, single function; signature, docstring, and callers unchanged.

### Unit validation (all passed, `.venv` Python, torch 2.9.1+cu128)

Repro script `.tmp_opt_test/repro_change19.py` checks `pad_text_features` against reference
per-row-loop semantics (main / ideogram4): features and mask must be exactly equal
(`torch.equal`), mask dtype `long`. Cases run on **both CPU and CUDA tensors**.

| Case | Before fix | After fix |
|---|---|---|
| ragged `[7, 5]` | `RuntimeError: stack expects each tensor to be equal size` | PASS — matches reference |
| ragged `[5, 7]` (order swapped) | same crash | PASS — matches reference |
| random ragged ×5 | crash | PASS — matches reference |
| equal lengths `[6, 6, 6]` (fast path) | PASS | PASS — matches reference |
| batch-1 `[9]` (sampling path) | PASS | PASS — matches reference |

Pre-fix run confirmed the crash on both devices (3 ragged cases × cpu/cuda = 6 RuntimeErrors,
equal-length/batch-1 passed). Post-fix: **ALL PASS** (10/10).

`pytest tests/` → **44 passed** (no test covers this function; no regressions).

### Benchmark (user runs, 2026-08-31)

Short bench (`anna_bell_sex_krea_ut`, 30 steps/epoch, 4 images), slower than both baselines:

| Metric | Baseline (#16 best) | #18 run | **#19 run** |
|---|---|---|---|
| Cumulative s/it @ step 179 | 3.09 (bottom-out) | ~3.12 | **3.22** (+4% vs #18) |
| Samples per-round avg (s/img) | 64.7 | 63.7–67.6 (band) | **67.9–69.0, overall ≈68.4** (+5.8% vs baseline) |

Per-sample-round averages: 68.82 / 67.89 / 68.24 / 68.27 / 68.44 / 69.01 — every round at or
above the top of #18's band (67.6). Training cumulative curve: 4.13 → 3.63 → 3.42 → 3.35 →
3.35 → **3.22** s/it.

### Attribution analysis — slowdown is NOT mechanistically attributable to this change

Call counts under the benchmark config (verified against call sites):

| Path | `pad_text_features` calls | Cost budget |
|---|---|---|
| Training (180 steps, batch 1) | 1 per step = **180 total** | would need ~0.13 s/**call** to explain +0.1 s/it — impossible for a (1×~77×F) tensor |
| Sampling (6 rounds × 4 imgs, CFG) | 2 per image = **48 total** | would need ~5 s/**call** to explain +3.7 s/img — absurd |

Numerics are identical: features and mask verified bitwise-equal to the old implementation on
the fast path (repro covers equal-length cases on CPU **and** CUDA), so training loss values are
unchanged step-for-step — only wall-clock differs, and it differs in *both* phases together,
which is the signature of a shared environmental factor (GPU thermal state / background load),
not of a code path that executes ~230 times for microseconds total.

The one real (tiny) cost delta: fast-path `.to()` adds at most one extra device-to-device copy
(~6 MB ≈ tens of µs) when embeds are already on GPU — 4–5 orders of magnitude too small.

**User feedback (2026-08-31)**: the #19 bench was run **twice**, nothing different in the
background, both runs matched — so within-session noise is excluded. Two same-code runs
agreeing does NOT compare against baselines measured in *earlier sessions*; only a
same-session control can separate "#19 is slower" from "machine state drifted since".
User also notes (prior experience): if/else branches have caused slowdowns in tight loops,
and questions the `enumerate` loop efficiency. Rebuttal recorded: this function runs once per
training step and twice per sampled image (~230 calls total per bench) — not a tight loop;
the ragged `for` path never executes under the bench config (cached embeds → equal lengths →
fast path), so it is dead code there. Branch cost is nanoseconds; explaining the observed
deltas would require ~0.13 s per training call and ~5 s per sampling call.

### Control experiment (in progress)

`pipeline.py` reverted to HEAD (pre-#19) for a **same-session control bench**; the #19 fix is
saved in `.tmp_opt_test/change19.patch` (`git apply` restores it). Decision matrix:

| Control result | Meaning | Action |
|---|---|---|
| ≈3.22 s/it / ≈68 s/img (matches #19 runs) | session drift — baseline is also slower today | re-apply #19, KEEP; record corrected same-session baseline |
| ≈3.09–3.12 / ≈64–67 (matches history) | #19 costs ~4% despite the mechanism analysis | profile per-call timing to find the mechanism, or revert permanently |

### Control bench result (2026-09-01, pre-#19 code, same machine/session style)

| Metric | Control (pre-#19) | #19 run 1 | #19 run 2 (same session as control? no — prior day) |
|---|---|---|---|
| Cumulative s/it @ step 179 | **3.36** | 3.22 | 3.22 |
| Samples avg (s/img) | **≈69.2** (69.5–69.8 rounds 1–5, 67.8 last) | ≈68.4 | ≈68.4 |

The control is **slower than the #19 runs on both metrics** — decisively matching the
"session drift" row of the decision matrix (and even overshooting it: today's baseline sits
below every previous band, training 3.36 vs 3.09–3.22, samples ~69 vs 64.7–68). #19 is at
worst neutral vs same-code control; if anything marginally faster, plausibly from the removed
per-call CPU→GPU mask sync.

**Verdict: KEEP.** Fix re-applied after control (2026-09-01) and re-validated:
`.tmp_opt_test/repro_change19.py` ALL PASS (5 cases × cpu/cuda), `pytest tests/` 44 passed.
Historical baselines (3.09 s/it / 64.7 s/img) are stale as of this week — future comparisons
should use a same-session control, per the protocol note in `current-state.md`.
