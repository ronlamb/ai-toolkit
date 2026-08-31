# Change #19: Fix ragged-caption crash in Krea2 `pad_text_features` (keep vectorized fast path)

**Status**: 📝 PROPOSED 2026-08-30 (approved for implementation in a separate session)
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

*(pending implementation session)*
