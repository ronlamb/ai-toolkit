# Change #18: Fix reversed sample↔index pairing in `_get_step_indices` (batch > 1 correctness bug)

**Status**: 📝 PROPOSED 2026-08-30 (approved for implementation in a separate session)
**Complexity**: Simple (~6 lines changed, single function, well under the 20-line limit)
**Impact**: Correctness fix. Prevents silent loss-weight corruption when timestep weighting is
enabled at batch_size > 1. **Dormant for current Krea2 configs** (see "Live status" below) — a
landmine removal, not a speed change. Expected benchmark result: exactly neutral.

## Issue — found during the main…krea_5 audit (2026-08-30)

`toolkit/samplers/custom_flowmatch_sampler.py`, `CustomFlowMatchEulerDiscreteScheduler._get_step_indices`
(new helper introduced on this branch; it replaced main's `(schedule_timesteps == t).nonzero().item()` loop):

```python
def _get_step_indices(self, timesteps: torch.Tensor) -> torch.Tensor:
    base = self.timesteps

    # ensure same dtype/device
    timesteps = timesteps.to(device=base.device, dtype=base.dtype)

    if base[0] > base[-1]:
        base = torch.flip(base, dims=[0])      # flip GRID ascending — correct
        flipped = True
    else:
        flipped = False

    if flipped:
        t = torch.flip(timesteps, dims=[0])    # BUG: flips the QUERIES too
    else:
        t = timesteps

    idx = torch.searchsorted(base, t)

    if flipped:
        idx = (len(self.timesteps) - 1) - idx  # compounds the reversal
    return idx
```

### Root cause — a false sorting assumption

`torch.searchsorted` requires only the **base** to be sorted; query order is irrelevant and
output position *i* must correspond to input position *i*. The code additionally flips the
queries, which makes output position *i* correspond to input position *N−1−i*. The subsequent
index inversion maps grid positions but never restores the query pairing.

The flip only "works" if the queries arrive in the same order as the grid. **Training timesteps
are random per sample** — `timestep_indices = torch.randint(...)`
(`jobs/process/BaseSDTrainProcess.py` ~L1320) → `self.sd.noise_scheduler.timesteps[timestep_indices]`.
They are never sorted. Batch-1 and all-equal queries are unaffected (flipping is a no-op), which
is why nothing visibly broke yet.

### Chroma precedent (user-reported, corroborates severity)

The same "assume queries are sorted" class of bug appeared during earlier chroma analysis:
sorting before lookup caused per-epoch samples to get progressively blurrier/noisier until they
degraded to random noise. Errors compound every step because each sample trains against another
sample's weight/sigma. It stays invisible at batch_size 1.

## Evidence (measured on this machine, RTX 4090)

Repro against main's exact-equality semantics:

| grid | queries | main (expected) | current branch | proposed fix |
|---|---|---|---|---|
| linear `[1000…1]` (1000 steps) | `[923, 456, 781]` | `[77, 544, 219]` | `[219, 544, 77]` ✗ reversed | `[77, 544, 219]` ✓ |
| sigmoid grid (non-integer values) | exact grid values at positions `[3, 17, 41]` | `[3, 17, 41]` | reversed | `[3, 17, 41]` ✓ |

(An earlier draft of the fix dropped the index inversion entirely — that was wrong on both grids.
The version below is the tested one.)

## Live status under current configs (blast-radius trace)

| Caller | Gate | Current Krea2 config (`anna_bell_sex_krea_ut`) | Status |
|---|---|---|---|
| `get_weights_for_timesteps` (`SDTrainer.calculate_loss` ~L889) | `linear_timesteps`/`linear_timesteps2` true **or** `timestep_type == "weighted"` | all false / `"linear"` | never called → dormant |
| scheduler `get_sigmas` | only reachable via `precondition_model_outputs_flow_match` (`toolkit/train_tools.py` ~L768) | function imported in SDTrainer but **never invoked anywhere** | dead path |
| `BaseSDTrainProcess.get_sigmas` (~L956) | main's equality loop, untouched by this branch | — | unaffected |

So the bug is **latent**: zero effect on past/current runs (batch_size 1 + linear timesteps), but it
silently corrupts any future run with `linear_timesteps: true` or `timestep_type: weighted` at
batch > 1 — per the chroma experience, that corruption ruins a run over many epochs.

## Proposed change (minimal — correctness only)

Replace the body of `_get_step_indices`:

```python
def _get_step_indices(self, timesteps: torch.Tensor) -> torch.Tensor:
    base = self.timesteps

    # ensure same dtype/device
    timesteps = timesteps.to(device=base.device, dtype=base.dtype)

    if base[0] > base[-1]:
        # searchsorted requires an ascending base. Invert the resulting indices
        # back to descending-grid positions. The queries themselves are NOT
        # sorted (timesteps are sampled randomly per sample during training),
        # so they must never be reordered — output position i must match input
        # position i.
        return (len(base) - 1) - torch.searchsorted(torch.flip(base, dims=[0]), timesteps)

    return torch.searchsorted(base, timesteps)
```

Removed: `flipped` flag, query flip, separate index-inversion block. Net 18 → 12 lines.
No API change; callers unchanged.

## Explicitly NOT in this change

- **Cache-thrash** in `get_weights_for_timesteps` (`_cached_device != timesteps.device` compares
  unindexed `cuda` vs `cuda:0`, so the 3 weight tensors are re-copied every call) → separate task,
  to be proposed as **#19**. Kept out per user decision ("keep A strictly minimal").
- The dead import `precondition_model_outputs_flow_match` in SDTrainer — audit note only.

## Validation plan

1. **Unit repro** (before/after): the two table rows above must match "expected" after the fix;
   batch-1 and all-equal cases must remain unchanged.
2. `pytest tests/` → expect 44 passed (no test currently covers this function).
3. **Benchmark per protocol anyway** (user rule: non-performance changes are benchmarked too):
   short bench 6 epochs × 30 steps, 4 images. Expect **exactly neutral** vs current best
   (bottom-out 3.09 s/it, samples 64.7 s/img) because the function is never called under the
   benchmark config — this run is a no-regression confirmation only.

## Results

*(pending implementation session)*
