# Change #20: Fix `to_device_if_needed` — device compare never matches on CUDA; dtype silently skipped

**Status**: IMPLEMENTED + BENCHMARKED 2026-09-01 (branch `krea_5`) — no regression. KEEP
**Complexity**: Simple (~15 lines, one module-level function + call sites unchanged)
**Impact**: Restores the function's stated purpose ("only transfer if needed") on CUDA/MPS, and
fixes a silent dtype-skip. **Expected neutral-to-neutral+** on the benchmark: `.to()` is already a
cheap no-op internally, so this removes wasted Python work rather than GPU copies. Dormant-but-real
correctness edge for `PromptEmbeds` objects.

## Issue — found during the main…krea_5 audit (2026-08-30)

`extensions_built_in/sd_trainer/SDTrainer.py`, module-level `to_device_if_needed` (~lines 50–81),
introduced by the MPS performance work (`docs/optimization/mac-performance-improvements-findings.md`
item B3: *"One `.device` comparison … avoids unnecessary copies when tensor is already on correct
device"*). It has **~59 call sites** in SDTrainer.

### Before — current function (verbatim)

```python
def to_device_if_needed(tensor: Union[torch.Tensor, 'PromptEmbeds'], device: torch.device, dtype: torch.dtype = None) -> Union[torch.Tensor, 'PromptEmbeds']:
    """
    Only transfer tensor to device if it's not already there, avoiding unnecessary copies.
    This is critical for MPS memory fragmentation prevention.
    ...
    """
    # Handle PromptEmbeds objects
    if hasattr(tensor, 'to') and not isinstance(tensor, torch.Tensor):
        # Check if already on correct device
        embed_device = tensor.text_embeds[0].device if isinstance(tensor.text_embeds, list) else tensor.text_embeds.device
        if embed_device == device:
            # Already on correct device, just return as-is (PromptEmbeds.to() returns self)
            return tensor                      # <-- BUG 2: dtype argument silently ignored
        else:
            # Need to transfer to new device
            return tensor.to(device, dtype=dtype) if dtype is not None else tensor.to(device)

    # Handle regular tensors
    if tensor.device != device:                # <-- BUG 1: always True on CUDA
        if dtype is not None:
            return tensor.to(device, dtype=dtype)
        else:
            return tensor.to(device)
    elif dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype=dtype)
    return tensor
```

### Bug 1 — the device comparison never matches on CUDA (or MPS)

Verified on this machine (RTX 4090):

```
Accelerator().device   = device(type='cuda')           # index is None
tensor.device          = device(type='cuda', index=0)  # index is 0
tensor.device == Accelerator().device                  -> False
```

`torch.device('cuda') != torch.device('cuda:0')` — PyTorch compares `(type, index)` tuples and does
**not** resolve `None` to the current device. The comparison only succeeds when *both* sides carry
an index (or both are unindexed).

Where do the targets come from? `BaseSDTrainProcess.__init__` (~L103):
`self.device_torch = self.accelerator.device` → **unindexed** `cuda`. So every call site passing
`self.device_torch` / `self.sd.device_torch` (the large majority) makes the guard always false:

| Consequence | Detail |
|---|---|
| Purpose defeated | the "only transfer if needed" branch is never taken → always calls `.to()` |
| Perf cost, small | measured: 1000 × `big.to(device=unindexed_cuda)` = **0.4 ms** — `.to()` short-circuits internally and returns *the same object* (`is` → True). So the real waste is ~59 extra Python calls/step, not GPU copies. |
| Correctness | none for tensors — `.to()` with matching device+dtype is a no-op returning self |

Same class of mismatch applies on MPS (`mps` vs `mps:0`), which is the platform this function was
written *for* — so even its original goal was not achieved.

### Bug 2 — dtype silently skipped for `PromptEmbeds` when devices do match

When the comparison *does* succeed (indexed targets, e.g. call sites at L631/L771/L832/L2048 that
pass `noise.device`, `target.device`, `original_samples.device`), the PromptEmbeds branch returns
early and **drops the requested dtype**. Verified:

```python
pe = PromptEmbeds(torch.randn(3, 8, device='cuda', dtype=torch.float32))   # cuda:0, fp32
out = to_device_if_needed(pe, torch.device('cuda:0'), dtype=torch.bfloat16)
out.text_embeds.dtype   # -> torch.float32   (bf16 requested, silently skipped)
```

The tensor branch has no such hole (it falls through to the `elif dtype is not None …` cast). So a
`PromptEmbeds` arriving on-device in fp32 stays fp32 while every other input becomes bf16 — a silent
dtype mismatch fed into the model. Whether it currently bites depends on embed dtypes at those call
sites; it is latent, not observed in training today.

### Related asymmetry noted (not fixed here)

`PromptEmbeds.to()` **mutates in place** (`toolkit/prompt_utils.py` L40–52 assigns to `self.*` and
returns self). So the transfer branch modifies the caller's object, while the early-return branch
does not. Verified: a non-matching-device call casts *and* mutates the source embeds. Most call
sites pass `.clone().detach()` first (safe); the ones that don't (`conditional_embeds`,
`unconditional_embeds`, `self.diff_output_preservation_embeds`) receive freshly-encoded objects, so
no shared state is corrupted today. Out of scope for this change — recorded as an audit note.

## Proposed change (minimal, call sites untouched)

Normalize the device comparison, and make the PromptEmbeds branch honor `dtype`:

```python
def _devices_match(tensor_device: torch.device, target: torch.device) -> bool:
    """True if both refer to the same physical device, resolving unindexed devices
    (``cuda`` == ``cuda:<current>``, ``mps`` == ``mps:0``). ``torch.device`` equality
    does not do this: ``cuda`` != ``cuda:0``."""
    if tensor_device == target:
        return True
    if tensor_device.type != target.type:
        return False
    if target.index is None or tensor_device.index is None:
        # resolve the unindexed side to its current/default device
        if target.type == 'cuda':
            current = torch.cuda.current_device()
        elif target.type == 'mps':
            current = 0
        else:
            current = None
        return (tensor_device.index in (None, current)
                and target.index in (None, current))
    return False


def to_device_if_needed(tensor, device, dtype=None):
    if hasattr(tensor, 'to') and not isinstance(tensor, torch.Tensor):
        # PromptEmbeds: compare on the first stored tensor's device
        embed_device = (tensor.text_embeds[0].device if isinstance(tensor.text_embeds, list)
                        else tensor.text_embeds.device)
        if _devices_match(embed_device, device):
            # already resident; still honour a requested dtype cast
            return tensor.to(dtype=dtype) if dtype is not None else tensor
        return tensor.to(device, dtype=dtype) if dtype is not None else tensor.to(device)

    if _devices_match(tensor.device, device) and (dtype is None or tensor.dtype == dtype):
        return tensor
    return tensor.to(device, dtype=dtype) if dtype is not None else tensor.to(device)
```

Kept: signature, docstring intent, PromptEmbeds duck-typing (`hasattr(tensor, 'to')`), and the
"return same object when nothing to do" behavior. Removed: the two-level `if/elif` tensor block in
favour of one condition. Net ≈ +12 lines including the helper — still a single-function change.

## Validation performed (2026-08-30, CUDA)

| Case | Current code | Proposed |
|---|---|---|
| tensor fp32 on `cuda:0`, target unindexed `cuda` + bf16 | casts (guard false → `.to()`) | casts; **source unchanged** ✓ |
| tensor already exact match (`cuda:0`, fp32) | returns self via `.to()` | returns self directly ✓ |
| PromptEmbeds fp32 on `cuda:0`, target indexed + bf16 | **fp32 returned — cast skipped** ✗ | **bf16** ✓ |
| PromptEmbeds fp32, target unindexed + bf16 | casts (guard false) | casts ✓ |
| CPU tensor → cuda target | copies to `cuda:0` ✓ | copies to `cuda:0` ✓ |

## Validation plan (implementation session)

1. Re-run the table above on CUDA tensors, plus an MPS check if available.
2. `pytest tests/` → expect 44 passed (no existing test covers this function).
3. **Benchmark per protocol** (non-performance changes are benchmarked too): short bench
   6 epochs × 30 steps, 4 images vs current best (bottom-out 3.09 s/it, samples 64.7 s/img).
   Expect neutral; the function's savings were never real on CUDA, so treat any gain as noise.

## Results

### Implementation (2026-09-01, branch `krea_5`)

Applied the proposed change **verbatim** in `extensions_built_in/sd_trainer/SDTrainer.py`:
added module-level `_devices_match` helper + rewired both branches of
`to_device_if_needed`. Net **+21 lines** (−24/+45), single function + one new helper;
all ~59 call sites untouched.

### Validation (2026-09-01, CUDA / RTX 4090, `.venv`) — ALL PASS

Repro harness run against the **fixed** code, covering every row of the validation table
above plus extras:

| Case | Result |
|---|---|
| tensor fp32 on `cuda:0`, target unindexed `cuda` + bf16 → casts; **source unchanged** | PASS |
| tensor exact match (`cuda:0` fp32, indexed target) → returns self directly (no `.to()` call) | PASS |
| PromptEmbeds fp32 on `cuda:0`, target indexed + bf16 → **bf16 honoured** (the old silent-skip) | PASS |
| PromptEmbeds fp32, target unindexed + bf16 → casts | PASS |
| CPU tensor → cuda target → copies to `cuda:0` | PASS |
| PromptEmbeds no-dtype, matching device → returns same object, no mutation | PASS |
| `_devices_match`: `cuda:0`≡`cuda` ✓, `cuda:1`≢`cuda` ✓, identical ✓, cross-type ✗ | PASS |

- `pytest tests/` → **44 passed**.
- No existing test covers this function (as predicted in the validation plan).

### Benchmark (2026-09-01, short bench: 6 epochs × 30 steps, 4 images)

| Epoch | cum s/it @ epoch end | incremental s/it | samples s/img (×4) | samples avg |
|-------|----------------------|------------------|--------------------|-------------|
| 1 | 3.87 | 3.73 | 62.74 / 63.47 / 64.88 / 64.41 | 63.9 |
| 2 | 3.50 | 3.13 | 62.41 / 62.61 / 62.65 / 62.79 | 62.6 |
| 3 | 3.35 | 3.07 | 65.47 / 66.77 / 66.92 / 67.01 | 66.5 |
| 4 | 3.27 | 3.03 | 67.49 / 67.30 / 67.24 / 67.34 | 67.3 |
| 5 | 3.14 | 2.60 | 67.01 / 67.39 / 67.28 / 67.31 | 67.2 |
| 6 | **3.08** | 2.80 | 67.65 / 67.72 / 67.74 / 67.83 | 67.7 |

Bottom-out cumulative: **3.08 s/it** @ step 179 — lowest value ever recorded on this bench
(prior best 3.09 = #16; #18 ≈3.12; #19 3.22; same-session control pre-#19 3.36). Samples:
overall avg **≈65.9 s/img**, epochs 4–6 avg ≈67.4, min per-image 62.4 — inside the documented
64–70 band and better than both #19 runs (≈68.4) and the control (≈69.2).

**Verdict: neutral, as predicted.** The +0.3% bottom-out delta vs the historical best is noise;
today's machine state sits in a faster regime than the #19 sessions anyway (their same-code
control was 3.36). Bench agrees with mechanism analysis → no same-session control needed
(Testing Protocol #5 not triggered).

**Status: IMPLEMENTED — benchmarked, no regression. KEEP** (correctness fix; speed delta
negligible — revert only if you object to the +21-line helper on other grounds).
