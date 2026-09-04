---
agent: 'agent'
description: 'Stage 4 — merge analyzed changes into implementation units, mark each SELF-CONTAINED or DEPENDS-ON, and produce an ordered work queue in proposed-changes/'
---

# Stage 4 — Populate `proposed-changes/`

This is the deliverable. Everything before this was preparation. The output is a queue of
implementation units the user works through **one at a time** in the base repository, each with
enough detail to implement without re-reading the archive. This is **stage 4 of 4**.

Read `docs/optimization-documentation/README.md` first for the ID namespace and rules.

## Inputs

- `docs/optimization-documentation/manifest.json` — statuses and merge hints from stages 1–2.
- `docs/optimization-documentation/performance-change-analysis/**` — `summary.md`,
  `change-analysis.md`, and the orphan test for every surviving change.
- Live source, to confirm the exact current lines each unit will touch.

## Hard rules

1. **Do not modify any `.py` file.** You write only under
   `docs/optimization-documentation/proposed-changes/`.
2. **≤20 changed lines per function, surgical edits only, no rewrites.** Any unit that would
   exceed this must be split, or marked `NEEDS DESIGN` and handed back to the user.
3. **No benchmarks, no invented numbers.** Carry forward recorded metrics with their source
   citations. Expected impact must be phrased as a mechanism, not a promise.
4. **Every unit must be independently revertable** with a single `git checkout -- <path>`.
   If two units share a file, they must still be separable commits — say so explicitly.
5. **Do not resurrect rejected work.** Nothing from `obsolete/`, and never re-propose the
   known-dead items: RoPE `omega`/freqs caching, removing padding entirely, CFG cond+uncond
   batching, SDPA backend preference list, `torch.compile` on `predict_velocity` (Windows
   `OverflowError`).

## Merging

Merge changes that touch the **same function** into one unit — the user should not edit
`pad_text_features` twice.

| Cluster | Members | Unit | Why |
|---------|---------|------|-----|
| `pad_text_features` | K3 + K19 | `P-pad-text-features` | Same function; K3 vectorized fast path, K19 added the ragged branch. One coherent function. |
| `encode_images` / `decode_latents` | K1 + K6 | `P-vae-encode-decode` | Same two functions; per-image loop + cached norm constants. |
| `SingleStreamDiT.forward` | K10 + K14 | `P-dit-forward` | Same function; `fused_context` threading + `% 32` padding. K10 also touches `pipeline.py`, so this unit spans two files — call that out. |

Merge only when stage 3 rated it `SAFE`. If stage 3 flagged `RISKY` (different numerics, e.g. a
dtype change fused with a structural change), **keep them separate** and note the reason.
Anything not in a cluster becomes its own unit.

## Ordering

Order the queue by: **correctness fixes first** (they change behavior and must be baseline-clean),
then **self-contained generic wins**, then **CUDA-specific**, then **MPS-specific**, then
**dependent units last**. Within a tier, prefer lowest-risk-highest-clarity.

Reason: a correctness fix applied after a perf change makes attribution impossible. The user's
protocol requires clean per-change attribution and same-session controls.

## Output

### `proposed-changes/<unit-slug>/proposal.md`

````markdown
# <unit-slug>: <title>

**Absorbs:** <IDs merged here, or single ID>
**Type:** SELF-CONTAINED | DEPENDS-ON | NEEDS DESIGN
**Files:** `<path>` (<n> functions) · **Est. changed lines:** <n> (≤20 per function)
**Loop:** train/sample/both · **Platform:** generic / CUDA-only / MPS-only
**Assessed value:** HIGH/MEDIUM/LOW/CORRECTNESS · **Dormant under current config:** yes/no

## What to change
<Exact, imperative. Name the function and quote the current live code, then give the target
code. The user must be able to implement this without opening any archive file.>

### Current (live)
```python
<quoted from the live tree, with file:line as of this audit>
```

### Target
```python
<the proposed result, ≤20 changed lines>
```

## Why
<Mechanism. Cite the source doc for any number.>

## Apply
```
<the exact edit, or a patch-style block>
```

## Verify
1. Unit test: `Copy-Item docs/optimization-documentation/performance-change-analysis/<ID>-<slug>/test_change_<ID>_<slug>.py tests/` then run pytest (Linux/macOS: `cp` + `.venv/bin/python -m pytest`).
2. Full suite: `.venv/Scripts/python.exe -m pytest tests/ -v` (record the pass count).
3. Benchmark — **user runs this**, never the assistant:
   short bench, 6 epochs × 30 steps, 4 images, same dataset mix.
   Compare **bottom-out** s/it and samples s/img against the current baseline
   (~3.08–3.09 s/it, ~64.7 s/img — see `docs/code-optimization/current-state.md`).
4. Keep / revert decision belongs to the user. ±1–2% → user decides. If a result contradicts
   the mechanism, run a same-session control before deciding.

## Revert
```
git checkout -- <path>
```

## Risks
<Numerics change? visual A/B needed? dormant code? VRAM impact?>
````

### `proposed-changes/<unit-slug>/dependencies.md`

```markdown
# Dependencies — <unit-slug>

**Type:** SELF-CONTAINED | DEPENDS-ON

## Depends on
| Unit | Reason | Blocking? |
|------|--------|-----------|
<Write "None — can be implemented on a clean tree." when self-contained.>

## Blocks
| Unit | Reason |
|------|--------|

## Shared files
<Other units touching the same file, and the required commit order to avoid conflicts.>

## Suggested queue position
<#N of M, and why here rather than elsewhere.>
```

### `proposed-changes/QUEUE.md`

The single page the user works from.

```markdown
# Implementation queue

| # | Unit | Absorbs | Type | Platform | Value | Files | Est. lines | Revert |
|---|------|---------|------|----------|-------|-------|-----------|--------|

## Read first
<3–8 bullets: the global constraints — ≤20 lines/function, user owns benchmarks, bottom-out
comparison, same-session control on contradiction, one unit per session for clean attribution.>

## Explicitly out of scope
<List the never-re-propose items with a one-line reason each.>
```

## Self-check before finishing

- Every `valid`/`superseded` change from `manifest.json` appears in exactly one unit — no
  change silently dropped, no change in two units.
- Every unit is ≤20 lines per function, or is marked `NEEDS DESIGN`.
- Every `DEPENDS-ON` names a unit that exists in the queue, and the dependency points backwards
  (earlier in the queue), never forwards.
- Every unit has a working `git checkout --` revert.
- `QUEUE.md` row count matches the number of unit folders.
- No `.py` file outside `performance-change-analysis/` was created or touched.

## Handoff

Print the queue table and state which unit you recommend starting with and why. From here the
user implements one unit per session, benchmarks it, and records the verdict back into
`docs/code-optimization/current-state.md` and `change-progress.md`.
