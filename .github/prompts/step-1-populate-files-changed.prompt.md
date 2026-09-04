---
agent: 'agent'
description: 'Stage 1 — inventory every kept optimization change into docs/optimization-documentation/files-changed/ with before/after records and emit manifest.json'
---

# Stage 1 — Populate `files-changed/`

You are auditing historical optimization work in this repository and turning it into a
structured, per-module record. This is **stage 1 of 4**. You are documenting, not optimizing.

Read `docs/optimization-documentation/README.md` first — it defines the ID namespace,
directory layout, and non-negotiable rules. That file is authoritative for naming.

## Hard rules

1. **Do not modify any `.py` file.** You write only under
   `docs/optimization-documentation/`.
2. **Do not run benchmarks, do not invent numbers.** Every metric you write must be copied
   verbatim from an existing doc, with the source file cited next to it.
3. **Do not invent before/after code.** An "after" snippet must be quotable from either an
   archive proposal file or the live source tree. If you cannot quote it, mark the change
   `UNVERIFIED` and continue — do not reconstruct code from memory.
4. **Use prefixed change IDs** (`K`, `C`, `X`) everywhere. Bare `#N` is forbidden in output
   files, because three colliding numbering schemes exist.

## Input inventory

### Krea2 track (`K`) — source: `docs/code-optimization/archive/krea2/set-1..set-5/`

| ID | Title | Module | Verdict |
|----|-------|--------|---------|
| K1 | VAE frame-dim: encode images individually | `krea2` | KEPT |
| K3 | Vectorize `pad_text_features` | `krea2.pipeline` | KEPT |
| K4 | Gradient checkpointing in `TextFusionBlock`/`TextFusionTransformer` | `krea2.mmdit` | KEPT |
| K5 | Timestep dtype: model dtype instead of fp32 | `krea2` | KEPT |
| K6 | Cache VAE `latents_mean`/`latents_std` at load | `krea2` | KEPT |
| K9 | Single dtype conversion per CFG step | `krea2.pipeline` | KEPT |
| K10 | Pre-fuse text context before the sampling loop | `krea2.pipeline` + `krea2.mmdit` | KEPT |
| K14 | Sequence padding `% 256` → `% 32` | `krea2.mmdit` | KEPT |
| K16 | Lean `ropeapply` — bf16, no fp32 round-trip | `krea2.mmdit` | KEPT |
| K18 | `_get_step_indices` reversal fix (batch > 1) | `toolkit.samplers.custom_flowmatch_sampler` | KEPT |
| K19 | `pad_text_features` ragged-caption crash fix | `krea2.pipeline` | KEPT |
| K20 | `to_device_if_needed` device-compare + dtype-skip | `sd_trainer.SDTrainer` | KEPT |
| K21 | `flip_x` `UnboundLocalError` + duplicate import | `toolkit.data_loader` | KEPT |

**Excluded — do not document:** K2 (`torch.compile`, reverted: Windows `OverflowError`),
K7 (+8.6% regression), K8, K11, K12, K13 (all reverted), K15, K17 (reverted / open design
question), and everything under "Audited and rejected" in `docs/code-optimization/change-progress.md`.

### Legacy Chroma track (`C`) — source: `docs/optimization/README.md`

| ID | Title | Module | Verdict |
|----|-------|--------|---------|
| C1 | Eliminate CPU→GPU copy in state-dict loading | `chroma.chroma_model` | COMPLETED |
| C2 | Remove redundant `.clone()` before CPU transfer | `jobs.process.BaseExtractProcess` | COMPLETED |
| C4 | Cache pipeline creation | `chroma.pipeline` | INCONCLUSIVE — keep & monitor |
| C5 | Use `torch.inference_mode()` | `toolkit.stable_diffusion_model` | ALREADY IMPLEMENTED |

**Excluded:** C3 (batched prompt encoding, reverted).

### Legacy Krea2 checklist (`X`) — source: `docs/optimization/implementation-checklist.md`

This file has its **own** `#1`–`#5` that collide with `K`. Most are REVERTED there.
Document **only** `X2` (latents dtype conversion in `predict_velocity`, marked COMPLETED),
and flag in `manifest.json` that `X2` may be the same change as `K9`. Do not merge them —
Stage 2 decides.

## Procedure

1. Read `docs/code-optimization/change-progress.md` and `docs/code-optimization/current-state.md`
   to confirm the kept list above still matches. If a listed change is recorded as reverted
   there, drop it and note the discrepancy in `manifest.json` under `"discrepancies"`.
2. For each kept change, open its archive proposal file
   (`archive/krea2/set-N/implementation-proposal-change-<n>.md`) and any sibling
   `results-change-<n>.md`, and extract: issue, mechanism, before code, after code,
   recorded metrics, verdict.
3. Group changes by module. Create one folder per module under `files-changed/` using the
   dotted module path from the README.
4. Write the files per the templates below.
5. Write `manifest.json`.

## Output templates

### `files-changed/<module-path>/change-summary.md`

```markdown
# <module path> — change summary

**Live file:** `<absolute repo-relative path>`
**Changes recorded:** <n> — <comma-separated prefixed IDs>
**Source docs:** <archive files cited>

## Overview
<2–5 sentences: what this module does in the Krea2/Chroma pipeline, and what class of
performance problem the recorded changes address (CPU↔GPU transfer, dtype round-trip,
redundant allocation, correctness bug, etc.).>

## Changes in this module

| ID | Title | Kind | Loop | Recorded impact | Verdict |
|----|-------|------|------|-----------------|---------|

`Kind` is one of: `perf` / `correctness` / `memory`. `Loop` is `train` / `sample` / `both`.
Cite the source file for every impact number.

## Interactions
<Note when two changes in this module touch the same function — e.g. K3 and K19 both
rewrite pad_text_features — so Stage 4 can merge them.>
```

### `files-changed/<module-path>/<ID>-<slug>.md`

````markdown
# <ID>: <title>

**Verdict (as recorded):** <KEPT / COMPLETED / ...>
**Complexity:** <Simple 1–5 / Moderate 6–10 / Complex 11–20 lines>
**Kind:** <perf / correctness / memory>  **Loop:** <train / sample / both>
**Live location:** `<path>` — `<function name>`
**Source:** `<archive file it came from>`

## Issue
<Why the change was made. Quote the original code as the "before".>

## Before
```python
<exact snippet from the proposal or git history>
```

## After
```python
<exact snippet currently recorded / currently in the tree>
```

## Mechanism
<Why this is faster or more correct. Keep the original doc's reasoning; do not add new
performance claims.>

## Recorded metrics
<Copy the table or numbers verbatim, with the source file named. If none exist, write
"No benchmark recorded.">

## Notes / risks
<Any caveat the original author recorded — numerics change, dormant under current config,
needs same-session control, etc.>
````

### `manifest.json`

```json
{
  "stage": 1,
  "generated": "<YYYY-MM-DD>",
  "branch": "<current branch>",
  "modules": {
    "krea2.mmdit": {
      "live_file": "extensions_built_in/diffusion_models/krea2/src/mmdit.py",
      "changes": ["K4", "K10", "K14", "K16"]
    }
  },
  "changes": {
    "K16": {
      "title": "Lean ropeapply - bf16 instead of fp32 round-trip",
      "module": "krea2.mmdit",
      "function": "ropeapply",
      "kind": "perf",
      "loop": "both",
      "verdict_recorded": "KEPT",
      "source": "docs/code-optimization/archive/krea2/set-4/implementation-proposal-change-16.md",
      "files": ["files-changed/krea2.mmdit/K16-lean-ropeapply.md"],
      "status": "documented"
    }
  },
  "excluded": ["K2", "K7", "K8", "K11", "K12", "K13", "K15", "K17", "C3"],
  "ambiguous": [],
  "discrepancies": []
}
```

Slug rule: lowercase, hyphenated, ≤ 4 words, derived from the title (`K16-lean-ropeapply.md`).

## Handoff to next step

When done, print a summary table: module → IDs → files written, plus the total count, and
confirm `manifest.json` validates as JSON and its counts match the folders on disk.
Stage 2 will read `manifest.json` and verify each "after" snippet against the live tree.
