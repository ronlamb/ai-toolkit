# Optimization Documentation Pipeline

A four-stage, auditable pipeline that converts the historical optimization work recorded in
`docs/code-optimization/` and `docs/optimization/` into a clean, ordered queue of proposed
changes that can be implemented one at a time in the base repository.

**Nothing in this tree is authoritative source code.** Every stage reads, cross-checks, and
documents. Source files are never modified by any stage. All benchmarks are run by the user.

---

## Directory layout

```
docs/optimization-documentation/
  README.md                        <- this file: rules, naming, step order
  manifest.json                  <- machine-readable handoff between stages
  files-changed/                 <- STAGE 1: per-module record of every kept change
    <module-path>/
      change-summary.md
      <ID>-<slug>.md             <- one file per change: before / after
  obsolete/                      <- STAGE 2: changes no longer present in live code
    <module-path>/
      <ID>-<slug>.md
      OBSOLETE.md                <- why it was quarantined
  performance-change-analysis/   <- STAGE 3: critique + CUDA/MPS/generic split per change
    <ID>-<slug>/
      summary.md
      change-analysis.md
      test_change_<ID>_<slug>.py <- orphan pytest test (NOT copied into tests/)
  proposed-changes/            <- STAGE 4: merged, dependency-ordered implementation units
    <ID>-<slug>/
      proposal.md
      dependencies.md
```

---

## CRITICAL: change-ID namespaces

The repository contains **three independent `#N` numbering schemes** that collide. A bare
`#4` is ambiguous. Every file in this pipeline uses a **prefixed ID** so provenance is
never lost.

| Prefix | Source of truth | Range | Notes |
|--------|-----------------|-------|-------|
| `K` | `docs/code-optimization/archive/krea2/set-1..set-5/` | `K1`–`K21` | Canonical Krea2 optimization track. Primary scope. |
| `C` | `docs/optimization/README.md` | `C1`–`C5` | Legacy Chroma track. |
| `X` | `docs/optimization/implementation-checklist.md` | `X1`–`X5` | A *separate* Krea2 checklist with its own `#1`–`#5`. |

Known collisions that must **never** be silently merged:

- `K2` = `torch.compile` on `predict_velocity` (**REVERTED**, Windows `OverflowError`)
  vs `X2` = latents dtype conversion in `predict_velocity` (**COMPLETED**). Different changes,
  same function, same bare number.
- `K1` = VAE frame-dim in `encode_images` vs `X1` = `non_blocking` in `pad_text_features`.
- `C1` = Chroma state-dict load vs `K1`/`X1`.

**Rule:** if a stage cannot resolve which scheme a bare `#N` belongs to, it records the
ambiguity in `manifest.json` under `"ambiguous"` and does not guess.

---

## Scope

**Included** — changes whose recorded verdict is KEPT / COMPLETED / IMPLEMENTED-and-retained:

- Krea2: `K1`, `K3`, `K4`, `K5`, `K6`, `K9`, `K10`, `K14`, `K16`, `K18`, `K19`, `K20`, `K21`
- Legacy Chroma: `C1`, `C2`, `C4`, `C5` (`C3` batched prompt encoding was reverted)
- `X2` only if Stage 2 confirms it is live and distinct from `K9`

**Excluded** — reverted / rejected / never-applied, never enters Stage 1:

- `K2`, `K7`, `K8`, `K11`, `K12`, `K13`, `K15`, `K17`
- `C3`
- `X1`, `X3`, `X4`, `X5` (all marked REVERTED in that checklist)
- Everything under "Audited and rejected" in `docs/code-optimization/change-progress.md`

Stage 2 may further remove any change whose code is no longer in the live tree.

---

## Stage order and prompts

Run strictly in order. Each stage consumes the previous stage's output and rewrites
`manifest.json`.

| Stage | Prompt | Reads | Writes |
|-------|--------|-------|--------|
| 1 | `.github/prompts/step-1-populate-files-changed.prompt.md` | archive + legacy docs | `files-changed/`, `manifest.json` |
| 2 | `.github/prompts/step-2-validate-changes.prompt.md` | `files-changed/` + **live source** | `obsolete/`, updated `manifest.json` |
| 3 | `.github/prompts/step-3-performance-analysis.prompt.md` | surviving `files-changed/` | `performance-change-analysis/` |
| 4 | `.github/prompts/step-4-proposed-changes.prompt.md` | `performance-change-analysis/` | `proposed-changes/` |

### Stage 1 — Populate `files-changed/`
One folder per **source module**, using the dotted module path so same-named files
(`pipeline.py` exists in both `krea2/src/` and `ideogram4/src/`) cannot collide:

`krea2`, `krea2.pipeline`, `krea2.mmdit`, `chroma`, `chroma.model`,
`toolkit.samplers.custom_flowmatch_sampler`, `toolkit.data_loader`,
`toolkit.stable_diffusion_model`, `sd_trainer.SDTrainer`,
`jobs.process.BaseExtractProcess`

### Stage 2 — Validate against live code
Docs are a **hint, not proof**. Each change's "after" snippet must be located in the live
file on the current branch. Not found, superseded, or restructured → move to `obsolete/`
with a written reason.

### Stage 3 — Analyze
Per surviving change: critique, merge opportunities, code improvements, and a hard split into
**CUDA-only / MPS-only / generic**. Tests are written into the analysis folder only.

### Stage 4 — Propose
Merge changes touching the same function into one implementation unit; mark each unit
`SELF-CONTAINED` or `DEPENDS-ON`.

Known merge clusters:

| Cluster | Members | Unit |
|---------|---------|------|
| `pad_text_features` | `K3` + `K19` | one unit |
| `encode_images` / `decode_latents` | `K1` + `K6` | one unit |
| `SingleStreamDiT.forward` | `K10` + `K14` | one unit (`K10` also spans `pipeline.py`) |
| standalone | `K4`, `K5`, `K16`, `K18`, `K20`, `K21`, `C1`, `C2`, `C4`, `C5` | one each |

---

## Non-negotiable rules

1. **No source edits.** Stages write only under `docs/optimization-documentation/` and
   `.github/prompts/`.
2. **No benchmarks.** The assistant never runs, simulates, or estimates measured results.
   Numbers are copied verbatim from existing docs and labelled with their source file.
3. **No fabricated before/after.** If an "after" snippet cannot be quoted from a real
   proposal file or the live source, the change is marked `UNVERIFIED` and skipped.
4. **Nothing enters `tests/`.** Stage-3 tests stay in their analysis folder with a
   `cp` command in the header.
5. **Preserve provenance.** Every documented change cites the archive file it came from.
6. **User owns decisions.** Ambiguous verdicts, borderline deltas, and namespace
   collisions are surfaced, never resolved unilaterally.

---

## Current baseline (copied from `docs/code-optimization/current-state.md`)

| Metric | Value |
|--------|-------|
| Training, short bench bottom-out | ~3.08–3.09 s/it |
| Samples, short bench (1024-mix) | ~64.7 s/img |
| Full-run bottom-out | 2.86 s/it |
| Current code stack | `K10 + K14 + K16 + K18 + K19 + K20 + K21` |

Machine state drifts ~5–8% across days; a same-session control is required whenever a
benchmark contradicts mechanism analysis.
