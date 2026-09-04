---
agent: 'agent'
description: 'Stage 3 — critique each surviving change, split it into CUDA-only / MPS-only / generic, find merge opportunities, and write an orphan pytest test per change'
---

# Stage 3 — Populate `performance-change-analysis/`

You are critically analyzing each surviving optimization so the user can implement them one at a
time in the base repository. This is **stage 3 of 4**. You are analyzing, not implementing.

Read `docs/optimization-documentation/README.md` first for the ID namespace and rules.

## Inputs

- `docs/optimization-documentation/manifest.json` — work only on changes with
  `"status"` of `valid` or `superseded`. Skip `obsolete` and `unverified`.
- `docs/optimization-documentation/files-changed/**` — the before/after records.
- Live source, for ground truth on what the code actually does today.
- `.github/skills/cuda-optimization/SKILL.md` and `.github/skills/mps-optimization/SKILL.md`
  as *reference* for platform patterns.

## Hard rules

1. **Do not modify any `.py` file in the project.** Tests you write go **only** into
   `docs/optimization-documentation/performance-change-analysis/<ID>-<slug>/`.
   **Never write into `tests/`.**
2. **No benchmarks, no invented numbers.** You may reason about mechanism and cite numbers
   already recorded in the docs (with the source file named). You may not claim a speedup you
   did not measure — and you cannot measure.
3. **Do not trust the skill docs' platform claims blindly.** They state MPS is "float32 only".
   That is version-dependent — bf16 support on MPS has landed in recent PyTorch. Before using
   any platform claim as the basis for a CUDA/MPS split, check the installed torch version
   (`Get-Content .venv\pyvenv.cfg`, or read `torch.__version__` if you run anything) and say
   which version your claim assumes. Flag the claim as `version-dependent`.
4. **Critique honestly.** Some of these changes were kept at neutral or within-noise impact,
   or kept "for cleanliness". Say so. A change that is structurally valid but performance-neutral
   should be labelled `LOW VALUE` rather than presented as a win.

## Procedure

For each surviving change, create `performance-change-analysis/<ID>-<slug>/` and write:

### 1. `summary.md`

```markdown
# <ID>: <title> — summary

**Module:** `<path>` · **Function:** `<name>` · **Kind:** perf/correctness/memory
**Loop:** train/sample/both · **Stage-2 status:** valid/superseded
**Dormant under user config:** yes/no (<which config knob>)
**Recorded impact:** <verbatim from docs, cited> · **Assessed value:** HIGH/MEDIUM/LOW

<3–6 sentences: what it does, why, and whether the recorded gain survives scrutiny.>
```

### 2. `change-analysis.md`

```markdown
# <ID>: <title> — analysis

## Critique
<Is the mechanism sound? Does the claimed gain follow from the code? Was the verdict justified
given the variance band (±1–2% bottom-out, ~5–8% day-to-day machine drift)? Was a same-session
control run when required? Is the change still needed after the upstream merge? If the change is
a correctness fix with no perf effect, say plainly that it should not be sold as an optimization.>

## Merge opportunities
<Which other changes touch this same function or the same data path, and what a merged version
would look like. Known clusters: K3+K19 (pad_text_features), K1+K6 (encode_images/decode_latents),
K10+K14 (SingleStreamDiT.forward). Note cross-module couplings too — K10 spans pipeline.py and
mmdit.py. State whether merging is SAFE (same lines) or RISKY (different numerics).>

## Code improvements
<Concrete, better ways to write this change — not new optimizations, refinements of THIS one.
e.g. caching on the right key, avoiding a hidden sync, hoisting out of a loop, using an existing
helper. Respect the repo rule: ≤20 changed lines per function, surgical edits, no rewrites.>

## Platform split

### CUDA-only
<Parts that only matter, or only work, on NVIDIA: bf16/fp8 dtype round-trips, cuDNN SDPA
backend choice, CUDA graphs, torch.compile, non_blocking pinned transfers, VRAM tiling.
If none, write "None — this change is platform-neutral.">

### MPS-only
<Parts that matter only on Apple Silicon: fp64 unsupported, 8-bit optimizers unsupported,
command-buffer sync from in-place requires_grad_, Metal kernel shape limits, per-image VAE loop
motivation. If none, write "None.">

### Generic (both)
<Parts that help on any device: algorithmic/vectorization wins, removed redundant allocation,
hoisted loop-invariant work, correctness fixes.>

**Split confidence:** high/medium/low — <why, and which torch version the split assumes>

## Risk & rollback
<What breaks, what numerics change, the one-line revert command
(`git checkout -- <path>`), and whether a fixed-seed visual check is needed.>

## Test strategy
<What the accompanying pytest proves, what it cannot prove, and what only a user benchmark can
answer.>
```

### 3. `test_change_<ID>_<slug>.py`

An orphan pytest test, **not** added to `tests/`. Header must tell the user exactly how to
install and run it:

```python
"""
Test for <ID>: <title>

NOT installed in tests/ by design. To use:

    Windows (PowerShell):
        Copy-Item docs/optimization-documentation/performance-change-analysis/<ID>-<slug>/test_change_<ID>_<slug>.py tests/
        .venv/Scripts/python.exe -m pytest tests/test_change_<ID>_<slug>.py -v

    Linux / macOS:
        cp docs/optimization-documentation/performance-change-analysis/<ID>-<slug>/test_change_<ID>_<slug>.py tests/
        .venv/bin/python -m pytest tests/test_change_<ID>_<slug>.py -v

Covers: <what is asserted>
Does not cover: <what needs a real benchmark>
"""
```

Test requirements:
- Import the real symbol from the real module path; do not reimplement the logic under test.
- Assert the **invariant** the change protects (shape, dtype, mask correctness, index↔query
  pairing, gradient flow, no-crash on ragged input), not a hardcoded timing.
- Never assert wall-clock performance.
- Skip gracefully when hardware is absent: `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`,
  and an MPS equivalent. CPU must always be able to run the logic-level assertions.
- Follow the conventions in `tests/test_torch_util.py` (module-level imports, `class TestX:`
  grouping, device-agnostic helpers).
- Keep it runnable in seconds, not minutes — no full model load.

## Assessment vocabulary (use exactly these)

| Label | Meaning |
|-------|---------|
| `HIGH VALUE` | Mechanism sound, recorded gain outside variance, still live |
| `MEDIUM VALUE` | Mechanism sound, gain within noise, kept for cumulative effect |
| `LOW VALUE` | Neutral / "kept for cleanliness" — implement only if free |
| `CORRECTNESS` | Not a perf change; justified by bug, not speed |
| `DORMANT` | Present but not exercised under the user's current config |
| `SUPERSEDED` | Upstream rewrote it; intent may still be re-appliable |

## Handoff to next step

Print a table: ID → assessed value → platform split (CUDA/MPS/generic counts) → merge cluster →
test file written. Stage 4 will merge same-function changes and order the queue, so be explicit
about which IDs must land together and which conflicts exist.
