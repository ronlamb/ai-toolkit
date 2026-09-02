# Optimization Documentation Skill

Generate standardized documentation for optimization changes, results tracking, and implementation checklists.

## File Organization

The `docs/code-optimization/` directory uses a 3-file structure:

| File | Purpose | Size |
|------|---------|------|
| `current-state.md` | **Current best metrics + pending work only.** Short bench best, full-run best, pending items, testing protocol. | ~100 lines |
| `change-progress.md` | **Historical record.** Progress across sets, kept/reverted changes, audited items, qualitative validation, open questions. | ~200 lines |
| `gpu-performance-modes.md` | **GPU state observation.** Bimodal performance modes, benchmarking implications, same-session control protocol. | ~80 lines |
| `archive/krea2/set-N/` | **Per-change details.** `implementation-proposal-change-N.md` (issue, proposal, validation, benchmark results). | one per change |

## Available Patterns

### 1. Change Proposal Template
Generate documentation for a new optimization change.

**Input**: Bottleneck description, location, current code, optimized code
**Output**: `archive/krea2/set-N/implementation-proposal-change-N.md`

**Template**:
```markdown
# Change #N: <title>

**Status**: PROPOSED / IMPLEMENTED / ✅ COMPLETED / ⚠️ REVERTED
**Complexity**: Simple (1-5 lines) / Moderate (6-10 lines) / Complex (11-20 lines)
**Expected Impact**: ~X% training, ~Y% sampling (or "neutral — correctness fix")
**Applies to**: training loop / sampling loop / both

## Issue

<description of the bottleneck or bug, with code snippet>

## Evidence

<measurements, micro-benchmarks, or crash repro>

## Proposed change

<code diff or replacement, ≤20 lines>

### What changed and why

<table explaining each part of the change>

## Validation plan

- Unit/equivalence check
- `pytest tests/` (44 passed baseline)
- Benchmark expectations
- Revert command

## Benchmark results (fill after user test)

<table with epoch-by-epoch data>

| Metric | Baseline | This change | Delta |
|--------|----------|-------------|-------|

**Verdict**: KEEP / REVERT / USER DECIDES
```

### 2. Results Tracking
Document test results for optimization changes.

**Input**: Baseline metrics, change results, analysis
**Output**: Append benchmark results section to the proposal file in `archive/krea2/set-N/`

**Standardized Template** (append to proposal file):
```markdown
## Benchmark results (tested YYYY-MM-DD)

Short bench, same dataset mix. Cumulative `s/it` at epoch end; per-step avg from total-time deltas.

| Epoch | Cum s/it | Per-step avg s/it | Samples avg (s/img) |
|-------|----------|-------------------|---------------------|

| Metric | Baseline | This change | Delta |
|--------|----------|-------------|-------|

**Verdict**: ✅ KEEP / ⚠️ REVERTED / 🤔 USER DECIDES
```

### 3. Implementation Checklist
Generate checklist for tracking optimization progress.

**Input**: List of changes with status
**Output**: Markdown checklist following implementation-checklist.md format

## When Recording Benchmark Results

After the user provides benchmark logs:

1. **Add benchmark results section** to the proposal file (append at end).
2. **Update `current-state.md`**:
   - If the change is KEPT and improves metrics → update "Current best metrics" table.
   - If the change is KEPT (neutral/correctness) → update "Current code stack" note.
   - If the change is REVERTED → no metric change needed.
3. **Update `change-progress.md`**:
   - Add the change to the "Kept changes" or "Reverted changes" table.
   - If a new open question arises, add to "Open questions" section.

## Key Rules

- **Never add benchmark tables to `current-state.md`** — they belong in the proposal file.
- **`current-state.md` should never exceed ~100 lines** — if it grows, move content to `change-progress.md`.
- **Compare bottom-out values**, not early-epoch numbers (warm-up drags cumulative average up).
- **Run same-session control** if a bench result disagrees with mechanism analysis (see `gpu-performance-modes.md`).
- **One change per session** — cleanest attribution.
- **User decides** on borderline results (±1–2% on bottom-out metric).

## Usage

Invoke this skill when you need to:
- Document a new optimization change
- Record test results after validation
- Generate progress checklists

## Reference

See also: `.github/optimization-workflow.md` for detailed protocols and decision rules.
