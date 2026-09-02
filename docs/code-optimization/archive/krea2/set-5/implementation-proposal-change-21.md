# Change #21: Fix `data_loader.py` flip_x crash (`UnboundLocalError`) + duplicate import

**Status**: IMPLEMENTED + BENCHMARKED 2026-09-02 (branch `krea_5`) — no regression. KEEP
**Complexity**: Trivial (2 lines: restore 1 assignment, delete 1 duplicate import)
**Impact**: **Hard crash fix.** Any dataset configured with `flip_x: true` fails during dataset
setup with `UnboundLocalError: cannot access local variable 'new_file_item'`. Dormant under the
current Krea 2 config (`flip_x: false` in `.job_config.json`), so neutral for the active benchmark —
but this is a latent crash for every user who enables horizontal-flip augmentation.

## Issue — found during the main…krea_5 audit (2026-08-30)

`toolkit/data_loader.py`, dataset constructor — **class is `AiToolkitDataset`** (`__init__` starts
L388; this doc originally said `LoRADataset`, corrected 2026-09-01), flip_x block at L565–571.

### Before — current code (verbatim, L565–581)

```python
        # handle x axis flips
        if self.dataset_config.flip_x:
            print_acc("  -  adding x axis flips")
            current_file_list = [x for x in self.file_list]
            for file_item in current_file_list:
                # create a copy copy.deepcopy(file_item)          # <-- assignment commented out
                new_file_item.flip_x = True                        # <-- NameError at runtime
                self.file_list.append(new_file_item)

        # handle y axis flips
        if self.dataset_config.flip_y:
            print_acc("  -  adding y axis flips")
            current_file_list = [x for x in self.file_list]
            for file_item in current_file_list:
                # create a copy that is flipped on the y axis
                new_file_item = copy.deepcopy(file_item)           # <-- flip_y twin, intact
                new_file_item.flip_y = True
                self.file_list.append(new_file_item)
```

and at the top of the file (L1–2):

```python
import copy
import copy          # <-- duplicate import
```

### Root cause — merge damage, verified with git blame

| Line | Commit | What happened |
|---|---|---|
| L569 `# create a copy copy.deepcopy(file_item)` | `5d50f814` "Finalized z-image" (2026-06-26) | original comment + the assignment line were fused: `new_file_item = copy.deepcopy(file_item)` was swallowed into the comment |
| L570–571 | `181f237a` (2023, original) | left untouched — now orphaned uses of an unassigned variable |
| L2 `import copy` | `5d50f814` | duplicate added during the same merge |

`main` still has the correct block:

```python
                # create a copy that is flipped on the x axis
                new_file_item = copy.deepcopy(file_item)
                new_file_item.flip_x = True
```

So this is a **regression introduced by the branch's merge**, not original upstream code. The
flip_y twin (L579) was left intact and serves as the reference pattern.

### Why it fails at runtime (empirical repros, 2026-08-30)

`new_file_item` is only ever *assigned* inside the flip_y block (L579). Because that assignment
exists anywhere in the function body, Python marks the name as **local** for the whole of
`__init__` — so the read at L570 hits the local slot before any binding and raises:

```python
# repro with the exact control flow of __init__ (both flip blocks present):
> D()   # flip_x=True path
UnboundLocalError : cannot access local variable 'new_file_item' where it is not associated with a value
```

Without the flip_y block in scope (stripped repro) it degrades to
`NameError: name 'new_file_item' is not defined`. Either way: **the flip_x path always raises**,
on the very first iteration — there is no input for which L570 can succeed.

### Blast radius

- Triggered only when `self.dataset_config.flip_x` is true — dataset *construction* time, i.e.
  before any training step; crash happens in `LoRADataset.__init__`.
- Current user config: `"flip_x": false` → dormant today (confirmed in `.job_config.json`).
- No other code depends on the broken behavior.

## Proposed change (restore to main's semantics)

### After — flip_x block

```python
        # handle x axis flips
        if self.dataset_config.flip_x:
            print_acc("  -  adding x axis flips")
            current_file_list = [x for x in self.file_list]
            for file_item in current_file_list:
                # create a copy that is flipped on the x axis
                new_file_item = copy.deepcopy(file_item)
                new_file_item.flip_x = True
                self.file_list.append(new_file_item)
```

### After — imports (L1–2 → L1)

```python
import copy
import json
```

The `deepcopy` is **required**, not cosmetic: a shallow `new_file_item = file_item` would set
`flip_x = True` on the *original* item too (same object), silently flipping every base sample. The
deepcopy twin at L579 confirms intent.

Net change: **+1 / −2 lines** (restore assignment + real comment; delete duplicate import).

## Validation plan (implementation session)

1. **Repro test**: with the fix applied, run the same minimal control-flow repro with `flip_x=True`
   → no exception; list doubles; originals unmutated (`file_item.flip_x` stays False on originals).
2. `pytest tests/` → expect 44 passed (no existing test covers dataset construction).
3. **Benchmark per protocol**: dormant under the current config (`flip_x: false`), so the short bench
   (6 epochs × 30 steps, 4 images) is expected to be *byte-identical* in timing — run it anyway as a
   no-regression confirmation vs bottom-out 3.09 s/it / samples 64.7 s/img.

## Results

### Implementation (2026-09-01, branch `krea_5`)

Applied exactly as proposed in `toolkit/data_loader.py`: restored the swallowed
`new_file_item = copy.deepcopy(file_item)` assignment + real comment in the flip_x block
(now identical in shape to the flip_y twin and to `main`), and deleted the duplicate
`import copy` at L2. Net **−1 line** (2 changed locations).

### Validation (2026-09-01, `.venv`) — ALL PASS

Control-flow repros (exact structure of the two flip blocks in one `__init__`, per the
proposal's root-cause analysis):

| Case | Result |
|---|---|
| `flip_x=True`: no exception; list doubles; flipped copies appended; **originals unmutated**; no accidental y flips | PASS |
| `flip_x=True` + `flip_y=True`: list ×4, all four (x,y) combos present exactly once, originals untouched | PASS |
| Real `AiToolkitDataset.__init__` source: 2× `copy.deepcopy(file_item)` (flip_x + flip_y), assignment present in flip_x block | PASS |
| Module header: single `import copy`, then `import json`; module compiles & imports cleanly | PASS |

- `pytest tests/` → **44 passed**.

### Benchmark (2026-09-02, short bench: 6 epochs × 30 steps, 4 images)

| Epoch | cum s/it @ epoch end | incremental s/it | samples s/img (×4) | samples avg |
|-------|----------------------|------------------|--------------------|-------------|
| 1 | 3.66 | 3.53 | 65.16 / 64.18 / 63.73 / 63.87 | 64.2 |
| 2 | 3.55 | 3.43 | 63.31 / 62.95 / 62.87 / 62.83 | 63.0 |
| 3 | 3.29 | 2.80 | 63.06 / 62.86 / 62.80 / 62.83 | 62.9 |
| 4 | 3.11 | 2.53 | 62.72 / 62.73 / 63.09 / 62.99 | 62.9 |
| 5 | 3.11 | 3.13 | 64.41 / 63.47 / 63.53 / 63.36 | 63.7 |
| 6 | **3.09** | 2.97 | 62.73 / 62.83 / 62.87 / 62.87 | 62.8 |

Bottom-out cumulative: **3.09 s/it** @ step 179 vs **#20 run's 3.08** — flat (−0.3%, noise).
Samples: overall avg ≈63.2 s/img, epochs 4–6 avg ≈63.1, min per-image 62.7. Samples came in
≈6% faster than the #20 run (≈65.9 / ≈67.4), but the fixed lines are **dormant under this
config** (`flip_x: false`) — they never execute, so nothing here is attributable to #21;
that delta is session drift (same pattern as #18/#19/20). Training flat as predicted.

Bench agrees with mechanism analysis (neutral) → no same-session control needed
(Testing Protocol #5 not triggered).

**Status: IMPLEMENTED — benchmarked, no regression. KEEP** (hard crash fix for `flip_x: true`
datasets; zero runtime cost under other configs). Historical best-sample figures (64.7, #16)
NOT updated from this run — cross-session numbers aren't comparable without a control.
