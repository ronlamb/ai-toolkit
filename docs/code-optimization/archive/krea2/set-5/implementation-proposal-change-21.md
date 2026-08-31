# Change #21: Fix `data_loader.py` flip_x crash (`UnboundLocalError`) + duplicate import

**Status**: PROPOSED 2026-08-30 (approved for implementation in a separate session)
**Complexity**: Trivial (2 lines: restore 1 assignment, delete 1 duplicate import)
**Impact**: **Hard crash fix.** Any dataset configured with `flip_x: true` fails during dataset
setup with `UnboundLocalError: cannot access local variable 'new_file_item'`. Dormant under the
current Krea 2 config (`flip_x: false` in `.job_config.json`), so neutral for the active benchmark —
but this is a latent crash for every user who enables horizontal-flip augmentation.

## Issue — found during the main…krea_5 audit (2026-08-30)

`toolkit/data_loader.py`, `LoRADataset.__init__` (function starts L388), flip_x block at L565–571.

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
- Current user config: `"flip_x": false` → dormant today (confirmed in
  `output/anna_bell_sex_krea_ut/.job_config.json`).
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

*(pending implementation session)*
