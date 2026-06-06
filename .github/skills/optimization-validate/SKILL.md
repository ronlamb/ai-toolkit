# Optimization Validation Skill

## Purpose
Validate optimization changes using standardized test protocol and verification steps.

## Test Protocol

### Validation Requirements
Each change requires:
1. **Unit tests** proving correctness (no functionality broken)
2. **Speed test**: 3 epochs × 30 steps, generate 2 images

### Metrics to Collect
1. **Training time per iteration**: `X.XXs/it` from progress bar
2. **Sample generation time**: Time per image from "Generating Samples" progress

### Baseline Protocol
Run before any changes to establish baseline:
```
Training:   0%|          | 29/21300 [XX:XX<XX:XX:XX,  X.XXs/it, lr: X.Xe-X loss: X.XXXe-XX]
Samples:    Generating Samples:   0%|          | 0/2 [00:00<?, ?it/s]
            Generating Samples:  50%|#####     | 1/2 [XX:XX<XX:XX, XX.XXs/it]
            Generating Samples: 100%|##########| 2/2 [XX:XX<00:00, XX.XXs/it]
```

## Verification Checklist

### Performance Validation
- [ ] Baseline metrics recorded before change
- [ ] Test results collected after change (3 epochs × 30 steps, 2 images)
- [ ] Improvement calculated: `(baseline - new) / baseline * 100`
- [ ] Improvement >2% threshold met

### Functionality Validation
- [ ] Unit tests pass without errors
- [ ] No API breaks introduced
- [ ] Code maintains backward compatibility

### Code Quality Validation
- [ ] Change is ≤20 lines (surgical improvement)
- [ ] No rewrites or major refactoring
- [ ] Code is maintainable and readable

## Decision Matrix

| Criterion | Pass? | Action |
|-----------|-------|--------|
| Improvement >2% | ✅ Yes | Keep change |
| Improvement >2% | ❌ No | Revert or monitor |
| Unit tests pass | ✅ Yes | Continue |
| Unit tests fail | ❌ No | Revert immediately |
| API breaks | ✅ None | Continue |
| API breaks | ❌ Yes | Revert immediately |

## Validation Workflow

```
┌─────────────────────────────────────┐
│  Run baseline test (3 epochs × 30) │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌────────────────────┐
    │  Apply optimization │
    └──────────────┬─────┘
                   ▼
    ┌────────────────────┐
    │  Run test (3 epochs × 30) │
    └──────────────┬─────┘
                   ▼
    ┌────────────────────┐
    │  Calculate improvement │
    └──────────────┬─────┘
                   ▼
          ┌────────┴────────┐
          │ Improvement >2%?│
          └────────┬────────┘
              Yes  │  No
               ▼   ▼
        ┌───────┐ ┌───────┐
        │  Keep │ │ Revert│
        └───────┘ └───────┘

**Note**: If the test is successful and shows improvement over baseline, the new results become the **new baseline** for future optimization attempts. This allows incremental improvements to compound over time.
```

## Usage

Invoke this skill when:
- After implementing an optimization change
- Before committing changes to git
- When verifying results are statistically significant

## Reference

See also:
- **[Optimization Workflow](../optimization-workflow.md)** - General optimization protocols
- **[Decision Rules](../copilot-instructions.md#decision-rules)** - Proceed/revert criteria