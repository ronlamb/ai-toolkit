# AI Toolkit Optimization Workflow

This document provides detailed protocols for the optimization process. See `copilot-instructions.md` for core workflow and decision rules.

## Search Targets

Focus on these areas when identifying bottlenecks:

- `extensions_built_in/diffusion_models/chroma/`
- `toolkit/stable_diffusion_model.py` (`generate_images()` method)
- `toolkit/memory_management/`
- `jobs/process/BaseExtractProcess.py`

## Key Patterns

Look for these common optimization opportunities (in priority order):

1. **CPU-to-GPU copies**: `load_file(path, 'cpu')` → `.to('cuda')`
2. **Multiple device transfers**: Sequential `.to(device)` calls that could be consolidated
3. **Redundant clone operations**: `.detach().clone().to("cpu")` patterns
4. **Unbatched operations**: Loop-based prompt encoding instead of batch processing

## Output Format

### For Each Optimization Change

```markdown
## Change #X: [Concise Title]

**Issue**: Description of the bottleneck

**Location**: File path, line numbers

**Current Code**:
```python
[relevant snippet]
```

**Optimized Code**:
```python
[new optimized version, ≤20 lines]
```

**Expected Impact**: Speed improvement estimate

**Test Plan**: How to validate this change
```

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

## Platform-Specific Results

### macOS (MPS - Apple Silicon)
- Record test results in `docs/optimization/mac-results.md`
- Use same test protocol: 3 epochs × 30 steps, generate 2 images
- Note any MPS-specific optimizations or compatibility fixes applied

### Other Platforms (CUDA)
- Record test results in `docs/optimization/results.md`
- Include platform details: GPU model, CUDA version
- Use same test protocol

## Platform-Specific Skills

For platform-specific optimizations, use the appropriate skill:

| Platform | Skill | Purpose |
|----------|-------|---------|
| NVIDIA CUDA | `cuda-optimization` | AMP, torch.compile, CUDA graphs |
| Apple Silicon MPS | `mps-optimization` | float32 constraints, no 8-bit optimizers |

The **`optimization`** skill automatically detects the platform and routes to the appropriate skill.

## MPS Compatibility Checklist

When making changes for Apple Silicon (M-series), use the **[MPS Optimization Skill](../skills/mps-optimization/SKILL.md)** for guidance:

- [ ] Use `torch.float32` instead of `torch.float64` (MPS doesn't support float64)
- [ ] 8-bit optimizers (bitsandbytes, Prodigy8bit) don't support MPS - use standard PyTorch optimizers
- [ ] Add print statements for fallback logic (informational only, no warnings)
- [ ] Don't add `else` print statements if none existed before
- [ ] Test on MPS device before committing

## Decision Tree

Use this when evaluating optimization candidates:

```
Is it ≤20 lines? ─ No → Skip
     │
     Yes
     │
Expected speedup >2%? ─ No → Skip
     │
     Yes
     │
Passes unit tests? ─ No → Revert
     │
     Yes
     │
No API breaks? ─ No → Revert
     │
     Yes → Implement
```

## Implementation Workflow

1. **Analyze** codebase for bottlenecks using Key Patterns
2. **Propose** top 5 changes with hypotheses and expected impact
3. **Implement** one change at a time (≤20 lines per function)
4. **Validate** with test protocol (3 epochs × 30 steps, 2 images)
5. **Document** results in appropriate results file
6. **Commit & push** to forked repo before next change

## Notes

- User tests manually after implementation
- Keep changes surgical - no rewrites
- If results are inconclusive, keep the change if it's safe (no downside)
- Revert changes that don't meet the >2% improvement threshold

## See Also
- **[Optimization Documentation Skill](../skills/optimization-documentation/SKILL.md)** - Generate standardized change templates, results tracking, and checklists
