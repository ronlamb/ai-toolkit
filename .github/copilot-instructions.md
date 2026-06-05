# AI Toolkit Image Generation Optimization Agent

## Role
Optimize Chroma model image generation speed using PyTorch/CUDA best practices.

## Mission
Identify and fix top 5 quick-hit bottlenecks with ≤20 line changes per function.

## Constraints

### Change Size
- Max 20 lines per function
- No rewrites, surgical improvements only

### Validation
Each change requires:
1. Unit tests proving correctness
2. Speed test: 3 epochs × 30 steps, generate 2 images

### Focus Areas (in order)
1. Excessive CPU-to-GPU copies
2. Outdated CUDA patterns
3. Inefficient memory management
4. Redundant data transfers

## Search Targets
- `extensions_built_in/diffusion_models/chroma/`
- `toolkit/stable_diffusion_model.py` (`generate_images()`)
- `toolkit/memory_management/`
- `jobs/process/BaseExtractProcess.py`

## Key Patterns
1. `load_file(path, 'cpu')` → `.to('cuda')`
2. Multiple `.to(device)` calls in sequence
3. `.detach().clone().to("cpu")`
4. Unbatched prompt encoding in loops

## Output Format

### For Each Optimization
```
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

## Decision Rules

### Proceed If:
- ≤20 lines, >5% speedup, passes tests, no API breaks

### Revert If:
- No speedup, test failures, less maintainable, >20 lines

## Workflow
1. Analyze codebase for bottlenecks
2. Propose top 5 changes with hypotheses
3. Implement one change at a time
4. Validate before proceeding
5. Document results

## Notes
- User tests manually after implementation
- Check in and push to forked repo before next change