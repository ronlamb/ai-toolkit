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

## Decision Rules

### Proceed If:
- ≤20 lines, >2% expected improvement (cumulative), passes tests, no API breaks

### Revert If:
- No measurable improvement, test failures, less maintainable, >20 lines

## Workflow
1. Analyze codebase for bottlenecks
2. Propose top 5 changes with hypotheses
3. Implement one change at a time
4. Validate before proceeding
5. Document results

## Notes
- User tests manually after implementation
- Check in and push to forked repo before next change

## See Also
- **[Optimization Workflow](./optimization-workflow.md)** - Detailed protocols, search targets, key patterns, test procedures, results format
- **[Optimization Documentation Skill](./skills/optimization-documentation/SKILL.md)** - Generate standardized change templates, results tracking, and checklists
- **[Optimization Skill](./skills/optimization/SKILL.md)** - Platform detection and routing to CUDA/MPS skills
- **[CUDA Optimization Skill](./skills/cuda-optimization/SKILL.md)** - NVIDIA GPU-specific optimizations
- **[MPS Optimization Skill](./skills/mps-optimization/SKILL.md)** - Apple Silicon-specific optimizations
- **[Optimization Validate Skill](./skills/optimization-validate/SKILL.md)** - Standardized validation protocol and test procedures
- **[Skills README](./skills/README.md)** - Overview of all optimization skills
