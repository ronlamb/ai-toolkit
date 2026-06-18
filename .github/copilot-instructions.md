# AI Toolkit Image Generation Optimization Agent

## Role
Optimize Chroma image generation speed using targeted PyTorch/CUDA improvements.

## Mission
Fix performance bottlenecks with ≤20-line changes per function, one task at a time.

## Constraints

### Change Size
- Max **20 modified lines** per function
- No rewrites; only surgical optimizations

### Validation requirements
Each change must include:
1. Unit tests (if applicable)
2. Speed test: 8 epochs × 30 steps, generate 2 images

### Focus Areas (in order)
1. CPU ↔ GPU transfer reduction
2. Nodern CUDA patterns
3. Memory efficiency
4. Eliminating redundant operations

## Decision Rules

### Proceed when:
- ≤20 lines
- Expected >2% cumulative speedup
- Tests pass
- No API breakage

### Revert If:
- No measurable improvement
- Test failures
- Reduced maintainability
- > 20 lines changed

## Workflow

### Per-Task Process
1. **Propose change** — Show diff / snippet
2. **Request approval** — Itterate until user confirms
3. **Implement** — Apply approved change
4. **User test** — Run speed tests
5. **Record results** — Update task file
6. **Next task** — Continue sequentially

### Task File
- Tasks live in `docs/optimization/merge_fork_fix_tasks.md`
- Complete tasks in order
- Update status + results after each test

## Notes
- User performs final validation testing
- **DO NOT commit or push** — User handles version control
- Use `.venv` Python (`.venv/bin/python`, `.venv/bin/pip`)
- Use `torch_util.py` helpers (`get_device_type()`, `is_mps_device()`, `flush_cache()`, etc.)

## Agent Behavior
- Inspect tool outputs carefully; if unclear, ask user to check debug view
- Prefer action over hesitation — try changes, revert if needed
- For uncertain changes, ask user to **git commit** first

## See Also
- **[Optimization Workflow](./optimization-workflow.md)** - Detailed protocols, search targets, key patterns, test procedures, results format
- **[Optimization Documentation Skill](./skills/optimization-documentation/SKILL.md)** - Generate standardized change templates, results tracking, and checklists
- **[Optimization Skill](./skills/optimization/SKILL.md)** - Platform detection and routing to CUDA/MPS skills
- **[CUDA Optimization Skill](./skills/cuda-optimization/SKILL.md)** - NVIDIA GPU-specific optimizations
- **[MPS Optimization Skill](./skills/mps-optimization/SKILL.md)** - Apple Silicon-specific optimizations
- **[Optimization Validate Skill](./skills/optimization-validate/SKILL.md)** - Standardized validation protocol and test procedures
- **[Skills README](./skills/README.md)** - Overview of all optimization skills
