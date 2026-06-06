# AI Toolkit Optimization Skills

This directory contains VS Code agent skills for the AI Toolkit optimization project.

## Skill Hierarchy

```
.github/
├── copilot-instructions.md          # Core workflow and decision rules
├── optimization-workflow.md         # Detailed protocols and test procedures
└── skills/
    ├── README.md                    # This file - skill overview
    ├── optimization/                # Platform detection and routing
    │   └── SKILL.md                 # Routes to CUDA/MPS skills
    ├── cuda-optimization/           # NVIDIA GPU optimizations
    │   └── SKILL.md                 # CUDA-specific patterns
    ├── mps-optimization/            # Apple Silicon optimizations
    │   └── SKILL.md                 # MPS-specific patterns
    ├── optimization-documentation/  # Documentation templates
    │   └── SKILL.md                 # Change templates, results format
    └── optimization-validate/       # Validation protocol
        └── SKILL.md                 # Test protocol and verification
```

## Skill Descriptions

### 1. `optimization/` - Platform Detection
**Purpose**: Automatically detect the target platform and route to appropriate skill.

**When to use**: When you need to optimize performance but aren't sure which platform-specific skill to use.

**Features**:
- Detects CUDA vs MPS platforms
- Routes to appropriate optimization skill
- Provides platform comparison table

### 2. `cuda-optimization/` - NVIDIA GPU Optimizations
**Purpose**: Identify and fix performance bottlenecks for NVIDIA GPUs.

**When to use**: When optimizing for CUDA-enabled NVIDIA GPUs.

**Features**:
- AMP (Automatic Mixed Precision) patterns
- torch.compile optimization
- CUDA graphs
- Memory pinning strategies
- Gradient checkpointing

### 3. `mps-optimization/` - Apple Silicon Optimizations
**Purpose**: Identify and fix performance bottlenecks for Apple Silicon (MPS).

**When to use**: When optimizing for M1/M2/M3 Macs.

**Features**:
- float32-only constraints
- 8-bit optimizer limitations
- Device consistency patterns
- Pipeline caching for MPS

### 4. `optimization-documentation/` - Documentation Templates
**Purpose**: Generate standardized documentation for optimization changes.

**When to use**: When documenting a new optimization change or results.

**Features**:
- Change template format
- Results tracking format
- Implementation checklist generation

### 5. `optimization-validate/` - Validation Protocol
**Purpose**: Validate optimization changes using standardized test protocol.

**When to use**: After implementing an optimization change, before committing.

**Features**:
- Test protocol (3 epochs × 30 steps, 2 images)
- Performance verification checklist
- Decision matrix for keep/revert

## Usage Guide

### For New Users

1. **Start with the core instructions**: Read `.github/copilot-instructions.md`
2. **Understand the workflow**: Review `.github/optimization-workflow.md`
3. **Use platform-specific skills**: Invoke `cuda-optimization` or `mps-optimization`
4. **Document changes**: Use `optimization-documentation` skill
5. **Validate results**: Use `optimization-validate` skill

### For Experienced Users

1. **Use platform detection**: Invoke `optimization` skill for automatic routing
2. **Combine skills**: Use multiple skills in sequence (e.g., optimization + documentation)
3. **Reference templates**: Use `optimization-documentation` for consistent formatting

## Skill Invocation

Skills are invoked by the VS Code agent when:
- You ask about optimization patterns
- You need to document a change
- You want to validate results
- You're unsure which platform-specific skill to use

## Reference Files

| File | Purpose |
|------|---------|
| `copilot-instructions.md` | Core workflow and decision rules |
| `optimization-workflow.md` | Detailed protocols, test procedures, results format |
| `results.md` | CUDA optimization results history |
| `mac-results.md` | MPS optimization results history |
| `implementation-checklist.md` | High-level progress tracker |

## Platform Detection

The agent automatically detects the platform by checking:
1. `torch.cuda.is_available()` - NVIDIA CUDA
2. `torch.backends.mps.is_available()` - Apple Silicon MPS

## Decision Rules

### Proceed If:
- ≤20 lines, >2% expected improvement (cumulative), passes tests, no API breaks

### Revert If:
- No measurable improvement, test failures, less maintainable, >20 lines

## Notes

- User tests manually after implementation
- Check in and push to forked repo before next change
- Keep changes surgical - no rewrites