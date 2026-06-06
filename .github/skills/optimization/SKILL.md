# Optimization Skill

## Purpose
Route optimization tasks to platform-specific skills (CUDA or MPS) based on the target device.

## Platform Detection

The agent should detect the platform by checking:
1. `torch.cuda.is_available()` - Returns True for NVIDIA CUDA
2. `torch.backends.mps.is_available()` - Returns True for Apple Silicon MPS

```
Platform Detection Flow:
┌─────────────────────────────────────┐
│  Check torch.cuda.is_available()    │
└──────────────┬──────────────────────┘
               │
          True │
               ▼
    ┌────────────────────┐
    │  Use cuda-optimization skill  │
    └────────────────────┘

┌─────────────────────────────────────┐
│  Check torch.backends.mps.is_available() │
└──────────────┬──────────────────────┘
               │
          True │
               ▼
    ┌────────────────────┐
    │  Use mps-optimization skill   │
    └────────────────────┘
```

## Usage

Invoke this skill when:
- You need to optimize AI model performance
- You're unsure which platform-specific skill to use
- You want automatic platform detection

## Platform-Specific Skills

| Platform | Skill | Use When |
|----------|-------|----------|
| NVIDIA CUDA | cuda-optimization | Training on NVIDIA GPUs |
| Apple Silicon MPS | mps-optimization | Training on M1/M2/M3 Macs |

## Reference

See also:
- **[Optimization Workflow](../optimization-workflow.md)** - General optimization protocols
- **[CUDA Optimization Skill](./cuda-optimization/SKILL.md)** - NVIDIA GPU-specific optimizations
- **[MPS Optimization Skill](./mps-optimization/SKILL.md)** - Apple Silicon-specific optimizations