# CUDA Optimization Skill

## Purpose
Identify and fix performance bottlenecks in AI model training/inference for NVIDIA GPU (CUDA) platforms.

## Platform Detection

```
Is device 'cuda'? ─ No → Use mps-optimization skill
     │
     Yes
     │
Continue with CUDA patterns
```

## Focus Areas

### 1. AMP (Automatic Mixed Precision)
- Use `torch.cuda.amp.autocast()` for faster training with minimal accuracy loss
- Expected speedup: 30-50%

### 2. torch.compile (PyTorch 2.0+)
- Wrap model with `torch.compile()` for JIT compilation
- Expected speedup: 20-40%

### 3. CUDA Graphs
- Capture static computation graphs for replay
- Best for fixed-size workloads

### 4. Memory Pinning
- Use `pin_memory=True` in DataLoader
- Expected speedup: 5-15%

### 5. Gradient Checkpointing
- Trade compute for memory with `torch.utils.checkpoint`
- Expected memory reduction: 50-80%

## Key Patterns to Detect

1. **Missing AMP**: Training without `torch.cuda.amp.autocast()`
2. **No torch.compile**: Not using PyTorch 2.0+ compilation
3. **Unpinned memory**: DataLoader without `pin_memory=True`
4. **Synchronous transfers**: Missing async with `non_blocking=True`
5. **No gradient checkpointing**: Large models without memory optimization

## Decision Matrix

| Pattern | Expected Speedup | Complexity |
|---------|-----------------|------------|
| Enable AMP | 30-50% | Low |
| torch.compile | 20-40% | Low |
| Gradient checkpointing | 0-30% (memory) | Medium |
| Pinned memory | 5-15% | Low |
| Async transfers | 5-20% | Medium |

## MPS vs CUDA Differences

| Aspect | CUDA | MPS |
|--------|------|-----|
| Precision | float16, bfloat16 supported | float32 only (no float64) |
| Optimizers | 8-bit optimizers supported | Only standard PyTorch optimizers |
| Compilation | `torch.compile()` works well | Limited support |
| Graphs | CUDA graphs supported | Metal Performance graphs |

## Usage

Invoke this skill when:
- Optimizing for NVIDIA GPU platforms
- Reviewing code for CUDA-specific optimizations
- Training on CUDA-enabled GPUs

## Reference

See also:
- **[Optimization Workflow](../optimization-workflow.md)** - General optimization protocols
- **[MPS Optimization Skill](./mps-optimization/SKILL.md)** - Apple Silicon-specific optimizations