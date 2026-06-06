# MPS Optimization Skill

## Purpose
Identify and fix performance bottlenecks in AI model training/inference for Apple Silicon (MPS) platforms.

## Platform Detection

```
Is device 'mps'? ─ No → Use cuda-optimization skill
     │
     Yes
     │
Continue with MPS patterns
```

## Focus Areas

### 1. Precision Constraints
- **Use `torch.float32` only** - MPS doesn't support float64
- **Avoid double tensors** - Convert with `.float()` before MPS operations

### 2. Optimizer Limitations
- **8-bit optimizers NOT supported** - bitsandbytes, Prodigy8bit fail on MPS
- **Use standard PyTorch optimizers** - Adam, AdamW, SGD with default precision

### 3. Device Consistency
- **Pipeline device assignment** - Ensure `pipeline.to(device)` is called
- **Latent image IDs** - Create on correct device, not CPU then move
- **VAE device matching** - VAE should be on same device as latents

### 4. Caching Patterns
- **Timestep weights** - Cache and detect device changes
- **Pipeline caching** - Reuse pipelines instead of recreating

## Key Patterns to Detect

1. **float64 usage**: Tensors created with default dtype on MPS
2. **8-bit optimizers**: bitsandbytes, Prodigy8bit, etc.
3. **CPU↔GPU transfers**: Tensors bouncing between CPU and MPS
4. **Missing pipeline caching**: Pipeline recreated unnecessarily
5. **Device mismatches**: VAE, latents on different devices

## Decision Matrix

| Pattern | Expected Impact | Complexity |
|---------|-----------------|------------|
| float32 conversion | Required (no crash) | Low |
| Standard optimizers | Required (no crash) | Low |
| Pipeline caching | 10-20% | Medium |
| Device consistency | Required (no crash) | Low |
| Timestep caching | 5-15% | Medium |

## MPS vs CUDA Differences

| Aspect | CUDA | MPS |
|--------|------|-----|
| Precision | float16, bfloat16 supported | float32 only (no float64) |
| Optimizers | 8-bit optimizers supported | Only standard PyTorch optimizers |
| Compilation | `torch.compile()` works well | Limited support |
| Graphs | CUDA graphs supported | Metal Performance graphs |

## Usage

Invoke this skill when:
- Optimizing for Apple Silicon (M-series) platforms
- Reviewing code for MPS-specific issues
- Testing on M1/M2/M3 Macs

## Reference

See also:
- **[Optimization Workflow](../optimization-workflow.md)** - General optimization protocols
- **[CUDA Optimization Skill](./cuda-optimization/SKILL.md)** - NVIDIA GPU-specific optimizations