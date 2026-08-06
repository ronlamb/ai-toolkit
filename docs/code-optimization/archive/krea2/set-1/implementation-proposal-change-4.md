# Implementation Proposal #4: Aggressive Gradient Checkpointing

## Status
⚠️ PROPOSED - Awaiting user testing

## Complexity
Complex (11-20 lines changed)

## Expected Impact
5-7% speedup (VRAM reduction, potential speedup from reduced memory bandwidth)

## Issue Description

Gradient checkpointing is already implemented in the `SingleStreamBlock` but could be applied more aggressively to the `TextFusionTransformer` and other transformer blocks. This would reduce VRAM usage and potentially improve speed by reducing memory bandwidth requirements.

## Current Code

### Location: `extensions_built_in/diffusion_models/krea2/src/mmdit.py`, lines 300-450

The current implementation only applies checkpointing to `SingleStreamBlock`:

```python
class SingleStreamBlock(nn.Module):
    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
        ref_span: tuple[int, int] | None = None,
        kv_capture: list | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if torch.is_grad_enabled():
            return checkpoint(self._forward, x, vec, freqs, mask, ref_span, kv_capture, kv_cache)
        return self._forward(x, vec, freqs, mask, ref_span, kv_capture, kv_cache)
    
    def _forward(self, x, vec, freqs, mask=None, ref_span=None, kv_capture=None, kv_cache=None):
        # ... actual forward implementation
```

## Optimized Code

Apply checkpointing to `TextFusionBlock` and `SingleStreamDiT` as well:

```python
class TextFusionBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if torch.is_grad_enabled():
            return checkpoint(self._forward, x, mask)
        return self._forward(x, mask)
    
    def _forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(torch.nn.Module):
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if torch.is_grad_enabled():
            return checkpoint(self._forward, x, mask)
        return self._forward(x, mask)
    
    def _forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x.reshape(b * l, d, n))
        x = x.reshape(b, l, d)

        for block in self.refiner_blocks:
            x = block(x, mask=mask)

        return x


class SingleStreamDiT(nn.Module):
    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        reflen: int = 0,
        isolate_refs: bool = False,
        ref_kv_capture: list | None = None,
        ref_kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if torch.is_grad_enabled():
            return checkpoint(
                self._forward, img, context, t, pos, mask, reflen, isolate_refs,
                ref_kv_capture, ref_kv_cache
            )
        return self._forward(img, context, t, pos, mask, reflen, isolate_refs, ref_kv_capture, ref_kv_cache)
    
    def _forward(self, img, context, t, pos, mask=None, reflen=0, isolate_refs=False,
                 ref_kv_capture=None, ref_kv_cache=None):
        # ... actual forward implementation
```

## Changes Summary

- Added `torch.is_grad_enabled()` check to `TextFusionBlock.forward()`
- Added `torch.is_grad_enabled()` check to `TextFusionTransformer.forward()`
- Added `torch.is_grad_enabled()` check to `SingleStreamDiT.forward()`

## Reasoning

Gradient checkpointing trades compute for memory by recomputing intermediate activations during backpropagation. This is beneficial when:

1. **Memory bandwidth is the bottleneck** (common in transformer models)
2. **The model has many layers** (Krea2 has 28 layers in SingleStreamDiT)
3. **Activations are large** (image tokens + text tokens)

By applying checkpointing more aggressively, we:
1. Reduce peak VRAM usage
2. Potentially improve speed by reducing memory pressure
3. Allow larger batch sizes or higher resolutions

## Validation Protocol

Run benchmark test:
- 3 epochs × 30 steps
- Generate 4 images per epoch

Monitor VRAM usage and compare against baseline results in `results-baseline.md`.

## Expected Results

- **VRAM usage**: 10-15% reduction
- **Training time**: 3-5% improvement (reduced memory bandwidth)
- **Sample generation**: No change (checkpointing only affects training)

## Known Limitations

1. Training will be slower per iteration due to recomputation
2. May not help if memory is not the bottleneck
3. Requires PyTorch 2.0+ for efficient checkpointing

## User Action Required

1. Test this change with the benchmark protocol
2. Report training time, sample generation times, and VRAM usage
3. If improvement >5% or VRAM reduction is significant, keep the change; otherwise, revert
