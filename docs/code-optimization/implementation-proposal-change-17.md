# Change #17: Fix silently-dropped gradients in `txtfusion` (reentrant checkpoint bug)

**Status**: PROPOSED — **CORRECTNESS fix, not a speedup** (awaiting user decision)
**Complexity**: Simple (2 lines — add `use_reentrant=False` at two call sites)
**Impact**: Restores training of the entire `txtfusion` sub-network (4 transformer blocks + their LoRA adapters). Expected to make training *slower* per step, not faster — see cost table.

## Issue — this is a bug, found during Set-4 verification

`extensions_built_in/diffusion_models/krea2/src/mmdit.py`, two call sites:

```python
# TextFusionBlock.forward  (~line 280-283)
def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
    if torch.is_grad_enabled():
        return checkpoint(self._forward, x, mask)      # <-- line 282
    return self._forward(x, mask)

# TextFusionTransformer.forward  (~line 319-322)
def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
    if torch.is_grad_enabled():
        return checkpoint(self._forward, x, mask)      # <-- line 321
    return self._forward(x, mask)
```

Both call `torch.utils.checkpoint.checkpoint(...)` **without** `use_reentrant=`. On torch 2.9.1
this emits a `UserWarning` and silently uses the **reentrant** path (`CheckpointFunction.apply`).
The reentrant path builds **no backward graph at all when none of its inputs require grad** — it
returns the output with `requires_grad=False`.

In Krea 2 training the text context arrives as cached `AdvancedPromptEmbeds` (the dataset sets
`cache_text_embeddings: true`), so `x` and `mask` **do not require grad**. Result: the whole
`txtfusion` subgraph is detached from autograd, and every parameter inside it — including any LoRA
adapters attached to its Linears — receives **no gradient, ever**.

### Why this matters for LoRA training

`Krea2Model.get_transformer_block_names()` returns `["blocks"]`, and `lora_special.create_modules`
matches on a substring of the dotted module name. That substring matches
`txtfusion.layerwise_blocks.*` and `txtfusion.refiner_blocks.*`, so those Linears **do** get
trainable LoRA parameters — which then never move, because of the dropped-gradient bug above. The
sub-network is silently frozen in every training run to date.

## Evidence (measured on this machine, RTX 4090, torch 2.9.1+cu128)

**Model-level gradient test** (tiny `SingleStreamDiT`, LoRA-style "Linear weights trainable",
cached text embeds with `requires_grad=False`):

| config | txtfusion params with grad |
|---|---|
| current code (reentrant) | **0 / 33** — all dropped |
| `use_reentrant=False` at both sites | **33 / 33** restored |

Every other sub-network (`blocks.`, `txtmlp.`, `last.`, `first.`, `tproj.`, `tmlp.`) gets correct
gradients in *both* configs — only `txtfusion.` is affected, because only it uses the reentrant path.
(The main DiT blocks already use `use_reentrant=False` correctly at line ~587.)

**Component-level confirmation**: a standalone `TextFusionTransformer` forward under the default
(reentrant) checkpoint returns `requires_grad=False` / `grad_fn=None`; the direct (non-checkpoint)
path and the `use_reentrant=False` path both produce non-zero parameter gradients.

## Proposed change (minimal — correctness only)

Add `use_reentrant=False` at both call sites so the non-reentrant checkpoint always tracks
parameters through the recompute, independent of whether the inputs require grad:

```python
# TextFusionBlock.forward  and  TextFusionTransformer.forward
if torch.is_grad_enabled():
    return checkpoint(self._forward, x, mask, use_reentrant=False)
return self._forward(x, mask)
```

2 lines changed, both functions well under the 20-line limit. No API change; inference is unaffected
(the `torch.is_grad_enabled()` guard already skips checkpointing during sampling).

## Cost — fixing correctness makes training slower (this is expected)

Before this fix, `txtfusion`'s backward **never ran**, so its cost was invisible in the s/it metric.
After the fix, its forward + recompute + backward actually execute every step. Measured on a
standalone `TextFusionTransformer` at **real Krea 2 dims** (`txt_dim=2560`, heads=20, kvheads=20,
multiplier=4, 12 encoder layers → effective batch `B×12`), fixed upstream grad:

| checkpoint layout | Lt=256 fwd+bwd | Lt=512 fwd+bwd | peak (Lt=512) |
|---|---|---|---|
| **nested** (both sites, non-reentrant) ← what the minimal fix gives | 48.6 ms | 99.3 ms | 2284 MB |
| outer-only (transformer-level) | 41.1 ms | 83.1 ms | 2954 MB |
| inner-only (block-level) | 31.5 ms | 63.0 ms | 2954 MB |
| none (direct, no checkpoint) | 31.4 ms | 63.1 ms | 2954 MB |

So the minimal fix adds roughly **~100 ms/step at Lt=512** of real txtfusion compute that was
previously skipped — but `txtfusion` is a small slice vs the 28 main blocks on the full combined
sequence, so the end-to-end s/it impact should be modest. It also **lowers peak VRAM** (nested keeps
the lowest footprint), which is why the original author checkpointed it.

## Optional follow-up (separate decision — memory/perf tradeoff)

The two checkpoints are **accidentally nested**: `TextFusionTransformer.forward` wraps a checkpoint
around `_forward`, and each `TextFusionBlock` inside also wraps its own. That is double-recompute
(nested 99 ms vs single-level 63 ms at Lt=512). Flattening to a **single** checkpoint boundary would
save ~36 ms/step of txtfusion recompute, but raises peak VRAM by ~670 MB (2284 → 2954 MB). Because
that trades memory for speed and changes the training memory profile, it is left as a separate
user decision rather than folded into this correctness fix.

## Validation plan

1. Re-run the model-level gradient test after editing: confirm `txtfusion.` reports N/N params with
   non-zero grad (was 0/33).
2. Confirm no new `UserWarning` from `torch.utils.checkpoint`.
3. Full training benchmark per protocol (≥6 epochs × 30 steps, 4 images): expect s/it to **rise**
   slightly (txtfusion backward now runs) — this is the correctness cost, not a regression. Judge on
   whether generated samples improve, since txtfusion LoRA adapters finally train.

## Decision needed from user

This changes **what trains**, not just how fast. Approving it means accepting a small per-step slowdown
in exchange for actually training a sub-network that is currently dead weight. Recommend approving —
a frozen sub-network that consumes compute and VRAM but never learns is the worst outcome.
