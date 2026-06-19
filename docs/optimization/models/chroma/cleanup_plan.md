# Chroma Model - Device Check Cleanup Plan

## Overview
Chroma has 1 device check pattern that provides real value for consolidation.

**Note:** Simple device checks (`device.type == "mps"`, `torch.backends.mps.is_available()`, etc.) are left as-is — the utility functions add overhead without simplifying the code. We only replace patterns that encapsulate complex logic.

## Module 1: `extensions_built_in/diffusion_models/chroma/src/layers.py`

### Change 1.1: Replace autocast device check with `get_autocast_context()` ❌ REVERTED
**Line:** 290
**Status:** Reverted — function call overhead in hot path caused +0.35s/it degradation by step 119
**Root cause:** `get_autocast_context()` adds Python function call overhead on every forward pass. The inline conditional was faster.

**Lesson:** Even "clean" utility functions add measurable overhead in hot paths. Keep inline conditionals in tight loops.

---

## Validation Order
1. Apply Change 1.1 → Test training

## Rollback Plan
If any change causes issues, revert the specific change and mark as "reverted" in this plan.
