# Flux2 Model - Device Check Cleanup Plan

## Overview
No changes needed. All device checks in Flux2 are simple comparisons that don't benefit from utility wrappers.

**Skipped patterns:**
- `device == torch.device("cpu")` in `flux2_model.py` (lines 412, 529, 546) — simple equality check, no value in wrapping
