# Z Image Model - Device Check Cleanup Plan

## Overview
No changes needed. All device checks in Z Image are simple comparisons that don't benefit from utility wrappers.

**Skipped patterns:**
- `device == torch.device("cpu")` in `z_image.py` (lines 303, 329) — simple equality check, no value in wrapping
