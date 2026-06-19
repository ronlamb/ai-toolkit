# Ernie Image Model - Device Check Cleanup Plan

## Overview
No changes needed. All device checks in Ernie Image are simple comparisons that don't benefit from utility wrappers.

**Skipped patterns:**
- `device == torch.device("cpu")` in `ernie_image.py` (lines 211, 241, 269, 296, 317) — simple equality check, no value in wrapping
