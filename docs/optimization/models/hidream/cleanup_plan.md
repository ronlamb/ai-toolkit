# HiDream Model - Device Check Cleanup Plan

## Overview
No changes needed. All device checks in HiDream are simple comparisons that don't benefit from utility wrappers.

**Skipped patterns:**
- `device.type == "mps"` in schedulers — explicit and readable as-is
- `torch.backends.mps.is_available()` in pipeline — simple check, no value in wrapping
