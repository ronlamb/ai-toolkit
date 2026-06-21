# Z-Image MPS Optimization — Round 2

Please analyze the Z-Image base code, following the instructions in #file:z_image-mac-instructions.md

## Context from Round 1

**Accepted changes (current baseline):**
- **Task 1:** Cache pipeline in `get_generation_pipeline()` — eliminated redundant pipeline creation
- **Task 2:** Optimize `get_noise_prediction()` tensor ops — replaced `unsqueeze(2)` + `unbind()` with list comprehension, batched `.float()` conversion

**Baseline performance (Task 1+2):** ~7.48s/it training, ~36.3s/image generation

**Rejected changes (all showed regressions on MPS):**
- Task 3: Cache text_encoder device flag (+1.3% training, +1.1% gen)
- Task 4: Cache model device flag (+11% training, +2.2% gen)
- Task 5: Cache sigmas tensor (+7% training, +3.9% gen)
- Task 6: Cache prompt embeddings (+25% training, -4.7% gen — cache grew unbounded)
- Task 7: VAE to device at load (+3.5% training, flat gen)
- Task 8: flush() after generation (+9.4% training, +2% gen)

**Key lesson:** On MPS, device-state caching and flag checks add *more* overhead than the `.device` property or `.to()` calls they replace. The winning optimizations eliminated actual redundant work (allocations, object creation), not device transfers.

## Approach for Round 2

Focus on **eliminating computation**, not caching state. See `z_image-mac-instructions.md` for new task options.

Your first step is to plan the next change, getting input from me before implementing.

Remember: both training and sampling speed are important. Test 8 epochs × 30 steps, generate 2 images per epoch.