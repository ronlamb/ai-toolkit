# MPS Optimization Change #7: Missing MPS Logic in Chroma Training/Sampling Path

This change is an analysis if the code logic that determines whether to use "cuda", "mps" or "cpu", to reduce duplication.

Concentrating only on the code that is called during training and sampling the following model directories under `extensions_built_in/diffusion_models`:
    - `chroma`
    - `flux2`
    - `ernie_image`
    - `hidream`
    - `z_image`

This will be done in multiple steps defined below.

For each plan step:
    - Ask multiple questions to further refine the plan.
    - After each question show the full change and ask if there is any more changes to be made
    - If there are no changes then ask the user to switch to Agent mode and you will create the plan
    - Once the plan is created ask the user to review the plan and if it is OK
        - Go back to Plan mode and start the next Step.

## Step 1: Analyze the code

For each of these models there exists a directory of the same name under `docs/optimization/models`.  In each directory:
    - Create a file named <model_name>_modules.txt where model_name is the model name above ex: `chroma_modules.txt.
    - The main python application is `run.py`
    - in the file write all the python modules from gpu_checks_all.txt that are called during training and sampling
    - any python modules that are not called by one of these modules put in the file `docs/optimization/gpu_check_leftover`
    - also create a file called gpu_checks_change_7.txt in `docs/optimization` that contains all the python module names that were written in one or more of the <model_name>_modules.txt file.

## Step 2: Determine common gpu / cpu checks.

Create a markdown file called `docs/optimization/common_device_checks.md` that lists all distinct calls that could:
    - potentially be added to a utility module.
    - Moved inside another function

Potential code is code that is called more than once in multiple modules.

If it is code called more than once in a single module then it might be cleaner to move that code to a function in that module.

### Example code 

This next section shows examples of code:
    - Can be moved to utility function
    - Can be moved inside another function

For each code that can be moved into a utility function show the list of module names calling it, and how many total times it is called across all modules.

#### Example 1 - Can be added to a utility function

The following code exists in multiple modules and can be moved as a util function.

```python
rng_state = torch.get_rng_state()
cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
```

#### Example 2 - Can be moved inside another function

This code inside `extensions_built_in/diffusion_models/chroma/pipeline.py` is called twice.

```
if device.type == "mps":
    latent_image_ids = latent_image_ids.to(device)
else:
    latent_image_ids = latent_image_ids.to(device=device, dtype=dtype)
```

Both are called immediately after prepare_latent_image_ids. This code can be moved into prepare_latent_image_ids.

## Step 3: Create Plans for each model

The next step is to create a plan to cleanup the code for each model directory unders `docs/optimization/models`.

The code cleanup plan will go module by module, and will for each module, one by one
    - Add the code snippet to the utility file, if it's not already there.
    - Update for each code snippet for all occurances of that change in the module
        - test the change to see if it works
        - Ask the user to validate it as define in the "Validate against baseline" section.
        - If performance degrades, check logic to see if it can be fixed.
            - If not then revert the change.

## MPS Lessons Learned (from merge_fork_fix_tasks.md)

These rules were discovered through systematic testing. Apply them when refactoring device checks.

### What Works on MPS
- **Small cached constant tensors** — freqs, omega, etc. keyed by (shape, device) are safe and helpful
- **`detach().clone().requires_grad_(True)` outside `no_grad`** — avoids MPS command buffer sync every step
- **Skipping `gc.collect()` in hot paths** — reduces blocking pauses
- **Avoiding bf16↔fp32 round-trips** — expensive on MPS (57× per forward pass in Chroma)
- **`torch_util.py` helpers** — `is_mps_device()`, `get_text_dtype()`, `flush_cache(garbage_collect=False)`

### What Fails on MPS (Avoid These)
- **Persistent buffers for large tensors** — pins MPS memory, makes fragmentation worse (Task 8: +1.1 s/it)
- **Python micro-optimizations** — string matching, loop unrolling, index pre-computation add overhead (Tasks 3, 5, 6)
- **`spawn` multiprocessing workers** — can't share MPS tensor storage (Task 10: crash)
- **`.item()` in loops** — forces CPU sync per element
- **Direct device allocation** — sometimes regresses vs CPU-then-transfer (Task 5)
- **`.requires_grad_(True)` inside `torch.no_grad()`** — forces MPS command buffer sync every step (Task 4 root cause)

### MPS-Specific Validation Rules
1. **Measure after warmup** — MPS has warmup behavior; first iterations are slower
2. **Test stability over time** — check 8+ checkpoints, not just first epoch
3. **Check for degradation** — steady s/it increase indicates MPS sync or memory leak
4. **Sampling stability matters** — training speed gains mean nothing if sampling degrades

## Utility file

`torch_util.py` already exists in `toolkit/util/` with these functions:
- `get_device_type()`, `is_cuda_device()`, `is_mps_device()`
- `is_cuda_available()`, `is_mps_available()`, `get_default_device()`
- `save_rng_state()`, `restore_rng_state()`, `set_seed()`
- `get_autocast_context()`, `get_text_dtype()`, `mps_safe_float()`
- `synchronize()`, `flush_cuda_ipc()`, `flush_cache(garbage_collect=True)`

**Don't duplicate** — use existing helpers. Add new ones only if a pattern appears in 3+ modules.

## Files to check gpu_checks.txt

## Validate against baseline

After each change ask the user to run a test consisting of 3 epochs.
Each epoch consists of
- 30 training steps 
- 2 sample images, sampled at 4 steps

Use the baseline times in `mac-results.md` to determine whether a change adversely affected performance.

**Current baseline (after all optimizations):** ~11.97 s/it training, ~54.8 s/it sampling, stable over 12+ checkpoints.


