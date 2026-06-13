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

## Utility file

A new python module `torch_util.py` in the `toolkit/util` directory will be created.  Please feel free to come up with a better more standard name.

It will contain the following types functions.
- Functions that gets the current device type: `cpu`, `cuda`, `mps`, etc.
- Common routines and functions 

## Files to check gpu_checks.txt

## Validate against baseline

After each change ask the user to run a test consisting of 3 epochs.
Each epoch consists of
- 30 training steps 
- 2 sample images, sampled at 4 steps

Use the baseline times in `mac-results.md` to determine whether a change adversely affected performance.


