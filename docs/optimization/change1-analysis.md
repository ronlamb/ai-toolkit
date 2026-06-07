# Change #1 Analysis: Scheduler Weights Caching

## Issue Summary

**Performance Degradation**: Training time increased across epochs (12.35s → 12.38s → 12.43s) instead of stabilizing or improving as expected with caching.

**Expected Pattern**: Expected runs should be somehting like 11.67, 11.89, 11.55s/it (stable or improving).

**Actual Pattern**: 12.35, 12.38, 12.43s/it (consistently degrading).

## Changes Made

### 1. SDTrainer.py (`extensions_built_in/sd_trainer/SDTrainer.py`)

**Before**:
```python
timestep_weight = self.sd.noise_scheduler.get_weights_for_timesteps(
    timesteps,
    v2=self.train_config.linear_timesteps2,
    timestep_type=self.train_config.timestep_type
).to(loss.device, dtype=loss.dtype)  # <-- PROBLEM: Moves cached weights
```

**After**:
```python
# NOTE: get_weights_for_timesteps already returns weights on the correct device
# so we don't need to move them again. This fixes MPS memory fragmentation.
timestep_weight = self.sd.noise_scheduler.get_weights_for_timesteps(
    timesteps,
    v2=self.train_config.linear_timesteps2,
    timestep_type=self.train_config.timestep_type
)
# Removed .to(loss.device, dtype=loss.dtype) call
```

### 2. custom_flowmatch_sampler.py (`toolkit/samplers/custom_flowmatch_sampler.py`)

**Added Caching Infrastructure**:
```python
# Caching attributes for device optimization
self._cached_weights = {}
self._cached_device = None
self._cached_dtype = None

def _cache_weights_for_device(self, device: torch.device, dtype: torch.dtype):
    """Cache all weight tensors on the specified device."""
    device_str = str(device)
    if device_str not in self._cached_weights:
        self._cached_weights[device_str] = {
            'default': self.default_weighing_tensor.to(device=device, dtype=dtype),
            'linear': self.linear_timesteps_weights.to(device=device, dtype=dtype),
            'linear2': self.linear_timesteps_weights2.to(device=device, dtype=dtype),
        }
    self._cached_device = device
    self._cached_dtype = dtype
```

**Modified `get_weights_for_timesteps()`**:
```python
def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False, timestep_type="linear") -> torch.Tensor:
    step_indices = self._get_step_indices(timesteps.to(self.timesteps.device))
    device = timesteps.device
    dtype = timesteps.dtype

    # Cache weights on first call or if device/dtype changes
    if self._cached_device is None or str(device) != str(self._cached_device) or dtype != self._cached_dtype:
        self._cache_weights_for_device(device, dtype)

    if timestep_type == "weighted":
        weights = self._cached_weights[str(device)]['default'][step_indices]
    elif v2:
        weights = self._cached_weights[str(device)]['linear2'][step_indices]
    else:
        weights = self._cached_weights[str(device)]['linear'][step_indices]
    return weights
```

## Root Cause Analysis

### The Real Problem: Device String Comparison

**Issue**: The caching logic uses `str(device)` for comparison, but MPS device strings can vary:
- `mps` vs `mps:0`
- Different string representations across PyTorch versions

**Impact**: If device strings don't match exactly, the cache is invalidated and weights are recached every iteration.

### The Secondary Problem: Cache Invalidation Per Iteration

**Even if device strings match**, the cache might be invalidated because:
1. **Device object identity**: `torch.device('mps') != torch.device('mps:0')` even if they refer to the same device
2. **Dtype changes**: If dtype changes between iterations (unlikely but possible)
3. **Scheduler recreation**: If the scheduler is recreated each epoch

### Why Performance Degraded

The original code had `.to(loss.device, dtype=loss.dtype)` which:
1. Created new tensors each iteration (defeating caching)
2. Caused MPS memory fragmentation
3. Was expensive but consistent

The new code:
1. Should reuse cached weights
2. But if cache is being invalidated each iteration, it's doing the same work plus cache overhead
3. The cache lookup and string comparison add small overhead

## Hypotheses for Degradation

### Hypothesis 1: Device String Mismatch
- `loss.device` returns `"mps"` 
- Scheduler's cached weights are on `"mps:0"`
- Cache never hits, always recreates

### Hypothesis 2: Scheduler Recreation
- If scheduler is recreated each epoch, cache is lost
- First epoch builds cache (slow), subsequent epochs rebuild (still slow)

### Hypothesis 3: Cache Key Inconsistency
- `str(torch.device('mps'))` might return different values in different contexts
- Cache key doesn't match across calls

## Recommended Debug Steps

### 1. Add Cache Hit/Miss Logging

```python
def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False, timestep_type="linear") -> torch.Tensor:
    step_indices = self._get_step_indices(timesteps.to(self.timesteps.device))
    device = timesteps.device
    dtype = timesteps.dtype
    
    device_str = str(device)
    
    # Debug logging
    if self._cached_device is not None:
        print(f"Cache check: device_str={device_str}, cached_device={str(self._cached_device)}, dtype={dtype}, cached_dtype={self._cached_dtype}")
        print(f"  device match: {str(device) == str(self._cached_device)}")
        print(f"  dtype match: {dtype == self._cached_dtype}")
    
    if self._cached_device is None or str(device) != str(self._cached_device) or dtype != self._cached_dtype:
        print(f"  -> CACHE MISS, caching for {device_str}")
        self._cache_weights_for_device(device, dtype)
    else:
        print(f"  -> CACHE HIT for {device_str}")
    
    # ... rest of method
```

### 2. Check Device String Consistency

```python
# Add this somewhere in the training loop to see what device strings are being used
print(f"loss.device: {loss.device}, str(loss.device): {str(loss.device)}")
print(f"scheduler cached device: {self.sd.noise_scheduler._cached_device}")
if self.sd.noise_scheduler._cached_weights:
    print(f"  cached keys: {list(self.sd.noise_scheduler._cached_weights.keys())}")
```

### 3. Verify Scheduler Persistence

Check if the scheduler is being recreated each epoch by adding:

```python
# In SDTrainer.__init__ or setup
print(f"Scheduler ID: {id(self.sd.noise_scheduler)}")

# At start of each epoch
print(f"Epoch {epoch} Scheduler ID: {id(self.sd.noise_scheduler)}")
```

## Alternative Fix Approaches

### Option A: Normalize Device Strings

```python
def _normalize_device(self, device: torch.device) -> str:
    """Normalize MPS device strings for consistent caching."""
    device_str = str(device)
    if device_str.startswith('mps'):
        return 'mps'  # Normalize to base form
    return device_str

# Use in cache lookup:
device_key = self._normalize_device(device)
```

### Option B: Use Device Index Instead of String

```python
def _get_device_key(self, device: torch.device) -> tuple:
    """Create stable device key regardless of string representation."""
    if device.type == 'mps':
        return ('mps', 0)  # MPS only has one device
    return (device.type, device.index)

# Use in cache:
device_key = self._get_device_key(device)
if device_key not in self._cached_weights:
    # cache it
```

### Option C: Force Device Consistency

```python
# In get_weights_for_timesteps, ensure we always use the same device representation
device = torch.device('mps')  # Force normalized MPS device
```

## Conclusion

The caching logic is sound in principle, but the implementation has issues with:
1. Device string comparison inconsistency
2. Potential cache invalidation each iteration
3. No visibility into cache hits/misses

**Next Steps**:
1. Revert changes and run new baseline
2. Add debug logging to understand cache behavior
3. Identify why cache isn't being reused across iterations
4. Implement one of the alternative fix approaches based on findings
