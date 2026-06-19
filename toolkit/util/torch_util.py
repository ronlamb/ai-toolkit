"""
Torch device utility functions.

Consolidates common device-checking patterns (CUDA, MPS, CPU) into reusable utilities.
Created for Change #7: MPS Optimization - Device Check Consolidation.
"""

import contextlib
import gc
import torch


def get_device_type(device) -> str:
    """Get the device type string: 'cuda', 'mps', 'cpu', etc."""
    return getattr(device, 'type', str(device).split(':')[0])


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return torch.backends.mps.is_available()


def get_default_device() -> torch.device:
    """Get the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_cuda_device(device) -> bool:
    """Check if a device is CUDA."""
    return get_device_type(device) == 'cuda'


def is_mps_device(device) -> bool:
    """Check if a device is MPS."""
    return get_device_type(device) == 'mps'


def is_cpu_device(device) -> bool:
    """Check if a device is CPU."""
    return get_device_type(device) == 'cpu'


def memory_allocated_gb() -> float:
    """Get memory allocated in GB for CUDA, 0.0 for other devices."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def save_rng_state() -> dict:
    """Save CPU and CUDA RNG states."""
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }


def restore_rng_state(state: dict) -> None:
    """Restore CPU and CUDA RNG states."""
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def set_seed(seed: int) -> None:
    """Set both CPU and CUDA random seeds."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_autocast_context(device, enabled: bool = True, dtype=None):
    """Get autocast context for CUDA, or nullcontext for other devices."""
    device_type = get_device_type(device)
    if device_type == 'cuda':
        return torch.autocast(device_type='cuda', enabled=enabled, dtype=dtype)
    return contextlib.nullcontext()


def get_text_dtype(device) -> torch.dtype:
    """Get appropriate text encoding dtype: float32 for MPS, bfloat16 otherwise."""
    device_type = get_device_type(device)
    return torch.float32 if device_type == 'mps' else torch.bfloat16


def mps_safe_float(tensor, device=None):
    """Ensure floating point tensor is float32 on MPS."""
    if device is None:
        device = tensor.device
    if is_mps_device(device) and torch.is_floating_point(tensor):
        return tensor.to(torch.float32)
    return tensor


def synchronize(device=None) -> None:
    """Synchronize the appropriate device (CUDA or MPS)."""
    if device is None:
        device = get_default_device()
    device_type = get_device_type(device)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    elif device_type == 'mps':
        torch.mps.synchronize()


def flush_cuda_ipc() -> None:
    """Flush CUDA IPC resources (for OOM recovery)."""
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()


def flush_cache(garbage_collect: bool = True) -> None:
    """Flush CUDA and MPS caches, optionally run garbage collector."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if garbage_collect:
        gc.collect()
