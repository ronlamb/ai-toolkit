"""
Unit tests for toolkit/util/torch_util.py

Run with: pytest tests/test_torch_util.py -v
"""

import contextlib
import pytest
import torch
from unittest.mock import patch, MagicMock


# Import the module under test
from toolkit.util.torch_util import (
    get_device_type,
    is_cuda_available,
    is_mps_available,
    get_default_device,
    is_cuda_device,
    is_mps_device,
    save_rng_state,
    restore_rng_state,
    set_seed,
    get_autocast_context,
    get_text_dtype,
    mps_safe_float,
    synchronize,
    flush_cuda_ipc,
    flush_cache,
)


class TestGetDeviceType:
    def test_cuda_device(self):
        device = torch.device("cuda:0")
        assert get_device_type(device) == "cuda"

    def test_mps_device(self):
        device = torch.device("mps")
        assert get_device_type(device) == "mps"

    def test_cpu_device(self):
        device = torch.device("cpu")
        assert get_device_type(device) == "cpu"

    def test_string_device(self):
        # Edge case: string device
        assert get_device_type("cuda:0") == "cuda"
        assert get_device_type("mps") == "mps"
        assert get_device_type("cpu") == "cpu"


class TestIsCudaAvailable:
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_available(self, mock):
        assert is_cuda_available() is True

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_not_available(self, mock):
        assert is_cuda_available() is False


class TestIsMpsAvailable:
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_available(self, mock):
        assert is_mps_available() is True

    @patch("torch.backends.mps.is_available", return_value=False)
    def test_mps_not_available(self, mock):
        assert is_mps_available() is False


class TestGetDefaultDevice:
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_priority(self, mock):
        device = get_default_device()
        assert device.type == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_fallback(self, mock_cuda, mock_mps):
        device = get_default_device()
        assert device.type == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_cpu_fallback(self, mock_cuda, mock_mps):
        device = get_default_device()
        assert device.type == "cpu"


class TestIsCudaDevice:
    def test_cuda_device_true(self):
        assert is_cuda_device(torch.device("cuda:0")) is True

    def test_mps_device_false(self):
        assert is_cuda_device(torch.device("mps")) is False

    def test_cpu_device_false(self):
        assert is_cuda_device(torch.device("cpu")) is False


class TestIsMpsDevice:
    def test_mps_device_true(self):
        assert is_mps_device(torch.device("mps")) is True

    def test_cuda_device_false(self):
        assert is_mps_device(torch.device("cuda:0")) is False

    def test_cpu_device_false(self):
        assert is_mps_device(torch.device("cpu")) is False


class TestSaveRngState:
    def test_save_rng_state_structure(self):
        state = save_rng_state()
        assert "cpu" in state
        assert "cuda" in state
        assert isinstance(state["cpu"], torch.Tensor)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_rng_state", return_value=torch.tensor([1, 2, 3]))
    def test_save_rng_state_cuda(self, mock_get, mock_avail):
        state = save_rng_state()
        assert isinstance(state["cuda"], torch.Tensor)

    @patch("torch.cuda.is_available", return_value=False)
    def test_save_rng_state_no_cuda(self, mock):
        state = save_rng_state()
        assert state["cuda"] is None


class TestRestoreRngState:
    def test_restore_rng_state(self):
        # Save, modify, restore
        state = save_rng_state()
        torch.manual_seed(12345)
        restore_rng_state(state)
        # Should not raise

    @patch("torch.cuda.is_available", return_value=False)
    def test_restore_rng_state_no_cuda(self, mock):
        state = {"cpu": torch.get_rng_state(), "cuda": None}
        restore_rng_state(state)  # Should not raise


class TestSetSeed:
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.manual_seed")
    @patch("torch.manual_seed")
    def test_set_seed_cuda(self, mock_cpu, mock_cuda, mock_avail):
        set_seed(42)
        mock_cpu.assert_called_once_with(42)
        mock_cuda.assert_called_once_with(42)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.manual_seed")
    @patch("torch.manual_seed")
    def test_set_seed_no_cuda(self, mock_cpu, mock_cuda, mock_avail):
        set_seed(42)
        mock_cpu.assert_called_once_with(42)
        mock_cuda.assert_not_called()


class TestGetAutocastContext:
    def test_cuda_device_returns_autocast(self):
        device = torch.device("cuda:0")
        ctx = get_autocast_context(device)
        assert isinstance(ctx, torch.autocast)

    def test_mps_device_returns_nullcontext(self):
        device = torch.device("mps")
        ctx = get_autocast_context(device)
        assert isinstance(ctx, contextlib.nullcontext)

    def test_cpu_device_returns_nullcontext(self):
        device = torch.device("cpu")
        ctx = get_autocast_context(device)
        assert isinstance(ctx, contextlib.nullcontext)

    def test_enabled_false(self):
        device = torch.device("cuda:0")
        ctx = get_autocast_context(device, enabled=False)
        assert isinstance(ctx, torch.autocast)


class TestGetTextDtype:
    def test_mps_returns_float32(self):
        device = torch.device("mps")
        assert get_text_dtype(device) == torch.float32

    def test_cuda_returns_bfloat16(self):
        device = torch.device("cuda:0")
        assert get_text_dtype(device) == torch.bfloat16

    def test_cpu_returns_bfloat16(self):
        device = torch.device("cpu")
        assert get_text_dtype(device) == torch.bfloat16


class TestMpsSafeFloat:
    def test_mps_float64_converts_to_float32(self):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
        device = torch.device("mps")
        result = mps_safe_float(tensor, device=device)
        assert result.dtype == torch.float32

    def test_mps_float32_unchanged(self):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        device = torch.device("mps")
        result = mps_safe_float(tensor, device=device)
        assert result.dtype == torch.float32

    def test_cuda_float64_unchanged(self):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
        device = torch.device("cuda:0")
        result = mps_safe_float(tensor, device=device)
        assert result.dtype == torch.float64

    def test_integer_tensor_unchanged(self):
        tensor = torch.tensor([1, 2], dtype=torch.int64)
        device = torch.device("mps")
        result = mps_safe_float(tensor, device=device)
        assert result.dtype == torch.int64


class TestSynchronize:
    @patch("torch.cuda.synchronize")
    def test_cuda_synchronize(self, mock):
        synchronize(torch.device("cuda:0"))
        mock.assert_called_once()

    @patch("torch.mps.synchronize")
    def test_mps_synchronize(self, mock):
        synchronize(torch.device("mps"))
        mock.assert_called_once()

    @patch("torch.cuda.synchronize")
    @patch("torch.mps.synchronize")
    def test_cpu_no_synchronize(self, mock_mps, mock_cuda):
        synchronize(torch.device("cpu"))
        mock_cuda.assert_not_called()
        mock_mps.assert_not_called()


class TestFlushCudaIpc:
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.ipc_collect")
    def test_cuda_available(self, mock_ipc, mock_avail):
        flush_cuda_ipc()
        mock_ipc.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.ipc_collect")
    def test_cuda_not_available(self, mock_ipc, mock_avail):
        flush_cuda_ipc()
        mock_ipc.assert_not_called()


class TestFlushCache:
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("gc.collect")
    def test_cuda_cache_flush(self, mock_gc, mock_mps, mock_cuda, mock_avail):
        flush_cache()
        mock_cuda.assert_called_once()
        mock_gc.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("torch.mps.empty_cache")
    @patch("gc.collect")
    def test_mps_cache_flush(self, mock_gc, mock_mps, mock_cuda, mock_avail):
        flush_cache()
        mock_mps.assert_called_once()
        mock_gc.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("gc.collect")
    def test_no_cache_flush_gc_only(self, mock_gc, mock_mps, mock_cuda):
        flush_cache()
        mock_gc.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("gc.collect")
    def test_no_garbage_collect(self, mock_gc, mock_mps, mock_cuda):
        flush_cache(garbage_collect=False)
        mock_gc.assert_not_called()
