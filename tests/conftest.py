"""
Pytest configuration and shared fixtures for muGrid tests.

This module provides:
- Common fixtures (comm, gpu_device)
- GPU testing utilities (skip_if_gpu_unavailable, get_test_devices, create_device)
- CuPy availability checking
"""

import pytest

import muGrid

# =============================================================================
# CuPy availability
# =============================================================================

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


# =============================================================================
# GPU testing utilities
# =============================================================================


def get_test_devices():
    """Return list of device strings for parametrized tests.

    Returns ["cpu"] if GPU support is not compiled in, otherwise ["cpu", "gpu"].
    Use with @pytest.mark.parametrize("device", get_test_devices()).
    """
    devices = ["cpu"]
    if muGrid.has_gpu:
        devices.append("gpu")
    return devices


def create_device(device_str):
    """Create a Device object from string.

    Parameters
    ----------
    device_str : str
        Either "cpu" or "gpu".

    Returns
    -------
    Device or None
        None for CPU (uses default), Device.gpu() for GPU.
    """
    if device_str == "cpu":
        return None  # CPU uses default (no device argument)
    elif device_str == "gpu":
        return muGrid.Device.gpu()
    raise ValueError(f"Unknown device: {device_str}")


def skip_if_gpu_unavailable(device):
    """Skip a test if GPU is requested but not available.

    Call this at the start of parametrized tests that use the "device" parameter.
    Skips if:
    - device is "gpu" and no GPU hardware is available at runtime
    - device is "gpu" and CuPy is not installed (needed for GPU array access)

    Parameters
    ----------
    device : str
        The device string ("cpu" or "gpu").

    Example
    -------
    @pytest.mark.parametrize("device", get_test_devices())
    def test_something(device):
        skip_if_gpu_unavailable(device)
        # ... rest of test
    """
    if device == "gpu":
        if not muGrid.is_gpu_available():
            pytest.skip("No GPU device available at runtime")
        if not HAS_CUPY:
            pytest.skip("CuPy not available for GPU array access")


def get_array_module(device):
    """Get the array module (numpy or cupy) for the given device.

    Parameters
    ----------
    device : str
        Either "cpu" or "gpu".

    Returns
    -------
    module
        numpy for CPU, cupy for GPU.
    """
    if device == "gpu":
        return cp
    import numpy as np

    return np


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def comm():
    """Provide an MPI communicator (or serial fallback)."""
    try:
        from mpi4py import MPI

        return muGrid.Communicator(MPI.COMM_WORLD)
    except ImportError:
        return muGrid.Communicator()


@pytest.fixture
def gpu_device():
    """Provide a GPU device, skipping if unavailable.

    Use this fixture when a test requires GPU hardware.
    """
    if not muGrid.is_gpu_available():
        pytest.skip("No GPU device available at runtime")
    if not HAS_CUPY:
        pytest.skip("CuPy not available for GPU array access")
    return muGrid.Device.gpu()
