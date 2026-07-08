#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   10 Oct 2019

@brief  Main entry point for muGrid Python module

Copyright © 2018 Till Junge

µGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

# Import the C++ extension module
# Try relative import first (for installed wheels where _muGrid.so is in the
# package). Fall back to absolute import (for development where _muGrid.so is
# in build directory)
try:
    from . import _muGrid
except ImportError:
    import _muGrid

has_mpi = _muGrid.Communicator.has_mpi
if has_mpi and MPI is None:
    raise RuntimeError("MPI support is enabled for muGrid but mpi4py is not available.")

# Feature flags for compile-time configuration
has_cuda = _muGrid.has_cuda
has_rocm = _muGrid.has_rocm
has_gpu = _muGrid.has_gpu
has_netcdf = _muGrid.has_netcdf
host_arch = _muGrid.host_arch
device_arch = _muGrid.device_arch

# Device abstraction
Device = _muGrid.Device
DeviceType = _muGrid.DeviceType

# Memory profiler: the `muGrid.memory_profiler` namespace (enable/disable/reset/
# report/summary) plus the `muGrid.memory_profile` context manager.
from . import MemoryProfiler as memory_profiler  # noqa: F401, E402
from .DeviceMemory import (  # noqa: F401, E402
    clear_device_allocator,
    restore_default_cupy_allocator,
    route_cupy_through_mugrid,
    set_device_allocator,
    use_cupy_allocator,
)

# Field classes and utilities
from .Field import Field, wrap_field  # noqa: F401, E402
from .MemoryProfiler import memory_profile  # noqa: F401, E402

# MPI communicator and parallel utilities
from .Parallel import Communicator, parprint  # noqa: F401, E402

# Import Python wrappers for main classes (these accept wrapped Field objects)
from .Wrappers import (  # noqa: F401, E402, E305
    CartesianDecomposition,
    FEMGradientOperator,
    FFTEngine,
    FileIONetCDF,
    GenericLinearOperator,
    GlobalFieldCollection,
    IsotropicStiffnessOperator,
    IsotropicStiffnessOperator2D,
    IsotropicStiffnessOperator3D,
    LaplaceOperator,
    LocalFieldCollection,
)

# Cache for runtime GPU availability check
_gpu_available_cache = None


def is_gpu_available():
    """Check if a GPU device is available at runtime.

    This function checks whether GPU hardware is actually accessible,
    not just whether muGrid was compiled with GPU support. It attempts
    to create a GPU device and allocate memory to verify the GPU works.

    The result is cached after the first call for performance.

    Returns
    -------
    bool
        True if a GPU device is available and functional, False otherwise.

    Examples
    --------
    >>> import muGrid
    >>> if muGrid.is_gpu_available():
    ...     fc = muGrid.GlobalFieldCollection((10, 10), device=muGrid.Device.gpu())
    ... else:
    ...     fc = muGrid.GlobalFieldCollection((10, 10))

    See Also
    --------
    has_gpu : Compile-time flag indicating GPU support was built.
    """
    global _gpu_available_cache
    if _gpu_available_cache is not None:
        return _gpu_available_cache

    # If not compiled with GPU support, definitely not available
    if not has_gpu:
        _gpu_available_cache = False
        return False

    try:
        # Try to create a GPU device and allocate memory
        # Device.gpu() may succeed even without hardware, so we test allocation
        device = Device.gpu()
        # Use keyword argument for device since positional args differ
        fc = _muGrid.GlobalFieldCollection(
            nb_domain_grid_pts=[2, 2],
            device=device,
        )
        _ = fc.real_field("test", [1])
        _gpu_available_cache = True
        return True
    except Exception:
        _gpu_available_cache = False
        return False


# Expose OpenMode for FileIONetCDF if available
if hasattr(_muGrid, "FileIONetCDF"):
    OpenMode = _muGrid.FileIONetCDF.OpenMode

# Low-level C++ classes (for advanced use cases)
# These are prefixed with underscore to indicate they're internal
_CartesianDecomposition = _muGrid.CartesianDecomposition
_GenericLinearOperator = _muGrid.GenericLinearOperator
_FFTEngine = _muGrid.FFTEngine
_GlobalFieldCollection = _muGrid.GlobalFieldCollection
_LocalFieldCollection = _muGrid.LocalFieldCollection
if hasattr(_muGrid, "FileIONetCDF"):
    _FileIONetCDF = _muGrid.FileIONetCDF

# Base classes and utilities (always C++ objects)
GradientOperator = _muGrid.GradientOperator
# Backwards compatibility alias
ConvolutionOperatorBase = GradientOperator
Decomposition = _muGrid.Decomposition

# Low-level isotropic stiffness operators (raw C++ classes; the Python
# wrappers imported above accept wrapped Field objects)
_IsotropicStiffnessOperator2D = _muGrid.IsotropicStiffnessOperator2D
_IsotropicStiffnessOperator3D = _muGrid.IsotropicStiffnessOperator3D

# Finite-element selector for the fused stiffness operator (simplex / Q1).
FEMElement = _muGrid.FEMElement
DynCoord = _muGrid.DynCoord
DynRcoord = _muGrid.DynRcoord
IterUnit = _muGrid.IterUnit
Pixel = _muGrid.Pixel
StorageOrder = _muGrid.StorageOrder
SubPt = _muGrid.SubPt
Unit = _muGrid.Unit

# Linear algebra operations on fields (Python wrapper with automatic _cpp extraction)
from . import linalg  # noqa: E402

# FFT utility functions
# Note: fft_freq, fft_freqind, rfft_freq, rfft_freqind are now properties
# on the FFTEngine class (fftfreq, ifftfreq). Use engine.fftfreq instead.
fft_normalization = _muGrid.fft_normalization
get_hermitian_grid_pts = _muGrid.get_hermitian_grid_pts

# Domain indexing utilities
get_domain_ccoord = _muGrid.get_domain_ccoord
get_domain_index = _muGrid.get_domain_index

# Version information
__version__ = _muGrid.version.description()

# NetCDF backend ("PnetCDF" with MPI, serial "NetCDF" otherwise) and its library
# version string; None when muGrid was built without NetCDF support.
netcdf_backend = getattr(_muGrid, "netcdf_backend", None)
netcdf_version = getattr(_muGrid, "netcdf_version", None)


def version_string(communicator=None, device=None):
    """Return a one-line diagnostic describing the muGrid build and the current
    run configuration, e.g.::

        muGrid 0.111.1 (MPI: ON, rank 0/4; CUDA: ON, device cuda:2; PnetCDF 1.14.1)

    Disabled features are reported as ``OFF`` (e.g. ``MPI: OFF``).

    Parameters
    ----------
    communicator : Communicator or mpi4py communicator, optional
        Reports the MPI rank and communicator size. Defaults to a serial
        communicator (rank 0, size 1).
    device : Device, optional
        A GPU device in use for this run. When given, its id is appended to the
        GPU section (e.g. ``device cuda:2``), analogous to the MPI rank. The GPU
        backend itself is reported as ``ON``/``OFF`` based on the build,
        independent of this argument.
    """
    parts = []

    # MPI
    if has_mpi:
        comm = communicator
        if comm is None:
            comm = _muGrid.Communicator()
        elif MPI is not None and isinstance(comm, MPI.Comm):
            comm = _muGrid.Communicator(comm)
        parts.append(f"MPI: ON, rank {comm.rank}/{comm.size}")
    else:
        parts.append("MPI: OFF")

    # GPU backend. Reported like the MPI/NetCDF sections: ON/OFF reflects the
    # build (has_gpu), and the label names the compiled backend (CUDA / ROCm).
    # When a GPU device is passed for this run, its id is appended (analogous to
    # the MPI rank).
    if has_gpu:
        build_label = "CUDA" if has_cuda else "ROCm/HIP" if has_rocm else "GPU"
        if device is not None and getattr(device, "is_device", False):
            if device.device_type == DeviceType.CUDA:
                parts.append(f"CUDA: ON, device cuda:{device.device_id}")
            elif device.device_type == DeviceType.ROCm:
                parts.append(f"ROCm/HIP: ON, device rocm:{device.device_id}")
            else:
                parts.append(f"{build_label}: ON, device gpu:{device.device_id}")
        else:
            parts.append(f"{build_label}: ON")
    else:
        parts.append("GPU: OFF")

    # NetCDF
    if has_netcdf:
        parts.append(f"{netcdf_backend} {netcdf_version.split()[0]}")
    else:
        parts.append("NetCDF: OFF")

    return f"muGrid {__version__} ({'; '.join(parts)})"


# Define public API
__all__ = [
    # Feature flags
    "has_mpi",
    "has_cuda",
    "has_rocm",
    "has_gpu",
    "has_netcdf",
    "netcdf_backend",
    "netcdf_version",
    "version_string",
    "host_arch",
    "device_arch",
    # Runtime checks
    "is_gpu_available",
    # Main classes (Python wrappers)
    "CartesianDecomposition",
    "Communicator",
    "GenericLinearOperator",
    "FFTEngine",
    "Field",
    "FileIONetCDF",
    "GlobalFieldCollection",
    "LocalFieldCollection",
    # Field utilities
    "wrap_field",
    # FFT utilities
    "fft_normalization",
    "get_hermitian_grid_pts",
    # Domain utilities
    "get_domain_ccoord",
    "get_domain_index",
    # Enums and types
    "GradientOperator",
    "ConvolutionOperatorBase",  # Backwards compatibility alias
    "LaplaceOperator",
    "FEMGradientOperator",
    "IsotropicStiffnessOperator",
    "IsotropicStiffnessOperator2D",
    "IsotropicStiffnessOperator3D",
    "Decomposition",
    "DynCoord",
    "DynRcoord",
    "IterUnit",
    "Pixel",
    "StorageOrder",
    "SubPt",
    "Unit",
    # Utilities
    "parprint",
    # Linear algebra
    "linalg",
]

# OpenMode is only available on NetCDF-enabled builds
if hasattr(_muGrid, "FileIONetCDF"):
    __all__.append("OpenMode")
