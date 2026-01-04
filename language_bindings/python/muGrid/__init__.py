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

# Import Python wrappers for main classes (these accept wrapped Field objects)
from .Wrappers import (  # noqa: E402, E305
    CartesianDecomposition,
    FEMGradientOperator,
    FFTEngine,
    FileIONetCDF,
    GenericLinearOperator,
    GlobalFieldCollection,
    LaplaceOperator,
    LocalFieldCollection,
)

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

# Isotropic stiffness operators (fused elliptic kernels)
IsotropicStiffnessOperator2D = _muGrid.IsotropicStiffnessOperator2D
IsotropicStiffnessOperator3D = _muGrid.IsotropicStiffnessOperator3D
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

# Field classes and utilities
from .Field import Field  # noqa: E402
from .Field import wrap_field  # noqa: E402

# MPI communicator and parallel utilities
from .Parallel import Communicator, parprint  # noqa: E402

# Timing utility
from .Timer import Timer  # noqa: E402

# Version information
__version__ = _muGrid.version.description()

# Define public API
__all__ = [
    # Feature flags
    "has_mpi",
    "has_cuda",
    "has_rocm",
    "has_gpu",
    "has_netcdf",
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
    "IsotropicStiffnessOperator2D",
    "IsotropicStiffnessOperator3D",
    "Decomposition",
    "DynCoord",
    "DynRcoord",
    "IterUnit",
    "OpenMode",
    "Pixel",
    "StorageOrder",
    "SubPt",
    "Unit",
    "Verbosity",
    # Utilities
    "Timer",
    "parprint",
    # Linear algebra
    "linalg",
]
