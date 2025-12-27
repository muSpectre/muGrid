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
# Try relative import first (for installed wheels where _muGrid.so is in the package)
# Fall back to absolute import (for development where _muGrid.so is in build directory)
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

# Import Python wrappers for main classes (these accept wrapped Field objects)
from .Wrappers import (  # noqa: E402
    CartesianDecomposition,
    ConvolutionOperator,
    FEMGradientOperator,
    FFTEngine,
    FileIONetCDF,
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
_ConvolutionOperator = _muGrid.ConvolutionOperator
_FFTEngine = _muGrid.FFTEngine
_GlobalFieldCollection = _muGrid.GlobalFieldCollection
_LocalFieldCollection = _muGrid.LocalFieldCollection
if hasattr(_muGrid, "FileIONetCDF"):
    _FileIONetCDF = _muGrid.FileIONetCDF

# Base classes and utilities (always C++ objects)
ConvolutionOperatorBase = _muGrid.ConvolutionOperatorBase
Decomposition = _muGrid.Decomposition
DynCoord = _muGrid.DynCoord
DynRcoord = _muGrid.DynRcoord
IterUnit = _muGrid.IterUnit
Pixel = _muGrid.Pixel
StorageOrder = _muGrid.StorageOrder
SubPt = _muGrid.SubPt
Unit = _muGrid.Unit
Verbosity = _muGrid.Verbosity

# FFT utility functions
fft_freq = _muGrid.fft_freq
fft_freqind = _muGrid.fft_freqind
fft_normalization = _muGrid.fft_normalization
get_hermitian_grid_pts = _muGrid.get_hermitian_grid_pts
rfft_freq = _muGrid.rfft_freq
rfft_freqind = _muGrid.rfft_freqind

# Domain indexing utilities
get_domain_ccoord = _muGrid.get_domain_ccoord
get_domain_index = _muGrid.get_domain_index

# Field classes and utilities
from .Field import Field  # noqa: E402
from .Field import wrap_field  # noqa: E402

# MPI communicator
from .Parallel import Communicator  # noqa: E402

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
    # Main classes (Python wrappers)
    "CartesianDecomposition",
    "Communicator",
    "ConvolutionOperator",
    "FFTEngine",
    "Field",
    "FileIONetCDF",
    "GlobalFieldCollection",
    "LocalFieldCollection",
    # Field utilities
    "wrap_field",
    # FFT utilities
    "fft_freq",
    "fft_freqind",
    "fft_normalization",
    "get_hermitian_grid_pts",
    "rfft_freq",
    "rfft_freqind",
    # Domain utilities
    "get_domain_ccoord",
    "get_domain_index",
    # Enums and types
    "ConvolutionOperatorBase",
    "LaplaceOperator",
    "FEMGradientOperator",
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
]
