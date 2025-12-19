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
# FFT utility functions
CartesianDecomposition = _muGrid.CartesianDecomposition
ConvolutionOperator = _muGrid.ConvolutionOperator
ConvolutionOperatorBase = _muGrid.ConvolutionOperatorBase
Decomposition = _muGrid.Decomposition
DynCcoord = _muGrid.DynCcoord
DynRcoord = _muGrid.DynRcoord
FFTEngine = _muGrid.FFTEngine
GlobalFieldCollection = _muGrid.GlobalFieldCollection
IterUnit = _muGrid.IterUnit
LocalFieldCollection = _muGrid.LocalFieldCollection
Pixel = _muGrid.Pixel
StorageOrder = _muGrid.StorageOrder
SubPt = _muGrid.SubPt
Unit = _muGrid.Unit
Verbosity = _muGrid.Verbosity
fft_freq = _muGrid.fft_freq
fft_freqind = _muGrid.fft_freqind
fft_normalization = _muGrid.fft_normalization
get_domain_ccoord = _muGrid.get_domain_ccoord
get_domain_index = _muGrid.get_domain_index
get_hermitian_grid_pts = _muGrid.get_hermitian_grid_pts
has_cuda = _muGrid.has_cuda
has_gpu = _muGrid.has_gpu
has_netcdf = _muGrid.has_netcdf
has_rocm = _muGrid.has_rocm
rfft_freq = _muGrid.rfft_freq
rfft_freqind = _muGrid.rfft_freqind

# FileIONetCDF is only compiled into the library if NetCDF libraries exist
if hasattr(_muGrid, "FileIONetCDF"):
    OpenMode = _muGrid.FileIONetCDF.OpenMode

    def FileIONetCDF(file_name, open_mode=OpenMode.Read, communicator=None):
        """
        This function is used to open a NetCDF file with a specified mode and
        communicator.

        NetCDF (Network Common Data Form) is a set of software libraries
        and machine-independent data formats that support the creation, access,
        and sharing of array-oriented scientific data.

        Parameters
        ----------
        file_name : str
            The name of the NetCDF file to be opened. This should include the full
            path if the file is not in the current working directory.
        open_mode : OpenMode, optional
            The mode in which the file is to be opened. This should be a value from
            the OpenMode enumeration (Read, Write, Overwrite or Append).
            (Default: OpenMode.Read)
        communicator : Communicator, optional
            The MPI communicator to be used for parallel I/O. If this is not
            provided, the file I/O operations will be serial. (Default: None)

        Returns
        -------
        file : FileIONetCDF
            Returns a FileIONetCDF object which represents the opened file. This object
            can be used to read data from or write data to the file, depending on the
            open mode.
        """
        return _muGrid.FileIONetCDF(file_name, open_mode, Communicator(communicator))

else:

    def FileIONetCDF(*args, **kwargs):
        raise ModuleNotFoundError("muGrid was installed without netCDF support")


from .Field import Field  # noqa: F401, E402
from .Field import complex_field  # noqa: F401, E402
from .Field import fft_fourier_space_field  # noqa: F401, E402
from .Field import fft_real_space_field  # noqa: F401, E402
from .Field import int_field  # noqa: F401, E402
from .Field import real_field  # noqa: F401, E402
from .Field import uint_field  # noqa: F401, E402
from .Field import wrap_field  # noqa: F401, E402
from .Parallel import Communicator  # noqa: F401, E402

__version__ = _muGrid.version.description()
