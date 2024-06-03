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

import _muGrid

has_mpi = _muGrid.Communicator.has_mpi
if has_mpi and MPI is None:
    raise RuntimeError("MPI support is enabled for muGrid but mpi4py is not available.")

from _muGrid import (get_domain_ccoord, get_domain_index, Pixel, StorageOrder, SubPt, DynCcoord, DynRcoord, IterUnit,
                     Verbosity, GlobalFieldCollection, LocalFieldCollection, Unit, Dictionary)

# FileIONetCDF is only compiled into the library if NetCDF libraries exist
if hasattr(_muGrid, 'FileIONetCDF'):
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
from .Parallel import Communicator

__version__ = _muGrid.version.description()
