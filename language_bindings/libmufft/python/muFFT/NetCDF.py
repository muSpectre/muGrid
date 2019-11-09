#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   NetCDF.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   17 Jul 2019

@brief  NetCDF container for structured grid data with parallel I/O
        capabilities

Copyright © 2019 Lars Pastewka

µFFT is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µFFT is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µFFT; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import numbers
from functools import reduce
from math import sqrt

import numpy as np

from netCDF4 import Dataset

from .Communicator import Communicator

###

class NCStructuredGridFrame(object):
    def __init__(self, parent, index):
        self._parent = parent
        self._index = index
        if self._index < 0:
            self._index += len(parent)


    def _create_variable(self, name, value):
        if isinstance(value, numbers.Integral):
            self.parent._data.createVariable(name, 'i4', ('frame',))
        elif isinstance(value, numbers.Real):
            self.parent._data.createVariable(name, 'f8', ('frame',))
        else:
            self.parent._create_variable(name, value, prefix=['frame'])


    def _variable(self, name, template):
        if name not in self.parent._data.variables:
            self._create_variable(name, template)
        var = self.parent._data.variables[name]
        if self.parent.is_parallel:
            pass
            # Collective I/O causes trouble on some configurations. Disable
            # for now.
            #var.set_collective(True)
        return var


    def __getattr__(self, name):
        if name[0] == '_':
           # FIXME: Use getattr here?
           return self.__dict__[name]

        var = self.parent._data.variables[name]
        if var.dimensions[0] != 'frame':
            raise ValueError("Variable '{}' exists in NetCDF file, but it "
                             "does not store per-frame data.".format(name))
        if self.decomposition == 'subdomain':
            return var[self._index][self.subdomain_slices]
        else:
            return var[self._index]


    def __setattr__(self, name, value):
        if name[0] == '_':
            self.__dict__[name] = value
        elif isinstance(value, np.ndarray) and value.shape != ():
            var = self._variable(name, value)
            if self.decomposition == 'subdomain':
                var[tuple([self._index] + list(self.subdomain_slices))] = value
            else:
                var[self._index] = value
        else:
            var = self._variable(name, value)
            var[self._index] = value


    def __getitem__(self, name):
        return self.__getattr__(name)


    def __setitem__(self, name, value):
        return self.__setattr__(name, value)


    @property
    def index(self):
        return self._index


    @property
    def parent(self):
        return self._parent


    @property
    def decomposition(self):
        return self.parent.decomposition


    @property
    def nb_domain_grid_pts(self):
        return self.parent.nb_domain_grid_pts


    @property
    def subdomain_locations(self):
        return self.parent.subdomain_locations


    @property
    def nb_subdomain_grid_pts(self):
        return self.parent.nb_subdomain_grid_pts


    @property
    def subdomain_slices(self):
        return self.parent.subdomain_slices


    def sync(self):
        self.parent.sync()


class NCStructuredGrid(object):
    """
    Parallel I/O of structured grid data into a NetCDF file. The object once
    instantiated automatically creates NetCDF variables from data assigned to
    attributes. Data can be written to the file globally or to a frame.

    Serial example
    --------------

        import numpy as np
        from muFFT.NetCDF import NCStructuredGrid
        nb_grid_pts = [12, 11]
        nc = NCStructuredGrid('example.nc', 'w', nb_grid_pts)
        grid_data = np.random.random(nb_grid_pts)
        # Create a new variable `random_data` in the NetCDF file and write
        # `grid_data` to it.
        nc.random_data = grid_data
        # Create a new variable `per_frame_data` with first unlimited
        # dimension 'frame'. Assign `grid_data` to frame 1. (Frame 0
        # will be undefined in this case.)
        nc[1].per_frame_data = grid_data

    This example create a file with the following structure:
        netcdf example {
        dimensions:
            frame = UNLIMITED ; // (2 currently)
            grid_x = 12 ;
            grid_y = 11 ;
        variables:
            double random_data(grid_x, grid_y) ;
            double per_frame_data(frame, grid_x, grid_y) ;
        }

    Parallel example
    ----------------

        import numpy as np
        from mpi4py import MPI
        from muFFT import FFT
        from muFFT.NetCDF import NCStructuredGrid
        comm = MPI.COMM_WORLD
        # Initialize muFFT object - we need this to get domain decomposition
        # information.
        nb_grid_pts = [25, 25, 25]
        fft = FFT(nb_grid_pts, communicator=comm)
        # Compute something, here we simply use random numbers.
        local_grid = np.random.random(fft.nb_subdomain_grid_pts)
        # Initialize the I/O object
        nc = NCStructuredGrid('example.nc', 'w',
                              nb_domain_grid_pts=fft.nb_domain_grid_pts,
                              decomposition='subdomain',
                              subdomain_locations=fft.subdomain_locations,
                              nb_subdomain_grid_pts=fft.nb_subdomain_grid_pts,
                              communicator=comm)
        # We only write the local portion of the grid data because the
        # decomposition is 'subdomain'.
        nc[0].grid_data = local_grid

    This example create a file with the following structure:
        netcdf example {
        dimensions:
            frame = UNLIMITED ; // (1 currently)
            grid_x = 25 ;
            grid_y = 25 ;
            grid_z = 25 ;
        variables:
            double grid_data(frame, grid_x, grid_y, grid_z) ;
        }
    """

    _program = 'muFFT'
    _programVersion = 'N/A'

    def __init__(self, fn, mode='r', nb_domain_grid_pts=None,
                 decomposition='serial', subdomain_locations=None,
                 nb_subdomain_grid_pts=None, communicator=None,
                 frame=0, format='NETCDF3_64BIT_DATA'):
        """
        Open a NetCDF file for reading or writing.

        Parameters
        ----------
        fn : str
            Name of the NetCDF file.
        mode : str
            Opening mode: 'r' - read, 'w' - write, 'a' - append.
        decomposition : str
            Specification of the data decomposition of the heights array. If
            set to 'subdomain', the grids contain only the part of the full
            grid local to the present MPI process. If set to 'domain', the
            grids contains the global array. Default: 'serial', which fails
            for parallel runs.
        nb_domain_grid_pts : tuple of ints
            Number of grid points for the full topography. This is only
            required if decomposition is set to 'subdomain'.
        subdomain_locations : tuple of ints
            Origin (location) of the subdomain handled by the present MPI
            process.
        nb_subdomain_grid_pts : tuple of ints
            Number of grid points within the subdomain handled by the present
            MPI process. This is only required if decomposition is set to
            'domain'.
        communicator : mpi4py communicator
            The MPI communicator object.
        frame : int
            Initial frame. (Default: 0)
        format : str
            NetCDF format string. (Default: 'NETCDF3_64BIT_DATA')
        """
        self._data = None

        self._fn = fn
        self._decomposition = decomposition
        self._communicator = communicator
        self._mufft_communicator = Communicator(communicator)

        self._nb_domain_grid_pts = nb_domain_grid_pts
        self._subdomain_locations = subdomain_locations
        self._nb_subdomain_grid_pts = nb_subdomain_grid_pts

        # This is only a parallel run if more than one MPI process is spawned.
        # This test is necessary because on some configurations (unclear to me
        # exactly when) a call to `set_collective` complains with the error
        # "Parallel operation on file opened for non-parallel access" even
        # if there is only a single process even though the Dataset object was
        # instantiated with `parallel=True`.
        self._parallel = self._mufft_communicator.size > 1

        if decomposition == 'serial':
            if self.is_parallel:
                raise RuntimeError("'serial' decomposition requested by the "
                                   "size of the communicator is {} and hence "
                                   "larger than 1."
                                   .format(self._mufft_communicator.size))
            self._subdomain_locations = (0, 0)
            self._nb_subdomain_grid_pts = nb_domain_grid_pts
        else:
            if subdomain_locations is None or nb_subdomain_grid_pts is None:
                raise ValueError('Please specify `subdomain_locations` and '
                                 '`nb_subdomain_grid_pts` if decomposition is '
                                 'not serial.')
            if communicator is None:
                raise ValueError("Please specify a communicator since you are "
                                 "requesting parallel I/O with decomposition "
                                 "'{}'.".format(decomposition))

        self._data = Dataset(fn, mode, format=format,
                             parallel=self.is_parallel,
                             comm=self._communicator)

        if mode[0] == 'w':
            if nb_domain_grid_pts is None:
                raise ValueError('Please specify the number of grid points '
                                 '`nb_domain_grid_pts` when creating a new '
                                 'file.')
            self._data.program = self._program
            self._data.programVersion = self._programVersion

            self._create_grid_dimensions(nb_domain_grid_pts)
        else:
            self._read_grid_dimensions()

        if frame < 0:
            self._cur_frame = len(self)+frame
        else:
            self._cur_frame = frame


    def __del__(self):
        if self._data is not None:
            self._data.close()
            self._data = None


    @staticmethod
    def _grid_dimension_name(i):
        return 'grid_' + chr(ord('x') + i)


    def _create_grid_dimensions(self, nb_domain_grid_pts):
        self._nb_domain_grid_pts = tuple(nb_domain_grid_pts)

        if not 'frame' in self._data.dimensions:
            self._data.createDimension('frame', None)
        for i, n in enumerate(nb_domain_grid_pts):
            dim_name = self._grid_dimension_name(i)
            if not dim_name in self._data.dimensions:
                self._data.createDimension(dim_name, n)

        self._data.sync()


    def _read_grid_dimensions(self):
        i = 0
        nb_domain_grid_pts = []

        dim_exists = True
        while dim_exists:
            try:
                dim = self._data.dimensions[self._grid_dimension_name(i)]
                nb_domain_grid_pts += [len(dim)]
                i += 1
            except KeyError:
                dim_exists = False

        self._nb_domain_grid_pts = tuple(nb_domain_grid_pts)


    def _tensor_dimension_name(self, n):
        """
        Return a variable for a tensor component of length `n`.
        """
        dim_name = 'tensor_{}'.format(n)
        if not dim_name in self._data.dimensions:
            self._data.createDimension(dim_name, n)
        return dim_name


    def _create_variable(self, name, template, prefix=[]):
        """
        Heuristics to guess dimensions for a new variable that can contain
        multidimension arrays of types given by `template`.
        """
        nb_grid_dims = len(self._nb_domain_grid_pts)
        guessed_nb_grid_pts = template.shape[:nb_grid_dims]
        if self._decomposition == 'subdomain' and \
            len(template.shape) >= nb_grid_dims and \
            guessed_nb_grid_pts == tuple(self._nb_subdomain_grid_pts):
            component_dims = [self._tensor_dimension_name(i)
                              for i in template.shape[nb_grid_dims:]]
            grid_dims = [self._grid_dimension_name(i)
                         for i in range(nb_grid_dims)]
            dims = grid_dims + component_dims
        elif len(template.shape) >= nb_grid_dims and \
            guessed_nb_grid_pts == tuple(self._nb_domain_grid_pts):
            component_dims = [self._tensor_dimension_name(i)
                              for i in template.shape[nb_grid_dims:]]
            grid_dims = [self._grid_dimension_name(i)
                         for i in range(nb_grid_dims)]
            dims = grid_dims + component_dims
        else:
            dims = [self._tensor_dimension_name(i) for i in template.shape]
        dims = prefix + dims
        self._data.createVariable(name, template.dtype.str, tuple(dims))


    def _variable(self, name, template, prefix=[]):
        if name not in self._data.variables:
            self._create_variable(name, template, prefix)
        var = self._data.variables[name]
        if self.is_parallel:
            pass
            # Collective I/O causes trouble on some configurations. Disable
            # for now.
            #var.set_collective(True)
        return var


    def __len__(self):
        return len(self._data.dimensions['frame'])


    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None


    def get_filename(self):
        return self._fn


    def get_next_frame(self):
        frame = NCStructuredGridFrame(self, self._cur_frame)
        self._cur_frame += 1
        return frame


    def set_cursor(self, cur_frame):
        self._cur_frame = cur_frame


    def get_cursor(self):
        return self._cur_frame


    def __getattr__(self, name):
        if name[0] == '_':
            return self.__dict__[name]

        if name in self._data.variables:
            var = self._data.variables[name]
            if self.decomposition == 'subdomain':
                return var[self.subdomain_slices]
            else:
                return var

        return getattr(self._data, name)


    def __setattr__(self, name, value):
        if name[0] == '_':
            self.__dict__[name] = value
        elif isinstance(value, np.ndarray) and value.shape != ():
            var = self._variable(name, value)
            if self.decomposition == 'subdomain':
                var[self.subdomain_slices] = value
            else:
                var[...] = value
        else:
            return setattr(self._data, name, value)


    def __setitem__(self, i, value):
        if isinstance(i, str):
            return self.__setattr__(i, value)
        raise RuntimeError('Cannot set full frame.')


    def __getitem__(self, i):
        if isinstance(i, str):
            return self.__getattr__(i)
        if isinstance(i, slice):
            return [NCStructuredGridFrame(self, j)
                    for j in range(*i.indices(len(self)))]
        return NCStructuredGridFrame(self, i)


    def __iter__(self):
        for i in range(len(self)):
            yield NCStructuredGridFrame(self, i)


    @property
    def decomposition(self):
        return self._decomposition


    @property
    def nb_domain_grid_pts(self):
        return self._nb_domain_grid_pts


    @property
    def subdomain_locations(self):
        return self._subdomain_locations


    @property
    def nb_subdomain_grid_pts(self):
        return self._nb_subdomain_grid_pts


    @property
    def subdomain_slices(self):
        return tuple(slice(start, start + length)
                     for start, length in zip(self.subdomain_locations,
                                              self.nb_subdomain_grid_pts))


    @property
    def is_parallel(self):
        return self._parallel


    def sync(self):
        self._data.sync()


    def __str__(self):
        return self._fn

###

def open(fn, mode='r', frame=None, **kwargs):
    if isinstance(fn, NCStructuredGrid):
        return fn
    i = fn.find('@')
    if i > 0:
        n = int(fn[i+1:])
        fn = fn[:i]
        return NCStructuredGrid(fn, mode=mode, **kwargs)[n]
    elif frame is not None:
        return NCStructuredGrid(fn, mode=mode, **kwargs)[frame]
    else:
        return NCStructuredGrid(fn, mode=mode, **kwargs)
