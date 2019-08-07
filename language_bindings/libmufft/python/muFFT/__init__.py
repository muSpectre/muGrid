#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Mar 2018

@brief  Main entry point for muFFT Python module

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Lesser Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µSpectre; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import numpy as np

from muFFT.Communicator import Communicator
from muFFT.NetCDF import NCStructuredGrid

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import _muFFT
from _muFFT import (get_domain_ccoord, get_domain_index, get_hermitian_sizes,
                    FFT_PlanFlags)

has_mpi = _muFFT.Communicator.has_mpi

# This is a list of FFT engines that are potentially available.
#              |------------------------------- Identifier for 'FFT' class
#              |        |---------------------- Name of 1D engine class 
#              |        |          |----------- Name of 2D engine class
#              v        v          v         v- Name of 3D engine class
#                                   MPI parallel calcs --------|
#                                    Transposed output -v      v
_factories = {'fftw': ('FFTW_1d', 'FFTW_2d', 'FFTW_3d', False, False),
              'fftwmpi': ('FFTWMPI_1d', 'FFTWMPI_2d', 'FFTWMPI_3d', True, True),
              'pfft': ('PFFT_1d', 'PFFT_2d', 'PFFT_3d', True, True),
              'p3dfft': ('P3DFFT_1d', 'P3DFFT_2d', 'P3DFFT_3d', True, True)}


# Detect FFT engines. This is a convenience dictionary that allows enumeration
# of all engines that have been compiled into the library.
def _find_fft_engines():
    fft_engines = []
    for fft, (factory_name_1d, factory_name_2d, factory_name_3d, \
        is_transposed, is_parallel) in _factories.items():
        if factory_name_1d in _muFFT.__dict__ and \
            factory_name_2d in _muFFT.__dict__ and \
            factory_name_3d in _muFFT.__dict__:
            fft_engines += [(fft, is_transposed, is_parallel)]
    return fft_engines
fft_engines = _find_fft_engines()


def Communicator(communicator=None):
    """
    Factory function for the communicator class.

    Parameters
    ----------
    communicator: mpi4py or muFFT communicator object
        The bare MPI communicator. (Default: _muFFT.Communicator())
    """
    # If the communicator is None, we return a communicator that contains just
    # the present process.
    if communicator is None:
        communicator = _muFFT.Communicator()

    # If the communicator is already an instance if _muFFT.Communicator, just
    # return that communicator.
    if isinstance(communicator, _muFFT.Communicator):
        return communicator

    # Now we need to do some magic. See if the communicator that was passed
    # conforms with the mpi4py interface, i.e. it has a method 'Get_size'.
    # The present magic enables using either mpi4py or stub implementations
    # of the same interface.
    if hasattr(communicator, 'Get_size'):
        # If the size of the communicator group is 1, just return a
        # communicator that contains just the present process.
        if communicator.Get_size() == 1:
            return _muFFT.Communicator()
        # Otherwise, check if muFFT does actually have MPI support. If yes
        # we assume that the communicator is an mpi4py communicator.
        elif _muFFT.Communicator.has_mpi:
            return _muFFT.Communicator(MPI._handleof(communicator))
        else:
            raise RuntimeError('muFFT was compiled without MPI support.')
    else:
        raise RuntimeError("The communicator does not have a 'Get_size' "
                           "method. muFFT only supports communicators that "
                           "conform to the mpi4py interface.")


class FFT(object):
    def __init__(self, nb_grid_pts, nb_components=1, fft='fftw',
                 communicator=None):
        """
        The FFT class handles forward and inverse transforms and instantiates
        the correct engine object to carry out the transform.

        The class holds the plan for the transform. It can only carry out
        transforms of the size specified upon instantiation. All transforms are
        real-to-complex.
    
        Parameters
        ----------
        nb_grid_pts: list
            Grid nb_grid_pts in the Cartesian directions.
        nb_components: int
            Number of degrees of freedom per pixel in the transform. Default: 1
        fft: string
            FFT engine to use. Options are 'fftw', 'fftwmpi', 'pfft' and 'p3dfft'.
            Default: 'fftw'.
        communicator: mpi4py or muFFT communicator
            communicator object passed to parallel FFT engines. Note that
            the default 'fftw' engine does not support parallel execution.
            Default: None
        """
        fft = 'fftw' if fft == 'serial' else fft
        fft = 'fftwmpi' if fft == 'mpi' else fft

        communicator = Communicator(communicator)

        self._dim = len(nb_grid_pts)
        self._nb_grid_pts = nb_grid_pts
        self._nb_components = nb_components

        nb_grid_pts = list(nb_grid_pts)
        try:
            factory_name_1d, factory_name_2d, factory_name_3d, is_transposed, \
                is_parallel = _factories[fft]
        except KeyError:
            raise KeyError("Unknown FFT engine '{}'.".format(fft))
        if self._dim == 1:
            factory_name = factory_name_1d
        elif self._dim == 2:
            factory_name = factory_name_2d
        elif self._dim == 3:
            factory_name = factory_name_3d
        else:
            raise ValueError('{}-d transforms are not supported'
                             .format(self._dim))
        try:
            factory = _muFFT.__dict__[factory_name]
        except KeyError:
            raise KeyError("FFT engine '{}' has not been compiled into the "
                           "muFFT library.".format(factory_name))
        self.engine = factory(nb_grid_pts, nb_components, communicator)
        self.engine.initialise()
        # Is the output from the pybind11 wrapper transposed? This happens
        # because it eliminates a communication step in MPI parallel transforms.
        self._is_transposed = is_transposed

    def fft(self, data):
        """
        Forward real-to-complex transform.

        Parameters
        ----------
        data: array
            Array containing the data for the transform. For MPI parallel
            calculations, the array carries only the local subdomain of the
            data. The shape has to equal `nb_subdomain_grid_pts` with additional
            components contained in the fast indices. The shape of the component
            is arbitrary but the total number of data points must match
            `nb_components` specified upon instantiation.

        Returns
        -------
        out_data: array
            Fourier transformed data. For MPI parallel calculations, the array
            carries only the local subdomain of the data. The shape equals
            `nb_fourier_grid_pts` plus components.
        """
        field_shape = data.shape[:self._dim]
        component_shape = data.shape[self._dim:]
        if field_shape != self.nb_subdomain_grid_pts:
            raise ValueError('Forward transform received a field with '
                             '{} grid points, but FFT has been planned for a '
                             'field with {} grid points'.format(field_shape,
                                self.nb_subdomain_grid_pts))
        out_data = self.engine.fft(data.reshape(-1, self._nb_components).T)
        new_shape = tuple(self.engine.get_nb_fourier_grid_pts()) \
            + component_shape
        out_data = out_data.T.reshape(new_shape)
        if self._is_transposed:
            return out_data.swapaxes(0, 1)
        return out_data

    def ifft(self, data):
        """
        Inverse complex-to-real transform.

        Parameters
        ----------
        data: array
            Array containing the data for the transform. For MPI parallel
            calculations, the array carries only the local subdomain of the
            data. The shape has to equal `nb_fourier_grid_pts` with additional
            components contained in the fast indices. The shape of the component
            is arbitrary but the total number of data points must match
            `nb_components` specified upon instantiation.

        Returns
        -------
        out_data: array
            Fourier transformed data. For MPI parallel calculations, the array
            carries only the local subdomain of the data. The shape equals
            `nb_subdomain_grid_pts` plus components.
        """
        if self._is_transposed:
            data = data.swapaxes(0, 1)
        field_shape = data.shape[:self._dim]
        component_shape = data.shape[self._dim:]
        if field_shape != tuple(self.engine.get_nb_fourier_grid_pts()):
            raise ValueError('Inverse transform received a field with '
                             '{} grid points, but FFT has been planned for a '
                             'field with {} grid points'.format(field_shape,
                                tuple(self.engine.get_nb_fourier_grid_pts())))
        out_data = self.engine.ifft(data.reshape(-1, self._nb_components).T)
        new_shape = self.nb_subdomain_grid_pts + component_shape
        return out_data.T.reshape(new_shape)

    @property
    def nb_domain_grid_pts(self):
        return tuple(self.engine.get_nb_domain_grid_pts())

    @property
    def fourier_locations(self):
        fourier_locations = self.engine.get_fourier_locations()
        if self._is_transposed:
            if self._dim == 2:
                loc0, loc1 = fourier_locations
                return loc1, loc0
            elif self._dim == 3:
                loc0, loc1, loc2 = fourier_locations
                return loc1, loc0, loc2
        return tuple(fourier_locations)

    @property
    def nb_fourier_grid_pts(self):
        nb_fourier_grid_pts = self.engine.get_nb_fourier_grid_pts()
        if self._is_transposed:
            if self._dim == 2:
                n0, n1 = nb_fourier_grid_pts
                return n1, n0
            elif self._dim == 3:
                n0, n1, n2 = nb_fourier_grid_pts
                return n1, n0, n2
        return tuple(nb_fourier_grid_pts)

    @property
    def subdomain_locations(self):
        return tuple(self.engine.get_subdomain_locations())

    @property
    def nb_subdomain_grid_pts(self):
        return tuple(self.engine.get_nb_subdomain_grid_pts())

    @property
    def fourier_slices(self):
        return tuple(slice(start, start + length)
                     for start, length in zip(self.fourier_locations,
                                              self.nb_fourier_grid_pts))

    def wavevectors(self, domain_lengths=None):
        """
        Create an array containing wavevectors for the Fourier grid.

        Parameters
        ----------
        domain_lengths : array
            Phyiscal size of the simulation domain. If None, the method returns
            phases rather than wavevectors. (This is the default.)

        Returns
        -------
        wavevectors : array
            Wavevectors or phases
        """
        if domain_lengths is None:
            return (np.mgrid[self.fourier_slices].T /
                self.nb_domain_grid_pts).T
        else:
            return (np.mgrid[self.fourier_slices].T /
                np.asarray(domain_lengths)).T

    @property
    def subdomain_slices(self):
        return tuple(slice(start, start + length)
                     for start, length in zip(self.subdomain_locations,
                                              self.nb_subdomain_grid_pts))

    @property
    def normalisation(self):
        """
        1 / prod(self._nb_grid_pts)
        """
        return self.engine.normalisation()

    @property
    def is_transposed(self):
        """
        Return if internal storage order is transposed.
        """
        return self._is_transposed

    @property
    def communicator(self):
        return self.engine.get_communicator()
    
