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


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import _muFFT
from _muFFT import (get_domain_ccoord, get_domain_index, get_hermitian_sizes,
                    FFT_PlanFlags)

# This is a list of FFT engines that are potentially available.
#              |------------------------------- Identifier for 'FFT' class
#              |        |---------------------- Name of 2D engine class 
#              |        |          |----------- Name of 3D engine class
#              v        v          v         v- Supports MPI parallel calcs
_factories = {'fftw': ('FFTW_2d', 'FFTW_3d', False),
              'fftwmpi': ('FFTWMPI_2d', 'FFTWMPI_3d', True),
              'pfft': ('PFFT_2d', 'PFFT_3d', True),
              'p3dfft': ('P3DFFT_2d', 'P3DFFT_3d', True)}


# Detect FFT engines. This is a convenience dictionary that allows enumeration
# of all engines that have been compiled into the library.
def _find_fft_engines():
    fft_engines = []
    for fft, (factory_name_2d, factory_name_3d, is_parallel) in _factories.items():
        if factory_name_2d in _muFFT.__dict__ and \
            factory_name_3d in _muFFT.__dict__:
            fft_engines += [(fft, is_parallel)]
    return fft_engines
fft_engines = _find_fft_engines()


class FFT(object):
    def __init__(self, resolutions, nb_components=1, fft='fftw',
                 communicator=None):
        """
        The FFT class handles forward and inverse transforms and instantiates
        the correct engine object to carry out the transform.

        The class holds the plan for the transform. It can only carry out
        transforms of the size specified upon instantiation. All transforms are
        real-to-complex.
    
        Parameters
        ----------
        resolutions: list
            Grid resolutions in the Cartesian directions.
        nb_components: int
            Number of degrees of freedom per pixel in the transform. Default: 1
        fft: string
            FFT engine to use. Options are 'fftw', 'fftwmpi', 'pfft' and 'p3dfft'.
            Default: 'fftw'.
        communicator: mpi4py communicator
            mpi4py communicator object passed to parallel FFT engines. Note that
            the default 'fftw' engine does not support parallel execution.
            Default: None
        """
        self.dim = len(resolutions)
        self.resolutions = resolutions
        self.nb_components = nb_components
        self.communicator = communicator

        resolutions = list(resolutions)
        try:
            factory_name_2d, factory_name_3d, is_parallel = _factories[fft]
        except KeyError:
            raise KeyError("Unknown FFT engine '{}'.".format(fft))
        if self.dim == 2:
            factory_name = factory_name_2d
        elif self.dim == 3:
            factory_name = factory_name_3d
        else:
            raise ValueError('{}-d transforms are not supported'
                             .format(self.dim))
        try:
            factory = _muFFT.__dict__[factory_name]
        except KeyError:
            raise KeyError("FFT engine '{}' has not been compiled into the "
                           "muFFT library.".format(factory_name))
        if is_parallel:
            if MPI is None:
                raise RuntimeError('Parallel engine requested but mpi4py could'
                                   ' not be imported.')
            if communicator is None:
                communicator = MPI.COMM_SELF
            self.engine = factory(resolutions, nb_components,
                                  MPI._handleof(communicator))
        else:
            if communicator is not None and communicator.Get_size() > 1:
                raise ValueError("FFT engine '{}' does not support parallel "
                                 "execution.".format(fft))
            self.engine = factory(resolutions, nb_components)

        self.engine.initialise()

    def fft(self, data):
        """
        Forward real-to-complex transform.

        Parameters
        ----------
        data: array
            Array containing the data for the transform. For MPI parallel
            calculations, the array carries only the local subdomain of the
            data. The shape has to equal `subdomain_resolutions` with additional
            components contained in the fast indices. The shape of the component
            is arbitrary but the total number of data points must match
            `nb_components` specified upon instantiation.

        Returns
        -------
        out_data: array
            Fourier transformed data. For MPI parallel calculations, the array
            carries only the local subdomain of the data. The shape equals
            `fourier_resolutions` plus components.
        """
        field_shape = data.shape[:self.dim]
        component_shape = data.shape[self.dim:]
        if field_shape != self.subdomain_resolutions:
            raise ValueError('Forward transform received a field with '
                             'resolution {}, but FFT has been planned for a '
                             'field with resolution {}'.format(field_shape,
                                self.subdomain_resolutions))
        out_data = self.engine.fft(data.reshape(-1, self.nb_components).T)
        new_shape = self.fourier_resolutions + component_shape
        return out_data.T.reshape(new_shape)

    def ifft(self, data):
        """
        Inverse complex-to-real transform.

        Parameters
        ----------
        data: array
            Array containing the data for the transform. For MPI parallel
            calculations, the array carries only the local subdomain of the
            data. The shape has to equal `fourier_resolutions` with additional
            components contained in the fast indices. The shape of the component
            is arbitrary but the total number of data points must match
            `nb_components` specified upon instantiation.

        Returns
        -------
        out_data: array
            Fourier transformed data. For MPI parallel calculations, the array
            carries only the local subdomain of the data. The shape equals
            `subdomain_resolutions` plus components.
        """
        field_shape = data.shape[:self.dim]
        component_shape = data.shape[self.dim:]
        if field_shape != self.fourier_resolutions:
            raise ValueError('Inverse transform received a field with '
                             'resolution {}, but FFT has been planned for a '
                             'field with resolution {}'.format(field_shape,
                                self.fourier_resolutions))
        out_data = self.engine.ifft(data.reshape(-1, self.nb_components).T)
        new_shape = self.subdomain_resolutions + component_shape
        return out_data.T.reshape(new_shape)

    @property
    def domain_resolutions(self):
        return tuple(self.engine.get_domain_resolutions())

    @property
    def fourier_locations(self):
        return tuple(self.engine.get_fourier_locations())

    @property
    def fourier_resolutions(self):
        return tuple(self.engine.get_fourier_resolutions())

    @property
    def subdomain_locations(self):
        return tuple(self.engine.get_subdomain_locations())

    @property
    def subdomain_resolutions(self):
        return tuple(self.engine.get_subdomain_resolutions())

    @property
    def fourier_slices(self):
        return tuple((slice(start, start + length)
                      for start, length in zip(self.fourier_locations,
                                               self.fourier_resolutions)))

    @property
    def subdomain_slices(self):
        return tuple((slice(start, start + length)
                      for start, length in zip(self.subdomain_locations,
                                               self.subdomain_resolutions)))

    @property
    def normalisation(self):
        """
        1 / prod(self.resolutions)
        """
        return self.engine.normalisation()
