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
#              |------------------------------- Identifier for 'FFT' factory
#              |        |---------------------- Name of class for 
#              |        |          |----------- Name class for 3D grids
#              v        v          v         v- Supports MPI parallel calcs
_factories = {'fftw': ('FFTW_2d', 'FFTW_3d', False),
              'fftwmpi': ('FFTWMPI_2d', 'FFTWMPI_3d', True),
              'pfft': ('PFFT_2d', 'PFFT_3d', True),
              'p3dfft': ('P3DFFT_2d', 'P3DFFT_3d', True)}


# Detect FFT engines. This is a convenience dictionary that allows enumeration
# of all engines that have been compiled into the library.
fft_engines = []
for fft, (factory_name_2d, factory_name_3d, is_parallel) in _factories.items():
    if factory_name_2d in _muFFT.__dict__ and \
        factory_name_3d in _muFFT.__dict__:
        fft_engines += [(fft, is_parallel)]


def FFT(resolutions, nb_components, fft='fftw', communicator=None):
    """
    Instantiate a muFFT FFT class.

    Parameters
    ----------
    resolutions: list
        Grid resolutions in the Cartesian directions.
    nb_components: int
        number of degrees of freedom per pixel in the transform
    fft: string
        FFT engine to use. Options are 'fftw', 'fftwmpi', 'pfft' and 'p3dfft'.
        Default is 'fftw'.
    communicator: mpi4py communicator
        mpi4py communicator object passed to parallel FFT engines. Note that
        the default 'fftw' engine does not support parallel execution.


    Returns
    -------
    cell: object
        Return a muSpectre Cell object.
    """
    resolutions = list(resolutions)
    try:
        factory_name_2d, factory_name_3d, is_parallel = _factories[fft]
    except KeyError:
        raise KeyError("Unknown FFT engine '{}'.".format(fft))
    if len(resolutions) == 2:
        factory_name = factory_name_2d
    elif len(resolutions) == 3:
        factory_name = factory_name_3d
    else:
        raise ValueError('{}-d transforms are not supported'
                         .format(len(resolutions)))
    try:
        factory = _muFFT.__dict__[factory_name]
    except KeyError:
        raise KeyError("FFT engine '{}' has not been compiled into the "
                       "muFFT library.".format(factory_name))
    if is_parallel:
        if MPI is None:
            raise RuntimeError('Parallel solver requested but mpi4py could'
                               ' not be imported.')
        if communicator is None:
            communicator = MPI.COMM_SELF
        return factory(resolutions, nb_components, MPI._handleof(communicator))
    else:
        if communicator is not None:
            raise ValueError("FFT engine '{}' does not support parallel "
                             "execution.".format(fft))
        return factory(resolutions, nb_components)
