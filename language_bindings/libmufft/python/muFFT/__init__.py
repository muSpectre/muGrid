#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Mar 2018

@brief  Main entry point for muFFT Python module

Copyright © 2018 Till Junge

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

import numpy as np

# from muFFT.NetCDF import NCStructuredGrid

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# We need to import muGrid, otherwise DynCcoord_t and other types won't be
# registered and implicitly convertible.
import _muGrid

import _muFFT
from _muFFT import (version, FourierDerivative, DiscreteDerivative,
                    FFT_PlanFlags, get_nb_hermitian_grid_pts)

import muFFT.Stencils2D
import muFFT.Stencils3D
from .Communicator import Communicator

has_mpi = _muFFT.Communicator.has_mpi

# This is a list of FFT engines that are potentially available.
#              |------------------------------- Identifier for 'FFT' class
#              |           |------------------- Name of engine class
#              |           |          |-------- MPI parallel calcs
#              v           v          v      v- Transposed output
_factories = {'fftw':    ('FFTW',    False, False),
              'fftwmpi': ('FFTWMPI', True,  True),
              'pfft':    ('PFFT',    True,  True)}


# Detect FFT engines. This is a convenience dictionary that allows enumeration
# of all engines that have been compiled into the library.
def _find_fft_engines():
    fft_engines = []
    for fft, (factory_name, is_transposed, is_parallel) in _factories.items():
        if factory_name in dir(_muFFT):
            fft_engines += [(fft, is_transposed, is_parallel)]
    return fft_engines


fft_engines = _find_fft_engines()


def FFT(nb_grid_pts, fft='fftw', communicator=None):
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
    nb_dof_per_pixel: int
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

    communicator = Communicator(communicator)

    # 'mpi' is a convenience setting that falls back to 'fftw' for single
    # process jobs and to 'fftwmpi' for multi-process jobs
    if fft == 'mpi':
        if communicator.size > 1:
            fft = 'fftwmpi'
        else:
            fft = 'fftw'

    try:
        factory_name, is_transposed, is_parallel = _factories[fft]
    except KeyError:
        raise KeyError("Unknown FFT engine '{}'.".format(fft))
    try:
        factory = getattr(_muFFT, factory_name)
    except KeyError:
        raise KeyError("FFT engine '{}' has not been compiled into the "
                       "muFFT library.".format(factory_name))
    engine = factory(nb_grid_pts, communicator)
    return engine
