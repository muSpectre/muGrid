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

import muFFT.Stencils1D
import muFFT.Stencils2D
import muFFT.Stencils3D

from muGrid import Communicator

has_mpi = _muGrid.Communicator.has_mpi


class UnknownFFTEngineError(Exception):
    """
    Exception used to indicate and unknown FFT engine identifier.
    """
    pass


# This is a list of FFT engines that are potentially available.
#              |--------------------------------- String identifier for 'FFT' class
#              |             |------------------- Name of engine class
#              |             |          |-------- Is output transposed?
#              v             v          v      v- Supports MPI parallel calculations?
_factories = {'pocketfft': ('PocketFFT', False, False),
              'fftw': ('FFTW', False, False),
              'fftwmpi': ('FFTWMPI', True, True),
              'pfft': ('PFFT', True, True)}


# Detect FFT engines. This is a convenience dictionary that allows enumeration
# of all engines that have been compiled into the library.
def _find_fft_engines():
    fft_engines = {}
    for fft, (factory_name, is_transposed, is_parallel) in _factories.items():
        try:
            factory = getattr(_muFFT, factory_name)
            fft_engines[fft] = (factory, is_transposed, is_parallel)
        except AttributeError:
            pass  # FFT engine is not compiled into the C++ code
    return fft_engines


fft_engines = _find_fft_engines()


def mangle_engine_identifier(fft, communicator=None):
    """
    Return normalized engine identifier. This will turn 'serial' and 'mpi'
    engine identifiers into the respective best-performing engine compiled
    into the code.

    Parameters
    ----------
    fft : string
        FFT engine to use. Use 'mpi' if you want a parallel engine and 'serial'
        if you need a serial engine. It is also possible to specifically
        choose 'pocketfft', 'fftw', 'fftwmpi' or 'pfft'.
    communicator : mpi4py or muGrid communicator
        communicator object passed to parallel FFT engines. Note that
        the 'pocketfft' and 'fftw' engines do not support parallel execution.
        Default: None
    """
    communicator = Communicator(communicator)
    if fft == 'mpi' and communicator.size == 1:
        fft = 'serial'
    if fft == 'serial':
        if 'fftw' in fft_engines:
            # Use FFTW for serial calculations if available since it is more
            # optimized than PocketFFT.
            return 'fftw', communicator
        else:
            return 'pocketfft', communicator
    elif fft == 'mpi':
        if 'fftwmpi' in fft_engines:
            return 'fftwmpi', communicator
        elif 'pfft' in fft_engines:
            # This is a fallback in case there is not FFTWMPI. May not work
            # in all cases.
            return 'pfft', communicator
        else:
            raise RuntimeError('No MPI parallel FFT engine was compiled into the code.')

    return fft, communicator


def get_engine_factory(fft, communicator=None):
    """
    Get engine factory given factory string identifier.

    Parameters
    ----------
    fft : string
        FFT engine to use. Use 'mpi' if you want a parallel engine and 'serial'
        if you need a serial engine. It is also possible to specifically
        choose 'pocketfft', 'fftw', 'fftwmpi' or 'pfft'.
    communicator : mpi4py or muGrid communicator
        communicator object passed to parallel FFT engines. Note that
        the 'pocketfft' and 'fftw' engines do not support parallel execution.
        Default: None
    """
    original_identifier = fft
    fft, communicator = mangle_engine_identifier(fft, communicator)

    try:
        factory, is_transposed, is_parallel = fft_engines[fft]
    except KeyError:
        factory = None

    if factory is None:
        raise UnknownFFTEngineError(
            "FFT engine with identifier '{}' (internally mangled to '{}') "
            "does not exist. If you believe this engine should exist, check "
            "that the code has been compiled with support for it."
            .format(original_identifier, fft))

    return factory, communicator


def FFT(nb_grid_pts, fft='serial', communicator=None, **kwargs):
    """
    The FFT class handles forward and inverse transforms and instantiates
    the correct engine object to carry out the transform.

    The class holds the plan for the transform. It can only carry out
    transforms of the size specified upon instantiation. All transforms are
    real-to-complex. if

    Parameters
    ----------
    nb_grid_pts : list
        Grid nb_grid_pts in the Cartesian directions.
    fft : string
        FFT engine to use. Use 'mpi' if you want a parallel engine and 'serial'
        if you need a serial engine. It is also possible to specifically
        choose 'pocketfft', 'fftw', 'fftwmpi' or 'pfft'.
        Default: 'serial'.
    communicator : mpi4py or muGrid communicator
        communicator object passed to parallel FFT engines. Note that
        the 'pocketfft' and 'fftw' engines do not support parallel execution.
        Default: None
    """
    factory, communicator = get_engine_factory(fft, communicator)
    return factory(nb_grid_pts, communicator, **kwargs)
