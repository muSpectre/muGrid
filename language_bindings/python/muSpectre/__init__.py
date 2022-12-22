#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Mar 2018

@brief  Main entry point for muSpectre Python module

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

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

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import _muGrid
import _muFFT
import _muSpectre
from _muSpectre import (SplitCell, Formulation,
                        material, solvers, FiniteDiff, cell)
from _muSpectre import OneQuadPt, version

from _muSpectre import (version,
                        ProjectionApproxGreenOperator_2d,
                        ProjectionApproxGreenOperator_3d,
                        ProjectionSmallStrain_2d,
                        ProjectionSmallStrain_3d,
                        ProjectionSmallStrain_5q_3d,
                        ProjectionFiniteStrain_2d,
                        ProjectionFiniteStrain_3d,
                        ProjectionFiniteStrainFast_2d,
                        ProjectionFiniteStrainFast_3d,
                        ProjectionFiniteStrainFast_5q_3d,
                        ProjectionFiniteStrain_2q_2d,
                        ProjectionFiniteStrain_2q_3d,
                        ProjectionFiniteStrainFast_2q_2d,
                        ProjectionFiniteStrainFast_2q_3d,
                        FEMStencil,
                        Discretisation,
                        StoreNativeStress)

from muGrid import get_domain_ccoord, get_domain_index
from muFFT import (Communicator, DiscreteDerivative, FourierDerivative,
                   FFT_PlanFlags)
from _muGrid import Verbosity
from muFFT import Communicator, FourierDerivative, DiscreteDerivative, FFT, \
    FFT_PlanFlags, mangle_engine_identifier, UnknownFFTEngineError

from . import eshelby_slow
from . import gradient_integration
from . import stochastic_plasticity_search
from . import sensitivity_analysis
from . import vtk_export
from . import linear_finite_elements

from . import _muSpectre

_factories = {'pocketfft': ('PocketFFTCellFactory', False),
              'fftw': ('FFTWCellFactory', False),
              'fftwmpi': ('FFTWMPICellFactory', True),
              'pfft': ('PFFTCellFactory', True)}

_projections = {
    Formulation.finite_strain: 'FiniteStrainFast',
    Formulation.small_strain: 'SmallStrain',
}


def Cell(nb_grid_pts, domain_lengths, formulation=Formulation.finite_strain,
         gradient=None, weights=None, fft='serial', communicator=None,
         is_cell_split=SplitCell.non_split):
    """
    Instantiate a muSpectre Cell class.

    Parameters
    ----------
    nb_grid_pts: list
        Grid nb_grid_pts in the Cartesian directions.
    domain_lengths: list
        Physical size of the cell in the Cartesian directions.
    formulation: Formulation
        Formulation for strains and stresses used by the solver. Options are
        `Formulation.finite_strain` and `Formulation.small_strain`. Finite
        strain formulation is the default.
    gradient: list of subclasses of DerivativeBase
        This is the Nabla operator in vector form (a list of one instance of
        `DerivativeBase` per spatial direction). It is used to automatically
        construct the projection operator. The default is FourierDerivative for
        each direction.
    weights: list of floating point numbers
        Those are the weights associated with each quadrature points.
        Default: None; The quadrature point weight is set to one ([1]) if there
        is only a single quadrature point, otherwise an error is raised.
    fft: string
        FFT engine to use. Use 'mpi' if you want a parallel engine and 'serial'
        if you need a serial engine. It is also possible to specifically
        choose 'pocketfft', 'fftw', 'fftwmpi', 'pfft' or 'p3dfft'.
        Default: 'serial'.
    communicator: mpi4py or muFFT communicator
        communicator object passed to parallel FFT engines. Note that
        the 'pocketfft' and 'fftw' engines do not support parallel execution.


    Returns
    -------
    cell: object
        Return a muSpectre Cell object.
    """
    fft, communicator = mangle_engine_identifier(fft, communicator)

    if gradient is None:
        if weights is not None:
            raise ValueError('You cannot provide quadrature point weights if '
                             'no gradient is specified.')
        dims = len(nb_grid_pts)
        gradient = [FourierDerivative(dims, i) for i in range(dims)]
        weights = [1]
    elif weights is None:
        if len(gradient) == len(nb_grid_pts):
            weights = [1]
        else:
            raise ValueError('You must provide quadrature point weights if '
                             'you specify more than one quadrature point.')

    nb_grid_pts = list(nb_grid_pts)
    domain_lengths = list(domain_lengths)
    try:
        factory_name, is_parallel = _factories[fft]
    except KeyError:
        factory_name = None
    if factory_name is None:
        raise UnknownFFTEngineError("Unknown FFT engine '{}'.".format(fft))

    if is_cell_split == SplitCell.split:
        factory_name = factory_name + "Split"

    try:
        factory = _muSpectre.cell.__dict__[factory_name]
    except KeyError:
        factory = None

    if factory is None:
        raise UnknownFFTEngineError(
            "It seem that FFT engine '{}' has not been compiled into the "
            "muSpectre library, because the factory function '{}' is "
            "missing.".format(fft, factory_name))
    if communicator.size == 1:
        return factory(nb_grid_pts, domain_lengths, formulation, gradient, weights)
    else:
        return factory(nb_grid_pts, domain_lengths, formulation, gradient, weights,
                       communicator)


def Projection(nb_grid_pts, lengths,
               formulation=Formulation.finite_strain,
               gradient=None,
               weights=None,
               fft='serial', communicator=None):
    """
    Instantiate a muSpectre Projection class.

    Parameters
    ----------
    nb_grid_pts: list
        Grid nb_grid_pts in the Cartesian directions.
    lengths: list
        Physical length of the Cartesian direction.
    formulation: muSpectre.Formulation
        Determines whether to use finite or small strain formulation.
    gradient: list of subclasses of DerivativeBase
        Type of the derivative operator used for the projection for each
        Cartesian direction. Default is FourierDerivative for each direction.
    weights: list of floating point numbers
        Those are the weights associated with each quadrature points.
    fft: string
        FFT engine to use. Use 'mpi' if you want a parallel engine and 'serial'
        if you need a serial engine. It is also possible to specifically
        choose 'fftw', 'fftwmpi', 'pfft' or 'p3dfft'.
        Default: 'serial'.
    communicator: mpi4py or muFFT communicator
        communicator object passed to parallel FFT engines. Note that
        the default 'fftw' engine does not support parallel execution.


    Returns
    -------
    projection: object
        Return a muSpectre Projection object.
    """
    fft = FFT(nb_grid_pts, fft=fft, communicator=communicator)

    dims = len(nb_grid_pts)
    if gradient is None:
        if weights is not None:
            raise ValueError('You cannot provide quadrature point weights if '
                             'no gradient is specified.')
        gradient = [FourierDerivative(dims, i) for i in range(dims)]
        weights = [1]
    elif weights is None:
        if len(gradient) == len(nb_grid_pts):
            weights = [1]
        else:
            raise ValueError('You must provide quadrature point weights if '
                             'you specify more than one quadrature point.')

    nb_quad_pts = len(gradient)//dims
    if nb_quad_pts == 1:
        class_name = 'Projection{}_{}d'.format(
            _projections[formulation], len(nb_grid_pts))
    else:
        class_name = 'Projection{}_{}q_{}d'.format(
            _projections[formulation], nb_quad_pts, len(nb_grid_pts))

    try:
        factory = _muSpectre.__dict__[class_name]
    except KeyError:
        raise KeyError("Projection engine '{}' has not been compiled into the "
                       "muSpectre library.".format(class_name))
    return factory(fft, lengths, gradient, weights)
