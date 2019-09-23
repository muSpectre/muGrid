# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   gradient_integration.py

@author Till Junge <till.junge@epfl.ch>
        Richard Leute <richard.leute@imtek.uni-freiburg.de>
        Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   22 Nov 2018

@brief  Functions for the integration of periodic first- and second-rank
        tensor fields on an n-dimensional rectangular grid

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

import sys

import numpy as np

import muFFT

from . import Formulation

def make_grid(lengths, nb_grid_pts):
    """For a dim-dimensional pixel/voxel grid, computes the pixel/voxel
    centre and corner positions as a function of the grid's edge
    lengths and number of grid points

    Keyword Arguments:
    lengths     -- np.ndarray of length dim with the edge lengths in each
                   spatial dimension (dtype = float)
    nb_grid_pts -- np.ndarray of length dim with the nb_grid_pts in each
                   spatial dimension (dtype = int)
    Returns:
    (nodal_positions, center_positions) two ndarrays with nodal/corner
    positions and center positions respectively. `nodal_positions` has one
    more entry in every direction than the number of grid points of the grid
    (added points correspond to the periodic repetitions).
    """
    nodal_positions = np.array(np.meshgrid(
        *[np.linspace(0, l, r+1) for l, r in zip(lengths, nb_grid_pts)],
        indexing="ij"))

    dx = lengths/nb_grid_pts
    dim = len(dx)
    center_positions = np.array(np.meshgrid(
        *[np.linspace(0, l, r, endpoint=False) for l, r in
          zip(lengths, nb_grid_pts)],
        indexing="ij") + 0.5*dx.reshape((dim,)+(1,)*dim))

    return nodal_positions, center_positions


def reshape_gradient(F, nb_grid_pts):
    """reshapes a flattened second-rank tensor into a multidimensional array of
    shape nb_grid_pts + [dim, dim].

    Note: this reshape entails a copy, because of the column-major to
    row-major transposition between Eigen and numpy

    Keyword Arguments:
    F           -- flattenen array of gradients as in OptimizeResult
    nb_grid_pts -- np.ndarray of length dim with the nb_grid_pts in each
                   spatial dimension (dtype = int)

    Returns:
    np.ndarray
    """

    dim = len(nb_grid_pts)
    if not isinstance(nb_grid_pts, list):
        raise Exception("nb_grid_pts needs to be in list form, "+
                        "for concatenation")
    expected_input_shape = [np.prod(nb_grid_pts) * dim**2]
    output_shape = list(reversed(nb_grid_pts)) + [dim, dim]
    if not ((F.shape[0] == expected_input_shape[0]) and
            (F.size == expected_input_shape[0])):
        raise Exception("expected gradient of shape {}, got {}".format(
            expected_input_shape, F.shape))

    order = list(range(dim+2))
    order[-2:] = reversed(order[-2:])
    order[0:dim] = reversed(order[0:dim])
    return F.reshape(output_shape).transpose(*order)


def complement_periodically(array, dim):
    """Takes an arbitrary multidimensional array of at least dimension dim
    and returns an augmented copy with periodic copies of the
    left/lower entries in the added right/upper boundaries

    Keyword Arguments:
    array -- arbitrary np.ndarray of at least dim dimensions
    dim   -- nb of dimension to complement periodically

    """
    shape = list(array.shape)
    shape[:dim] = [d+1 for d in shape[:dim]]
    out_arr = np.empty(shape, dtype = array.dtype)
    sl = tuple([slice(0, s) for s in array.shape])
    out_arr[sl] = array

    for i in range(dim):
        lower_slice = tuple([slice(0,s) if (d != i) else  0 for (d,s) in
                             enumerate(shape)])
        upper_slice = tuple([slice(0,s) if (d != i) else -1 for (d,s) in
                             enumerate(shape)])
        out_arr[upper_slice] = out_arr[lower_slice]

    return out_arr


def get_integrator(fft, gradient_op, grid_spacing):
    """Returns the discrete Fourier-space integration operator as a function
    of the position grid (used to determine the spatial dimension and number
    of grid points), the wave vectors, and the integration order. Note that
    the integrator contains the FFT normalisation factor.

    Keyword Arguments:
    fft         -- µFFT FFT object performing the FFT for a matrix on the cell
    gradient_op -- List of µSpectre DerivativeBase objects representing the
                   gradient operator.
    grid_spacing     -- np.array of grid spacing in each spatial direction of shape
                   (dim,).
    Returns:
    np.ndarray containing the fourier coefficients of the integrator
    """
    dim = len(grid_spacing)
    nb_grid_pts = np.asarray(fft.nb_domain_grid_pts)

    phase = fft.wavevectors()
    # The shift is needed to move the Fourier integration from the cell center
    # to the cell edges. We only compute it if at least one of the directions
    # report a fourier derivative.
    if any([_derivative.wrapped_object.__class__.__name__.startswith('Fourier')
            for _derivative in gradient_op]):
        shift = np.exp(1j*np.pi*np.sum(phase, axis=0))

    xi = np.zeros(phase.shape, dtype=complex)
    for i, (_derivative, _grid_spacing) in enumerate(zip(gradient_op, grid_spacing)):
        if _derivative.wrapped_object.__class__.__name__.startswith('Fourier'):
            # Shift to cell edges.
            xi[i] = _derivative.fourier(phase) * shift / _grid_spacing
        else:
            xi[i] = _derivative.fourier(phase) / _grid_spacing
    # Corrects the denominator to avoid division by zero for freqs = 0
    for i in range(dim):
        xi[i][(0,) * dim] = 1
    # The following is the integrator because taking its derivative should
    # be the unit operation. Taking the derivative is simply a dot product
    # with xi.
    integrator = xi.conj() / (xi*xi.conj()).sum(axis=0)
    # Correct integrator for freqs = 0
    for i in range(dim):
        integrator[i][(0,) * dim] = 0

    return integrator


def integrate_tensor_2(grad, fft_vec, fft_mat, gradient_op, grid_spacing):
    """Integrates a second-rank tensor gradient field to a chosen order as
    a function of the given field, the grid positions, and wave
    vectors. Optionally, the integration can be performed on the
    pixel/voxel corners (staggered grid).

    Keyword Arguments:
    grad           -- np.ndarray of shape nb_grid_pts_per_dim + [dim, dim]
                      containing the second-rank gradient to be integrated
    fft_vec        -- µFFT FFT object performing the FFT for a vector on the cell
    fft_mat        -- µFFT FFT object performing the FFT for a matrix on the cell
    gradient_op    -- µSpectre DerivativeBase class representing the gradient
                      operator.
    grid_spacing   -- np.array of grid spacing in each spatial direction of
                      shape (dim,).

    Returns:
    np.ndarray containing the integrated field
    """
    dim = len(grid_spacing)
    nb_grid_pts = np.array(grad.shape[:dim])
    lengths = nb_grid_pts * grid_spacing
    x = make_grid(lengths, nb_grid_pts)[0]
    integrator = get_integrator(fft_mat, gradient_op, grid_spacing)
    grad_k = (fft_mat.fft(grad) * fft_mat.normalisation)
    f_k = np.einsum("j...,...ij->...i", integrator, grad_k)
    grad_k_0 = grad_k[(0,)*dim]
    #The homogeneous integration computes the affine part of the deformation
    homogeneous = np.einsum("ij,j...->i...", grad_k_0.real, x)

    fluctuation_non_pbe = fft_vec.ifft(f_k)
    if np.linalg.norm(fluctuation_non_pbe.imag) > 1e-10:
        raise RuntimeError("Integrate_tensor_2() computed complex placements, "
                           "probably there went something wrong.\n"
                           "Please inform the developers about this bug!")
    fluctuation = np.moveaxis(
        complement_periodically(fluctuation_non_pbe.real, dim), -1, 0)

    return fluctuation + homogeneous


def integrate_vector(grad, fft_sca, fft_vec, gradient_op, grid_spacing):
    """Integrates a first-rank tensor gradient field to a chosen order as
    a function of the given field, the grid positions, and wave
    vectors. Optionally, the integration can be performed on the
    pixel/voxel corners (staggered_grid)

    Keyword Arguments:
    df             -- np.ndarray of shape nb_grid_pts_per_dim + [dim] containing
                      the first-rank tensor gradient to be integrated
    freqs          -- wave vectors as computed by compute_wave_vectors
    staggered_grid -- (default False) if set to True, the integration is
                      performed on the pixel/voxel corners, rather than the
                      centres. This leads to a different integration scheme...
    order          -- (default 0) integration order.
                      0 stands for exact integration

    Returns:
    np.ndarray contaning the integrated field
    """
    dim = len(grid_spacing)
    nb_grid_pts = np.array(grad.shape[:dim])
    lengths = nb_grid_pts * grid_spacing
    x = make_grid(lengths, nb_grid_pts)[0]
    integrator = get_integrator(fft_vec, gradient_op, grid_spacing)
    grad_k = (fft_vec.fft(grad) * fft_vec.normalisation)
    f_k = np.einsum("j...,...j->...", integrator, grad_k)
    grad_k_0 = grad_k[(0,)*dim]
    #The homogeneous integration computes the affine part of the deformation
    homogeneous = np.einsum("j,j...->...", grad_k_0.real, x)

    fluctuation_non_pbe = fft_sca.ifft(f_k)
    if np.linalg.norm(fluctuation_non_pbe.imag) > 1e-10:
        raise RuntimeError("Integrate_tensor_2() computed complex placements, "
                           "probably there went something wrong.\n"
                           "Please inform the developers about this bug!")
    fluctuation = complement_periodically(fluctuation_non_pbe.real, dim)

    return fluctuation + homogeneous


def compute_placement(result, lengths, nb_grid_pts, gradient_op,
                      fft=None, formulation=None):
    """computes the placement (the sum of original position and
    displacement) as a function of a OptimizeResult, domain edge
    lengths, domain discretisation nb_grid_pts, the chosen
    integration order and the continuum mechanics description(small or finite
    strain description)

    Keyword Arguments:
    result      -- OptimiseResult, or just the grad field of an OptimizeResult
    lengths     -- np.ndarray of length dim with the edge lengths in each
                   spatial dimension (dtype = float)
    nb_grid_pts -- np.ndarray of length dim with the nb_grid_pts in each
                   spatial dimension (dtype = int)
    gradient_op -- µSpectre DerivativeBase class representing the gradient
                   operator.
    fft         -- (default None) can be used to pass the FFT object from a
                   parallel simulation. Up to now only "None" is implemented in
                   the code.
    formulation -- (default None) the formulation is derived from the
                   OptimiseResult argument. If this is not possible you have to
                   fix the formulation to either Formulation.small_strain or
                   Formulation.finite_strain.
    Returns:
    (placement, nodal_positions)
                   tuple of ndarrays containing the placement and the
                   corresponding original nodal positions

    """

    lengths = np.array(lengths)
    nb_grid_pts = np.array(nb_grid_pts)

    #Check whether result is a np.array or an OptimiseResult object
    if isinstance(result, np.ndarray):
        if formulation == None:
            #exit the program, if the formulation is unknown!
            raise ValueError('\n'
                'You have to specify your continuum mechanics description.\n'
                'Either you use a formulation="small_strain" or '
                '"finite_strain" description.\n'
                'Otherwise you can give a result=OptimiseResult object, which '
                'tells me the formulation.')
        form = formulation
        grad = reshape_gradient(result, nb_grid_pts.tolist())
    else:
        form = result.formulation
        if form != formulation and formulation != None:
            #exit the program, if the formulation is ambiguous!
            raise ValueError('\nThe given formulation "{}" differs from the '
                             'one saved in your result "{}"!'
                             .format(formulation, form))
        grad = reshape_gradient(result.grad, nb_grid_pts.tolist())

    #reshape the gradient depending on the formulation
    if form == Formulation.small_strain:
        raise NotImplementedError('\nIntegration of small strains'
                                  'is not implemented yet!')
    elif form == Formulation.finite_strain:
        grad = grad
    else:
        raise ValueError('\nThe formulation: "{}" is unknown!'
                         .format(formulation))

    #load or initialise muFFT.FFT engine
    if fft is None:
        dim = len(nb_grid_pts)
        fft_mat = muFFT.FFT(nb_grid_pts, dim*dim) #FFT for (dim,dim) matrix
        fft_vec = muFFT.FFT(nb_grid_pts, dim)     #FFT for (dim) vector
    #compute the placement
    nodal_positions, _ = make_grid(lengths, nb_grid_pts)
    grid_spacing = np.array(lengths / nb_grid_pts)
    placement = integrate_tensor_2(grad, fft_vec, fft_mat, gradient_op, grid_spacing)

    return placement, nodal_positions
