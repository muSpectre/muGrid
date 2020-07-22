#!/usr/bin/env python3
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

import numpy as np
import itertools

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
        indexing="ij"), order='f')

    dx = lengths/nb_grid_pts
    dim = len(dx)
    center_positions = np.array(np.meshgrid(
        *[np.linspace(0, l, r, endpoint=False) for l, r in
          zip(lengths, nb_grid_pts)],
        indexing="ij") + 0.5*dx.reshape((dim,)+(1,)*dim))

    return nodal_positions, center_positions


def complement_periodically(array, dim):
    """Takes an arbitrary multidimensional array of at least dimension dim
    and returns an augmented copy with periodic copies of the
    left/lower entries in the added right/upper boundaries.

    Keyword Arguments:
    array -- arbitrary np.ndarray of at least dim dimensions
    dim   -- nb of dimension to complement periodically

    Returns:
    np.ndarray with left/lower entries added in the right/upper boundaries
    """
    shape = list(array.shape)
    tensor_rank = len(shape) - dim
    shape[-dim:] = [d+1 for d in shape[-dim:]]
    out_arr = np.empty(shape, dtype=array.dtype)
    sl = tuple([slice(0, s) for s in array.shape])
    out_arr[sl] = array

    for i in range(tensor_rank, dim + tensor_rank):
        lower_slice = tuple([slice(0, s) if (d != i) else 0 for (d, s) in
                             enumerate(shape)])
        upper_slice = tuple([slice(0, s) if (d != i) else -1 for (d, s) in
                             enumerate(shape)])
        out_arr[upper_slice] = out_arr[lower_slice]

    return out_arr


def get_integrator(fft, gradient_op, grid_spacing):
    """Returns the discrete Fourier-space integration operator as a function
    of the position grid (used to determine the spatial dimension and number
    of grid points), the wave vectors, and the integration order. Note that
    the integrator contains the FFT normalisation factor.

    Keyword Arguments:
    fft          -- µFFT FFT object performing the FFT for a matrix on the cell
    gradient_op  -- List of µSpectre DerivativeBase objects representing the
                    gradient operator.
    grid_spacing -- np.array of grid spacing in each spatial direction of shape
                    (dim,).
    Returns:
    np.ndarray containing the fourier coefficients of the integrator
    """
    dim = len(grid_spacing)
    nb_der = len(gradient_op)

    phase = fft.fftfreq
    # The shift is needed to move the Fourier integration from the cell center
    # to the cell edges. We only compute it if at least one of the directions
    # report a fourier derivative.
    if any([_derivative.__class__.__name__.startswith('Fourier')
            for _derivative in gradient_op]):
        shift = np.exp(1j*np.pi*np.sum(phase, axis=0))

    xi = np.zeros((nb_der,) + fft.nb_fourier_grid_pts, dtype=complex)
    for i, _derivative in enumerate(gradient_op):
        d = i % dim
        if _derivative.__class__.__name__.startswith('Fourier'):
            # Shift to cell edges.
            xi[i] = _derivative.fourier(phase) * shift / grid_spacing[d]
        else:
            xi[i] = _derivative.fourier(phase) / grid_spacing[d]
    # Corrects the denominator to avoid division by zero for freqs = 0
    for i in range(nb_der):
        xi[i][(0,) * dim] = 1
    # The following is the integrator because taking its derivative should
    # be the unit operation. Taking the derivative is simply a dot product
    # with xi.
    integrator = xi.conj() / (xi*xi.conj()).sum(axis=0)
    # Correct integrator for freqs = 0
    for i in range(nb_der):
        integrator[i][(0,) * dim] = 0

    return integrator


def integrate_tensor_2(grad, fft_engine, gradient_op, grid_spacing):
    """Integrates a second-rank tensor gradient field, given on the center
    positions of the grid, by a compatible integration operator derived from
    the gradient operator. The integrated field is returned on the node
    positions.

    Keyword Arguments:
    grad           -- np.ndarray of shape [dim, dim] + nb_grid_pts_per_dim
                      containing the second-rank gradient to be integrated
    fft_engine     -- µFFT FFT object performing the FFT for a matrix on the
                      cell
    gradient_op    -- µSpectre DerivativeBase class representing the gradient
                      operator.
    grid_spacing   -- np.array of grid spacing in each spatial direction of
                      shape (dim,).

    Returns:
    np.ndarray containing the integrated field
    """
    dim = len(grid_spacing)
    nb_der = len(gradient_op)
    nb_grid_pts = np.array(grad.shape[-dim:])
    lengths = nb_grid_pts * grid_spacing
    x = np.vstack((make_grid(lengths, nb_grid_pts)[0],)*(nb_der//dim))
    integrator = get_integrator(fft_engine, gradient_op, grid_spacing)
    grad_k_field = fft_engine.fetch_or_register_fourier_space_field(
        "grad_k", (dim, nb_der))
    fft_engine.fft(grad, grad_k_field)
    grad_k = grad_k_field.array()
    grad_k *= fft_engine.normalisation
    f_k = np.einsum("j...,ij...->i...", integrator, grad_k)
    grad_k_0 = grad_k[np.s_[:, :] + (0,)*(dim+1)]
    # The homogeneous integration computes the affine part of the deformation
    homogeneous = np.einsum("ij,j...->i...", grad_k_0.real, x) * dim / nb_der

    fluctuation_non_pbe = np.empty([dim, *fft_engine.nb_subdomain_grid_pts],
                                   order="f")
    fft_engine.ifft(f_k.copy(order='f'), fluctuation_non_pbe)
    if np.linalg.norm(fluctuation_non_pbe.imag) > 1e-10:
        raise RuntimeError("Integrate_tensor_2() computed complex placements, "
                           "probably there went something wrong.\n"
                           "Please inform the developers about this bug!")
    fluctuation = complement_periodically(fluctuation_non_pbe.real, dim)

    return fluctuation + homogeneous


def full_matrix_to_Voigt_vector(full_matrix):
    """
    Takes a matrix in the full notation (tensor notation) and returns
    the coressponding Voigt vector notation of the matrix.

        a₁₁ a₁₂ a₁₃
    A = a₂₁ a₂₂ a₂₃ ⇒ (a₁₁,a₂₂,a₃₃, (a₂₃+a₃₂), (a₁₃+a₃₁), (a₁₂+a₂₁))
        a₃₁ a₃₂ a₃₃

    Keyword Arguments:
    full_matrix     -- np.ndarray of shape [dim, dim] containing the
                       full notation (tensor notation) of a matrix.

    Returns:
    np.ndarray of shape [dim * (dim+1) / 2] containing the voigt notation of
    the input matrix.
    """
    full_matrix = np.asarray(full_matrix)
    if full_matrix.shape[0] == 2 and full_matrix.shape[1] == 2:
        return np.transpose([full_matrix[0, 0, ...],
                             full_matrix[1, 1, ...],
                             full_matrix[0, 1, ...] + full_matrix[1, 0, ...]])
    elif full_matrix.shape[0] == 3 and full_matrix.shape[1] == 3:
        return np.transpose([full_matrix[0, 0, ...],
                             full_matrix[1, 1, ...],
                             full_matrix[2, 2, ...],
                             full_matrix[1, 2, ...] + full_matrix[2, 1, ...],
                             full_matrix[0, 2, ...] + full_matrix[2, 0, ...],
                             full_matrix[0, 1, ...] + full_matrix[1, 0, ...]])
    else:
        raise RuntimeError("Invalid full_matrix. "
                           "The matrix in full notation should be"
                           "either 2x2 or 3x3.")


def Voigt_vector_to_full_matrix(voigt_vector, order="voigt"):
    """
    Takes a tensor in its Voigt notation and returns the corresponding tensor
    in its full notation.
                                       a₁₁ a₁₂ a₁₃
    (a₁₁, a₂₂, a₃₃, a₂₃, a₁₃, a₁₂)ᵀ ⇒ a₁₂ a₂₂ a₂₃ = A
                                       a₁₃ a₂₃ a₃₃

    Keyword Arguments:
    voigt_vector -- np.array of shape (3,) or (6,)
    order        -- string (default "voigt") can be set to "voigt", "nye" or
                    "esh3d" to handle different input vector orders:
                       "voigt"   (a₁₁,a₂₂,a₃₃,a₂₃,a₁₃,a₁₂)ᵀ (default)
                       "nye"     (a₁₁,a₂₂,a₃₃,a₁₂,a₁₃,a₂₃)ᵀ
                       "esh3d"   (a₁₁,a₂₂,a₃₃,a₁₂,a₂₃,a₁₃)ᵀ
                    Only for voigt_vector of the shape (6,).
    Return:
    A   Corresponding tensor in its full notation.
    """
    vv = voigt_vector
    if len(vv) == 3:
        A = np.array([[vv[0], vv[2]],
                      [vv[2], vv[1]]])
    elif len(vv) == 6:
        if order == "voigt":
            A = np.array([[vv[0], vv[5], vv[4]],
                          [vv[5], vv[1], vv[3]],
                          [vv[4], vv[3], vv[2]]])
        elif order == "nye":
            A = np.array([[vv[0], vv[3], vv[4]],
                          [vv[3], vv[1], vv[5]],
                          [vv[4], vv[5], vv[2]]])
        elif order == "esh3d":
            A = np.array([[vv[0], vv[3], vv[5]],
                          [vv[3], vv[1], vv[4]],
                          [vv[5], vv[4], vv[2]]])
        else:
            raise RuntimeError("Invalid order {}.\nThe following orders are"
                               "supported: 'voigt', 'nye' and 'esh3d'."
                               .format(order))
    else:
        raise RuntimeError("Invalid Voigt notation.\n"
                           "The vector should have either 3 or 6 elements. "
                           "But has the shape {}.".format(vv.shape))

    return A


def integrate_tensor_2_small_strain(strain, fft_engine, grid_spacing):
    """
    This function solves the following equation for obtaining the displacements
    in 2D (or extension for 3D):
     --    —-   —-  —-   —-    —-
     |k₁  0 |   | u₁ |   |  ε₁₁ |
    i|0   k₂| × |    | = |  ε₂₂ |    ***
     |k₂  k₁|   | u₂ |   | 2ε₁₂ |
     —-    —-   —-  —-   —-    —-
    which is overdetermined and least square will be utilized ofr solving.

    Keyword Arguments:
    grad           -- np.ndarray of shape [dim, dim] + nb_grid_pts_per_dim
                      containing the second-rank gradient to be integrated
    fft_engine     -- µFFT FFT object performing the FFT for a matrix on the
                      cell
    gradient_op    -- µSpectre DerivativeBase class representing the gradient
                      operator.
    grid_spacing   -- np.array of grid spacing in each spatial direction of
                      shape (dim,).

    Returns:
    np.ndarray containing the integrated field
    """
    dim = len(grid_spacing)
    nb_grid_pts = np.array(strain.shape[-dim:])
    lengths = nb_grid_pts * grid_spacing
    x = make_grid(lengths, nb_grid_pts)[0]
    # aplying Fourier transform on strain field
    strain_k_field = fft_engine.fetch_or_register_fourier_space_field(
        "strain_k", (dim, dim))
    fft_engine.fft(strain, strain_k_field)
    strain_k = strain_k_field.array()
    strain_k *= fft_engine.normalisation
    # wave vectors :
    wv = fft_engine.fftfreq
    # making shift vectors from the center of the gird to the corners
    shift = np.exp(-1j*np.pi*np.sum(wv, axis=0))
    strain_k_0 = strain_k[np.s_[:, :] + (0,)*(dim+1)]
    # Solving the *** equations (independently for each Fourier componenet)
    if dim == 2:
        # the periods (ω) of the wave vectors
        wf = 2 * np.pi * wv / (grid_spacing)[:, np.newaxis, np.newaxis]
        # constructing internal variables for the function (TODO:fix for 3d)
        u_k = np.zeros((dim,) + (int(nb_grid_pts[0]*0.5)+1, nb_grid_pts[1]),
                       dtype=np.cdouble)
        for i, j in itertools.product(range(int(nb_grid_pts[0]*0.5)+1),
                                      range(nb_grid_pts[1])):
            if (i != 0 or j != 0):
                # known matrix in the equation *** in 2D
                strain_k_vec_loc = np.squeeze(
                    full_matrix_to_Voigt_vector(strain_k[..., i, j]))

                # coefficient matrix in the equation *** in 2D
                A_loc = \
                    1j*(np.array([[wf[0, i, j], 0],
                                  [0, wf[1, i, j]],
                                  [wf[1, i, j], wf[0, i, j]]]))
                u_k[..., i, j] = \
                    np.linalg.solve(np.matmul(A_loc.T, A_loc),
                                    np.matmul(A_loc.T, strain_k_vec_loc))
    elif dim == 3:
        # the periods (ω) of the wave vectors
        wf = 2 * np.pi * wv / (grid_spacing)[:, np.newaxis,
                                             np.newaxis,
                                             np.newaxis]
        u_k = np.zeros((dim,) + (int(nb_grid_pts[0]*0.5)+1,
                                 nb_grid_pts[1], nb_grid_pts[2]),
                       dtype=np.cdouble)
        for i, j, k in itertools.product(range(int(nb_grid_pts[0]*0.5)+1),
                                         range(nb_grid_pts[1]),
                                         range(nb_grid_pts[2])):
            if (i != 0 or j != 0 or k != 0):
                # known matrix in the equation *** in 3D
                strain_k_vec_loc = np.squeeze(full_matrix_to_Voigt_vector(
                    strain_k[..., i, j, k]))
                # coefficient matrix in the equation *** in 3D
                A_loc = \
                    1j*np.array([[wf[0, i, j, k], 0, 0],
                                 [0,  wf[1, i, j, k], 0],
                                 [0, 0,  wf[2, i, j, k]],
                                 [0,  wf[2, i, j, k],  wf[1, i, j, k]],
                                 [wf[2, i, j, k], 0,  wf[0, i, j, k]],
                                 [wf[1, i, j, k],  wf[0, i, j, k], 0]])
                u_k[..., i, j, k] = \
                    np.linalg.solve(np.matmul(A_loc.T, A_loc),
                                    np.matmul(A_loc.T, strain_k_vec_loc))

    # shifting the results from the center of the grid to the corners
    u_k_shifted = np.zeros_like(u_k)
    for i, _grid_spacing in enumerate(grid_spacing):
        u_k_shifted[i, ...] = u_k[i, ...] * shift  # * _grid_spacing

    # Applying inverse Fourier transform to on=btain the displacement field in
    # Real space
    fluctuation_non_pbe = np.empty([dim, *fft_engine.nb_subdomain_grid_pts],
                                   order="f")
    fft_engine.ifft(u_k_shifted, fluctuation_non_pbe)
    if np.linalg.norm(fluctuation_non_pbe.imag) > 1e-10:
        raise RuntimeError("Integrate_tensor_2() computed complex placements, "
                           "probably there went something wrong.\n"
                           "Please inform the developers about this bug!")

    # adding and extra row/column due to periodic boundary condition
    fluctuation = complement_periodically(fluctuation_non_pbe.real, dim)

    # The homogeneous integration computes the affine part of the deformation
    homogeneous = np.einsum("ij,j...->i...", strain_k_0.real, x)
    return fluctuation + homogeneous


def integrate_vector(grad, fft_engine, gradient_op, grid_spacing):
    """Integrates a first-rank tensor gradient field, given on the center
    positions of the grid, by a compatible integration operator derived from
    the gradient operator. The integrated field is returned on the node
    positions.

    Keyword Arguments:
    grad           -- np.ndarray of shape [dim] + nb_grid_pts_per_dim
                      containing the first-rank tensor gradient to be
                      integrated.
    fft_engine        -- µFFT FFT object performing the FFT for a vector on the
                      cell
    gradient_op    -- µSpectre DerivativeBase class representing the gradient
                      operator.
    grid_spacing   -- np.array of grid spacing in each spatial direction of
                      shape (dim,).

    Returns:
    np.ndarray contaning the integrated field
    """
    dim = len(grid_spacing)
    nb_grid_pts = np.array(grad.shape[-dim:])
    lengths = nb_grid_pts * grid_spacing
    x = make_grid(lengths, nb_grid_pts)[0]
    integrator = get_integrator(fft_engine, gradient_op, grid_spacing)
    grad_k_field = fft_engine.register_fourier_space_field('grad_k', dim)
    fft_engine.fft(grad, grad_k_field)
    grad_k = grad_k_field.array()
    grad_k *= fft_engine.normalisation
    f_k = np.einsum("j...,j...->...", integrator, grad_k)
    grad_k_0 = grad_k[np.s_[:, ] + (0,)*(dim+1)]
    # The homogeneous integration computes the affine part of the deformation
    homogeneous = np.einsum("j,j...->...", grad_k_0.real, x)

    fluctuation_non_pbe = np.empty([1, *fft_engine.nb_subdomain_grid_pts],
                                   order="f")
    fft_engine.ifft(f_k, fluctuation_non_pbe)
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
    result      -- OptimiseResult, or just the gradient field from an
                   OptimizeResult.
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
    dim = len(nb_grid_pts)
    nb_der = len(gradient_op)

    # Check whether result is a np.array or an OptimiseResult object
    if isinstance(result, np.ndarray):
        if formulation is None:
            # exit the program, if the formulation is unknown!
            raise ValueError('\n'
                             'You have to specify your continuum mechanics'
                             'description.\n'
                             'Either you use a formulation="small_strain" or '
                             '"finite_strain" description.\n'
                             'Otherwise you can give a result=OptimiseResult'
                             ' object, which '
                             'tells me the formulation.')
        strain = result.reshape((dim, nb_der) + tuple(nb_grid_pts), order='F')
    else:
        form = result.formulation
        if form != formulation and formulation is not None:
            # exit the program, if the formulation is ambiguous!
            raise ValueError('\nThe given formulation "{}" differs from the '
                             'one saved in your result "{}"!'
                             .format(formulation, form))
        elif formulation is None:
            formulation = form
        strain = result.grad.reshape((dim, nb_der) + tuple(nb_grid_pts),
                                     order='F')

    # load or initialise muFFT.FFT engine
    if fft is None:
        fft_engine = muFFT.FFT(nb_grid_pts)
        fft_engine.create_plan(dim * dim)  # FFT for (dim,dim) matrix
        fft_engine.create_plan(dim)  # FFT for (dim) vector
    # compute the placement
    nodal_positions, _ = make_grid(lengths, nb_grid_pts)
    grid_spacing = np.array(lengths / nb_grid_pts)
    if formulation == Formulation.finite_strain:
        placement = integrate_tensor_2(strain, fft_engine,
                                       gradient_op, grid_spacing)
        return placement, nodal_positions
    elif formulation == Formulation.small_strain:
        displacement = integrate_tensor_2_small_strain(
            strain, fft_engine, grid_spacing)
        return displacement + nodal_positions, nodal_positions
    else:
        raise ValueError('\nThe formulation: "{}" is unknown!'
                         .format(formulation))
