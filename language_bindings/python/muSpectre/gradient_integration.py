#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   gradient_integration.py

@author Till Junge <till.junge@epfl.ch>

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
import sys
from . import Formulation

def compute_wave_vectors(lengths, nb_grid_pts):
    """Computes the wave vectors for a dim-dimensional rectangular or
    cubic (or hypercubic) grid as a function of the edge lengths and
    number of grid points.

    Note: the norm of the wave vectors corresponds to the angular
    velocity, not the frequency.

    Keyword Arguments:
    lengths     -- np.ndarray of length dim with the edge lengths in each
                   spatial dimension (dtype = float)
    nb_grid_pts -- np.ndarray of length dim with the nb_grid_pts in each
                   spatial dimension (dtype = int)

    Returns:
    np.ndarary of shape nb_grid_pts + [dim]. The wave vector for a
    given pixel/voxel is given in the last dimension

    """
    return np.moveaxis(np.meshgrid(
        *[2*np.pi*np.fft.fftfreq(r, l/r) for l,r in zip(lengths, nb_grid_pts)],
        indexing="ij"), 0, -1)

def compute_grid(lengths, nb_grid_pts):
    """For a dim-dimensional pixel/voxel grid, computes the pixel/voxel
    centre and corner positions as a function of the grid's edge
    lengths and number of grid points

    Keyword Arguments:
    lengths     -- np.ndarray of length dim with the edge lengths in each
                   spatial dimension (dtype = float)
    nb_grid_pts -- np.ndarray of length dim with the nb_grid_pts in each
                   spatial dimension (dtype = int)
    Returns:
    tuple((x_n, x_c)) two ndarrays with nodal/corner positions and
    centre positions respectively. x_n has one more entry in every
    direction than the number of grid points of the grid (added points
    correspond to the periodic repetitions)

    """
    x_n = np.moveaxis(np.meshgrid(
        *[np.linspace(0, l, r+1) for l,r in zip(lengths, nb_grid_pts)],
        indexing="ij"), 0 ,-1)
    dx = lengths/nb_grid_pts

    x_c = np.moveaxis(np.meshgrid(
        *[np.linspace(0, l, r, endpoint=False) for l,r in
          zip(lengths, nb_grid_pts)],
        indexing="ij"), 0, -1) + .5*dx

    return x_n, x_c

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


def get_integrator(x, freqs, order=0):
    """returns the discrete Fourier-space integration operator as a
    function of the position grid (used to determine the spatial
    dimension and number of grid points), the wave vectors, and the integration
    order

    Keyword Arguments:
    x     -- np.ndarray of pixel/voxel centre positons in shape
             nb_grid_pts_per_dim + [dim]
    freqs -- wave vectors as computed by compute_wave_vectors
    order -- (default 0) integration order. 0 stands for exact integration

    Returns:
    (dim, shape, integrator)
    """
    dim = x.shape[-1]
    shape = x.shape[:-1]
    delta_x = x[tuple([1]*dim)] - x[tuple([0]*dim)]
    def correct_denom(denom):
        """Corrects the denominator to avoid division by zero
        for freqs = 0 and on even grids for freqs*delta_x=pi."""
        denom[tuple([0]*dim)] = 1.
        for i, n in enumerate(shape):
            if n%2 == 0:
                denom[tuple([np.s_[:]]*i + [n//2] + [np.s_[:]]*(dim-1-i))] = 1.
        return denom[..., np.newaxis]
    if order == 0:
        freqs_norm_square = np.einsum("...i,...i ->...", freqs, freqs)
        freqs_norm_square.reshape(-1)[0] = 1.
        integrator = 1j * freqs/freqs_norm_square[...,np.newaxis]
    # Higher order corrections after:
    # A. Vidyasagar et al., Computer Methods in Applied Mechanics and
    # Engineering 106 (2017) 133-151, sec. 3.4 and Appendix C.
    # and
    # P. Eisenlohr et al., International Journal of Plasticity 46 (2013)
    # 37-53, Appendix B, eq. 23.
    elif order == 1:
        sin_1 = np.sin(freqs*delta_x)
        denom = correct_denom((sin_1**2).sum(axis=-1))
        integrator = 1j*delta_x*sin_1 / denom
    elif order == 2:
        sin_1, sin_2 = 8*np.sin(freqs*delta_x), np.sin(2*freqs*delta_x)
        denom = correct_denom((sin_1**2).sum(axis=-1) - (sin_2**2).sum(axis=-1))
        integrator = 1j*6*delta_x*(sin_1+sin_2) / denom
    else:
        raise Exception("order '{}' is not implemented".format(order))
    return dim, shape, integrator


def integrate_tensor_2(grad, x, freqs, staggered_grid=False, order=0):
    """Integrates a second-rank tensor gradient field to a chosen order as
    a function of the given field, the grid positions, and wave
    vectors. Optionally, the integration can be performed on the
    pixel/voxel corners (staggered grid).

    Keyword Arguments:
    grad           -- np.ndarray of shape nb_grid_pts_per_dim + [dim, dim]
                      containing the second-rank gradient to be integrated
    x              -- np.ndarray of shape nb_grid_pts_per_dim + [dim] (or
                      augmented nb_grid_pts_per_dim + [dim]) containing the
                      pixel/voxel centre positions (for un-staggered grid
                      integration) or the pixel/voxel corner positions (for
                      staggered grid integration)
    freqs          -- wave vectors as computed by compute_wave_vectors
    staggered_grid -- (default False) if set to True, the integration is
                      performed on the pixel/voxel corners, rather than the
                      centres. This leads to a different integration scheme
    order          -- (default 0) integration order.
                      0 stands for exact integration

    Returns:
    np.ndarray contaning the integrated field
    """

    dim, shape, integrator = get_integrator(x, freqs, order)

    axes = range(dim)
    grad_k = np.fft.fftn(grad, axes=axes)
    f_k = np.einsum("...j,...ij ->...i", integrator, grad_k)
    normalisation = np.prod(grad.shape[:dim])
    grad_k_0 = grad_k[tuple((0 for _ in range(dim)))].real/normalisation
    homogeneous = np.einsum("ij,...j ->...i",
                            grad_k_0, x)
    if not staggered_grid:
        fluctuation = -np.fft.ifftn(f_k, axes=axes).real
    else:
        del_x = (x[tuple((1 for _ in range(dim)))] -
                 x[tuple((0 for _ in range(dim)))])
        k_del_x = np.einsum("...i, i ->...", freqs, del_x)[...,np.newaxis]
        if order == 0:
            shift = np.exp(-1j * k_del_x/2)
        elif order == 1:
            shift = (np.exp(-1j * k_del_x) + 1) / 2
        elif order == 2:
            shift = np.exp(-1j*k_del_x/2) * np.cos(k_del_x/2) *\
                    (np.cos(k_del_x) - 4) / (np.cos(k_del_x/2) - 4)
        fluctuation = complement_periodically(
            -np.fft.ifftn(shift*f_k, axes=axes).real, dim)

    return fluctuation + homogeneous


def integrate_vector(df, x, freqs, staggered_grid=False, order=0):
    """Integrates a first-rank tensor gradient field to a chosen order as
    a function of the given field, the grid positions, and wave
    vectors. Optionally, the integration can be performed on the
    pixel/voxel corners (staggered_grid)

    Keyword Arguments:
    df             -- np.ndarray of shape nb_grid_pts_per_dim + [dim] containing
                      the first-rank tensor gradient to be integrated
    x              -- np.ndarray of shape nb_grid_pts_per_dim + [dim] (or
                      augmented nb_grid_pts_per_dim + [dim]) containing the
                      pixel/voxel centre positions (for un-staggered grid
                      integration) or the pixel/voxel corner positions (for
                      staggered grid integration)
    freqs          -- wave vectors as computed by compute_wave_vectors
    staggered_grid -- (default False) if set to True, the integration is
                      performed on the pixel/voxel corners, rather than the
                      centres. This leads to a different integration scheme...
    order          -- (default 0) integration order.
                      0 stands for exact integration

    Returns:
    np.ndarray contaning the integrated field
    """
    dim, shape, integrator = get_integrator(x, freqs, order)

    axes = range(dim)
    df_k = np.fft.fftn(df, axes=axes)
    f_k = np.einsum("...i,...i ->...", df_k, integrator)
    df_k_0 = df_k[tuple((0 for _ in range(dim)))].real
    homogeneous = np.einsum("i,...i ->...", df_k_0, x/np.prod(shape))

    if not staggered_grid:
        fluctuation = -np.fft.ifftn(f_k, axes=axes).real
    else:
        del_x = x[tuple((1 for _ in range(dim)))] \
                - x[tuple((0 for _ in range(dim)))]
        k_del_x = np.einsum("...i, i ->...", freqs, del_x)
        if order == 0:
            shift = np.exp(-1j * k_del_x/2)
        elif order == 1:
            shift = (np.exp(-1j * k_del_x) + 1) / 2
        elif order == 2:
            shift = np.exp(-1j*k_del_x/2) * np.cos(k_del_x/2) *\
                    (np.cos(k_del_x) - 4) / (np.cos(k_del_x/2) - 4)
        fluctuation = complement_periodically(
            -np.fft.ifftn(shift*f_k, axes=axes).real, dim)

    return fluctuation + homogeneous


def compute_placement(result, lengths, nb_grid_pts, order=0, formulation=None):
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
    order       -- (default 0) integration order. 0 stands for exact integration
    formulation -- (default None) the formulation is derived from the
                   OptimiseResult argument. If this is not possible you have to
                   fix the formulation to either Formulation.small_strain or
                   Formulation.finite_strain.
    Returns:
    (placement, x_n) tuple of ndarrays containing the placement and the
                     corresponding original positions

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

    #compute the placement by integrating
    x_n, x_c = compute_grid(lengths, nb_grid_pts)
    freqs = compute_wave_vectors(lengths, nb_grid_pts)
    placement = integrate_tensor_2(grad, x_n, freqs,
                                   staggered_grid=True, order=order)

    return placement, x_n
