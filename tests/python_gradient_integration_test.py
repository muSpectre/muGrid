#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_gradient_integration_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   23 Nov 2018

@brief  test the functionality of gradient_integration.py

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

from python_test_imports import µ
import muGrid
import muFFT
from muFFT import Stencils2D

import unittest
import numpy as np
import itertools
import time

### Helper functions
def init_X_F_Chi(lens, res, rank=2):
    """
    Setup all the needed parameters for initialization of the deformation
    gradient F and the corresponding deformation map/field Chi_X.

    Keyword Arguments:
    lens -- list [Lx, Ly, ...] of box lengths in each direction (dtype=float)
    res  -- list [Nx, Ny, ...] of grid resoultions (dtype = int)
    rank -- int (default=2), rank of the deformation gradient tensor F.
            (dtype = int)

    Returns:
    delta_x : np.array of grid spacing for each spatial direction (dtype = float)
    dim  : int dimension of the structure, derived from len(res).
    x_n  : np.ndarray shape=(res.shape+1, dim) initial nodal/corner positions
           as created by gradient_integration.compute_grid (dtype = float)
    x_c  : np.ndarray shape=(res.shape+1, dim) initial cell center positions
           as created by gradient_integration.compute_grid (dtype = float)
    F    : np.zeros shape=(res.shape, dim*rank) initialise deformation gradient
           (dtype = float)
    Chi_n: np.zeros shape=((res+1).shape, dim) initialise deformation field
           (dtype = float)
    freqs: np.ndarray as returned by compute_wave_vectors(). (dtype = float)
    """
    lens = np.array(lens)
    res = np.array(res)
    delta_x = lens / res
    dim = len(res)
    x_n, x_c = µ.gradient_integration.make_grid(lens, res)
    F = np.zeros((dim,)*(rank) + x_c.shape[1:])
    Chi_n = np.zeros(x_n.shape)

    return delta_x, dim, x_n, x_c, F, Chi_n

class GradientIntegration_Check(unittest.TestCase):
    """
    Check the implementation of all muSpectre.gradient_integration functions.
    """

    def setUp(self):
        self.lengths = np.array([2.4, 3.7, 4.1])
        self.nb_grid_pts = np.array([5, 3, 5])
        self.norm_tol = 1e-8

        # set timing = True for timing information
        self.timing = False
        self.startTime = time.time()

    def tearDown(self):
        if self.timing:
            t = time.time() - self.startTime
            print("{}:\n{:.3f} seconds".format(self.id(), t))

    def test_make_grid(self):
        """
        Test the function compute_grid which creates an orthogonal
        equally spaced grid of the given number of grid points in each dimension
        and the corresponding  lengths.
        """
        lens = self.lengths
        res = self.nb_grid_pts
        d = np.array(lens)/np.array(res)
        grid_n = np.zeros((len(res),) + tuple(res+1))
        Nx, Ny, Nz = res+1
        for i, j, k in itertools.product(range(Nx), range(Ny), range(Nz)):
            grid_n[:, i, j, k] = np.array([i*d[0], j*d[1], k*d[2]])
        grid_c = (grid_n - d.reshape((3, 1, 1, 1))/2)[:, 1:, 1:, 1:]
        for n in range(1, 4):
            x_n, x_c = µ.gradient_integration.make_grid(lens[:n], res[:n])
            s = (np.s_[:n],) + (np.s_[:],)*n + (0,)*(3-n)
            self.assertLess(np.linalg.norm(x_c - grid_c[s]), self.norm_tol)
            self.assertLess(np.linalg.norm(x_n - grid_n[s]), self.norm_tol)

    def test_complement_periodically(self):
        """
        Test the periodic reconstruction of an array. Lower left entries are
        added into the upper right part of the array.
        """
        # 1D grid scalars
        x_test = np.array([0, 1, 2, 3])
        x_test_p = np.array([0, 1, 2, 3, 0])
        x_p = µ.gradient_integration.complement_periodically(x_test, 1)
        self.assertLess(np.linalg.norm(x_p-x_test_p), self.norm_tol)

        # 2D grid scalars
        x_test = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8]])
        x_test_p = np.array([[1, 2, 3, 4, 1],
                             [5, 6, 7, 8, 5],
                             [1, 2, 3, 4, 1]])
        x_p = µ.gradient_integration.complement_periodically(x_test, 2)
        self.assertLess(np.linalg.norm(x_p-x_test_p), self.norm_tol)

        # 2D grid vectors
        x_test = np.array([[[1, 2, 3], [3, 4, 5]],
                           [[6, 7, 8], [9, 10, 11]],
                           [[12, 13, 14], [15, 6, 17]]])
        x_test_p = np.array([[[1, 2, 3], [3, 4, 5], [1, 2, 3]],
                             [[6, 7, 8], [9, 10, 11], [6, 7, 8]],
                             [[12, 13, 14], [15, 6, 17], [12, 13, 14]],
                             [[1, 2, 3], [3, 4, 5], [1, 2, 3]]])
        x_p = µ.gradient_integration.complement_periodically(
            np.moveaxis(x_test, -1, 0), 2)
        self.assertLess(np.linalg.norm(x_p - np.moveaxis(x_test_p, -1, 0)),
                        self.norm_tol)

    def test_get_integrator(self):
        """
        Test if the right integrator is computed.
        """
        # Init:
        # even grid
        lens_e = np.array([1, 1, 1])
        res_e = np.array([2, 2, 2])
        delta_x_e = lens_e/res_e
        x_n_e, x_c_e = µ.gradient_integration.make_grid(lens_e, res_e)
        # odd grid
        lens_o = np.array([1, 1])
        res_o = np.array([3, 3])
        delta_x_o = lens_o/res_o
        x_n_o, x_c_o = µ.gradient_integration.make_grid(lens_o, res_o)

        # Fourier Derivative:
        # even grid
        dim = len(res_e)
        fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
        fft_engine = muFFT.FFT(list(res_e))

        #freqs = fft_engine.wavevectors(lens_e)
        freqs = fft_engine.fftfreq * res_e.reshape((dim,)+(1,)*dim)
        shift = np.exp(-1j*2*np.pi *
                       np.einsum("i...,i->...", freqs, delta_x_e/2))
        # analytic solution -i*q/|q|^2 * shift
        int_ana = 1j/(2*np.pi)*np.array([[[[0,   0,   0], [0,   0,   1]],
                                          [[0,   1,   0], [0, 1/2, 1/2]]],
                                         [[[1,   0,   0], [1/2,   0, 1/2]],
                                          [[1/2, 1/2,   0], [1/3, 1/3, 1/3]]]])\
            .transpose((3, 0, 1, 2)) * shift
        integrator = µ.gradient_integration.get_integrator(
            fft_engine, fourier_gradient, delta_x_e)
        self.assertLess(np.linalg.norm(integrator-int_ana), self.norm_tol)

        # odd grid
        dim = len(res_o)
        fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
        fft_engine = muFFT.FFT(list(res_o))
        freqs = fft_engine.fftfreq * res_o.reshape((dim,)+(1,)*dim)
        shift = np.exp(-1j*2*np.pi *
                       np.einsum("i...,i->...", freqs, delta_x_o/2))
        # analytic solution -i*q/|q|^2 * shift
        int_ana = -1j/(2*np.pi) *\
            np.array([[[0, 0, 0], [1, 1/2, 1/2]],
                      [[0, 1, -1], [0, 1/2, -1/2]]]) * shift[np.newaxis, :, :]
        integrator = µ.gradient_integration.get_integrator(
            fft_engine, fourier_gradient, delta_x_o)
        self.assertLess(np.linalg.norm(integrator-int_ana), self.norm_tol)

        # Discrete Derivatives:
        # odd grid
        dim = len(res_o)
        dy = muFFT.DiscreteDerivative([0, 0], [[-0.5, 0.5],
                                               [-0.5, 0.5]])
        dx = dy.rollaxes(-1)
        discrete_gradient = [dx, dy]
        fft_engine = muFFT.FFT(list(res_o))
        integrator = µ.gradient_integration.get_integrator(fft_engine,
                                                           discrete_gradient,
                                                           delta_x_o)
        int_ana = np.array(
            [[[0. + 0.j,  0. - 0.j,  0. - 0.j],
              [-0.16666667-0.09622504j, -0.16666667+0.09622504j,  0. - 0.19245009j]],
             [[0. + 0.j, -0.16666667-0.09622504j, -0.16666667+0.09622504j],
              [0. - 0.j, -0.16666667+0.09622504j,  0. + 0.19245009j]]])
        self.assertLess(np.linalg.norm(integrator-int_ana), 1e-7)


    def test_fourier_integrate_tensor_2(self):
        """
        Test the correct integration of a second-rank tensor gradient field,
        like the deformation gradient, using fourier integration.
        """
        # cosinus, diagonal deformation gradient 2D
        res = np.array([36, 14])
        lens = np.array([7, 1.4])
        delta_x, dim, x_n, x_c, F, u_n = init_X_F_Chi(lens, res)
        for i in range(dim):
            F[i, i, :, :] = 0.8*np.pi/lens[i]*np.cos(2*np.pi * x_c[i]/lens[i])
        Chi_n = 0.4 * np.sin(2*np.pi*x_n/lens.reshape((dim,)+(1,)*dim))

        # Fourier Derivative
        fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
        fft_engine = muFFT.FFT(list(res))
        fft_engine.create_plan(dim)
        fft_engine.create_plan(dim * dim)
        placement_n = µ.gradient_integration.integrate_tensor_2(
            F, fft_engine, fourier_gradient, delta_x)

        self.assertLess(np.linalg.norm(Chi_n - placement_n), self.norm_tol)


        # cosinus, diagonal deformation gradient 3D
        res = np.array([36, 14, 15])
        fft_engine = muFFT.FFT(list(res)) # new engine is now 3d
        lens = np.array([7, 1.4, 3])
        delta_x, dim, x_n, x_c, F, _ = init_X_F_Chi(lens, res)
        for i in range(dim):
            F[i, i, :, :, :] = 0.8*np.pi/lens[i] * \
                np.cos(2*np.pi * x_c[i]/lens[i])
        Chi_n = 0.4 * np.sin(2*np.pi*x_n/lens.reshape((dim,)+(1,)*dim))

        # Fourier Derivative
        fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]

        fft_engine.create_plan(dim)
        fft_engine.create_plan(dim * dim)
        placement_n = µ.gradient_integration.integrate_tensor_2(
            F, fft_engine, fourier_gradient, delta_x)

        self.assertLess(np.linalg.norm(Chi_n - placement_n), self.norm_tol)

        # cosinus, non diagonal deformation gradient
        res = [31, 19, 13]
        lens = [7, 1.4, 3]
        delta_x, dim, x_n, x_c, F, Chi_n = init_X_F_Chi(lens, res)

        F[0, 0, :, :, :] = 4*np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])
        F[1, 1, :, :, :] = 2*np.pi/lens[1]*np.cos(2*np.pi/lens[1]*x_c[1])
        F[2, 2, :, :, :] = 2*np.pi/lens[2]*np.cos(2*np.pi/lens[2]*x_c[2])
        F[1, 0, :, :, :] = 2*np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])
        F[2, 0, :, :, :] = 2*np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])
        for i in range(dim):
            Chi_n[i, :, :, :] = np.sin(2*np.pi*x_n[i]/lens[i])  \
                + np.sin(2*np.pi*x_n[0]/lens[0])

        # Fourier Derivative
        fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]

        fft_engine = muFFT.FFT(res)
        fft_engine.create_plan(dim)
        fft_engine.create_plan(dim * dim)
        placement_n = µ.gradient_integration.integrate_tensor_2(
            F, fft_engine, fourier_gradient, delta_x)
        self.assertLess(np.linalg.norm(Chi_n - placement_n), self.norm_tol)

    def test_shear_composite(self):
        # Realistic test:
        #   shear of a two dimensional material with two different Young moduli.
        # initialize material structure
        res = [9, 21]  # nb_grid_pts
        lens = [9, 21]  # lengths
        delta_x, dim, x_n, x_c, _, _ = init_X_F_Chi(lens, res)
        formulation = µ.Formulation.finite_strain
        Young = [10, 20]  # Youngs modulus for each phase (soft, hard)
        Poisson = [0.3, 0.3]  # Poissons ratio for each phase

        # geometry (two slabs stacked in y-direction with,
        # hight h (soft material) and hight res[1]-h (hard material))
        h = res[1]//2
        phase = np.zeros(tuple(res), dtype=int)
        phase[:, h:] = 1
        phase = phase.T.flatten()
        cell = µ.Cell(res, lens, formulation)
        mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
        for i, pixel in enumerate(cell.pixels):
            mat.add_pixel(i, Young[phase[i]], Poisson[phase[i]])
        cell.initialise()
        DelF = np.array([[0, 0.01],
                         [0, 0   ]])

        # µSpectre solution
        solver = µ.solvers.KrylovSolverCG(cell, tol=1e-6, maxiter=100,
                                          verbose=µ.Verbosity.Silent)
        result = µ.solvers.newton_cg(cell, DelF, solver,
                                     newton_tol=1e-6, equil_tol=1e-6,
                                     verbose=µ.Verbosity.Silent)
        F = cell.strain.array(muGrid.IterUnit.Pixel)

        # muSpectre Fourier integration
        fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
        fft_engine = muFFT.FFT(res)
        fft_engine.create_plan(dim)
        fft_engine.create_plan(dim * dim)

        placement_n = µ.gradient_integration.integrate_tensor_2(
            F, fft_engine, fourier_gradient, delta_x)
        # muSpectre "discrete" integration (forward upwind scheme)
        dy = muFFT.DiscreteDerivative([0, 0], [[-0.5, 0.5],
                                               [-0.5, 0.5]])
        dx = dy.rollaxes(-1)
        discrete_gradient = [dx, dy]
        placement_n_disc = µ.gradient_integration.integrate_tensor_2(
            F, fft_engine, discrete_gradient, delta_x)

        # analytic solution, "placement_ana" (node and center)
        l_soft = delta_x[1] * h  # height soft material
        l_hard = delta_x[1] * (res[1]-h)  # height hard material
        Shear_modulus = np.array(Young) / (2 * (1+np.array(Poisson)))
        mean_shear_strain = 2*DelF[0, 1]
        shear_strain_soft = (lens[1]*mean_shear_strain) / (l_soft
                            + l_hard * Shear_modulus[0]/Shear_modulus[1])
        shear_strain_hard = (lens[1]*mean_shear_strain) / (l_soft
                            * Shear_modulus[1]/Shear_modulus[0] + l_hard)
        placement_ana_n = np.zeros(x_n.shape)
        placement_ana_c = np.zeros(x_c.shape)

        # x-coordinate
        # soft material
        placement_ana_n[0, :, :h+1] = shear_strain_soft/2 * x_n[1, :, :h+1]
        placement_ana_c[0, :, :h] = shear_strain_soft/2 * x_c[1, :, :h]
        # hard material
        placement_ana_n[0, :, h+1:] =\
            (shear_strain_hard/2 * (x_n[1, :, h+1:]-l_soft) +
             shear_strain_soft/2 * l_soft)
        placement_ana_c[0, :, h:] = \
            (shear_strain_hard/2 * (x_c[1, :, h:]-l_soft) +
             shear_strain_soft/2 * l_soft)
        # y-coordinate
        placement_ana_n[1, :, :] = 0
        placement_ana_c[1, :, :] = 0

        # shift the analytic solution such that the average nonaffine deformation
        # is zero (integral of the nonaffine deformation gradient + N*const != 0)
        F_homo = (1./(np.prod(res)) * F.sum(axis=tuple(-(np.arange(dim)+1))
                                            )).reshape((dim,)*2 + (1,)*dim)
        # integration constant = integral of the nonaffine deformation gradient/N
        int_const =\
            - ((placement_ana_c[0, :, :] - F_homo[0, 1, :, :] * x_c[1, :, :])
                       .sum(axis=1))[0] / res[1]
        ana_sol_n = placement_ana_n + x_n + \
            np.array([int_const, 0]).reshape((dim,) + (1,)*dim)

        # check the numeric vs the analytic solution
        norm_n = \
            (np.linalg.norm(placement_n - ana_sol_n)/
             np.prod(np.array(res)))
        self.assertLess(norm_n, 1.17e-5)
        norm_n_disc = \
            (np.linalg.norm(placement_n_disc - ana_sol_n)/
             np.prod(np.array(res)))
        self.assertLess(norm_n_disc, 3.89e-6)



    def test_fourier_integrate_tensor_2_small_strain(self):
        """
        Test the correct integration of a second-rank tensor gradient field,
        like the deformation gradient, using fourier integration.
        """
        strain_amp = 1e-4
        # cosinus, diagonal deformation gradient 2D
        res = np.array([35, 19])
        lens = np.array([0.53, 1.42])
        delta_x, dim, x_n, x_c, E, _ = init_X_F_Chi(lens, res)
        for i in range(dim):
            E[i, i, :, :] = \
                (strain_amp * (2*np.pi/lens[i])
                 * np.cos(2*np.pi * x_c[i]/lens[i]))
            u_n =\
                strain_amp * np.sin(2*np.pi*x_n/(lens.reshape((dim,)+(1,)*dim)))
        fft_engine = muFFT.FFT(list(res))
        fft_engine.create_plan(dim)
        fft_engine.create_plan(dim * dim)
        u_integrated_n = µ.gradient_integration.integrate_tensor_2_small_strain(
            E, fft_engine, delta_x)
        self.assertLess(np.linalg.norm(u_n - u_integrated_n), self.norm_tol)

        #### Non Diagonal 2D:
        delta_x, dim, x_n, x_c, E, u_n = init_X_F_Chi(lens, res)
        for i in range(dim):
            u_n[i, :, :] = strain_amp * (np.sin(2*np.pi*x_n[i]/lens[i]) +
                                         np.sin(2*np.pi*x_n[0]/lens[0]))
        E[0, 0, :, :] = \
            strain_amp * 4 * np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])
        E[1, 1, :, :] =\
            strain_amp * 2 * np.pi/lens[1]*np.cos(2*np.pi/lens[1]*x_c[1])
        E[1, 0, :, :] =\
            strain_amp * np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])
        E[0, 1, :, :] =\
            strain_amp * np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])

        u_integrated_n = µ.gradient_integration.integrate_tensor_2_small_strain(
            E, fft_engine, delta_x)
        self.assertLess(np.linalg.norm(u_n - u_integrated_n), self.norm_tol)

        # cosinus, diagonal deformation gradient 3D
        res = np.array([37, 13, 15])
        lens = np.array([7, 1.4, 3])
        delta_x, dim, x_n, x_c, E, _ = init_X_F_Chi(lens, res)
        for i in range(dim):
            E[i, i, :, :] = \
                (strain_amp * (2*np.pi/lens[i])
                 *np.cos(2*np.pi * x_c[i]/lens[i]))
            u_n = \
                strain_amp * np.sin(2*np.pi*x_n/(lens.reshape((dim,)+(1,)*dim)))
        fft_engine = muFFT.FFT(list(res))
        fft_engine.create_plan(dim)
        fft_engine.create_plan(dim*dim)
        u_integrated_n = µ.gradient_integration.integrate_tensor_2_small_strain(
            E, fft_engine, delta_x)
        self.assertLess(np.linalg.norm(u_n - u_integrated_n), self.norm_tol)

        #### Non Diagonal 2D:
        delta_x, dim, x_n, x_c, E, u_n = init_X_F_Chi(lens, res)
        for i in range(dim):
            u_n[i, :, :] = strain_amp * (np.sin(2*np.pi*x_n[i]/lens[i]) +
                                         np.sin(2*np.pi*x_n[0]/lens[0]))
        E[0, 0, :, :] = \
            strain_amp * 4 * np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])
        E[1, 1, :, :] =\
            strain_amp * 2 * np.pi/lens[1]*np.cos(2*np.pi/lens[1]*x_c[1])
        E[2, 2, :, :] =\
            strain_amp * 2 * np.pi/lens[2]*np.cos(2*np.pi/lens[2]*x_c[2])
        E[1, 0, :, :] =\
            strain_amp * np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])
        E[0, 1, :, :] =\
            strain_amp * np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])
        E[0, 2, :, :] =\
            strain_amp * np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])
        E[2, 0, :, :] =\
            strain_amp * np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[0])

        u_integrated_n = µ.gradient_integration.integrate_tensor_2_small_strain(
            E, fft_engine, delta_x)
        self.assertLess(np.linalg.norm(u_n - u_integrated_n), self.norm_tol)


    def test_discrete_integrate_tensor_2(self):
        """
        Test the correct integration of a second-rank tensor gradient field,
        like the deformation gradient, using discrete integration.
        """
        F0 = np.array([[1.1, 0.2, 0.0],
                       [0.3, 1.2, 0.1],
                       [0.1, 0.0, 0.9]])
        res = [23, 45, 11]
        lens = [1.4, 2.3, 1.1]
        dim = len(res)
        delta_x = [l/r for l, r in zip(lens, res)]
        # Create a random displacement field
        x = (((np.random.random([len(res)]+res)).T-0.5)*delta_x).T
        for i in range(dim):
            x[i] -= x[i].mean()  # mean of random field should be zero
        x = µ.gradient_integration.complement_periodically(x, 3)
        # Create grid positions
        nodal_positions, center_positions = \
            µ.gradient_integration.make_grid(np.array(lens), np.array(res))
        # The displacement field lives on the corners
        x += np.einsum('ij,jxyz->ixyz', F0, nodal_positions)
        # Deformation gradient
        F = np.zeros(2*[dim] + res)
        for i in range(dim):
            for j in range(dim):
                F[j, i, :, :, :] = (
                    np.roll(x[j], -1, axis=i) - x[j])[:-1, :-1, :-1]/delta_x[i]

        self.assertTrue(np.allclose(np.mean(F, axis=(2, 3, 4)), F0))

        fft_engine = muFFT.FFT(res)
        fft_engine.create_plan(dim)
        fft_engine.create_plan(dim * dim)
        dz = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]])
        dy = dz.rollaxes(-1)
        dx = dy.rollaxes(-1)
        discrete_gradient = [dx, dy, dz]
        placement_c = µ.gradient_integration.integrate_tensor_2(
            F, fft_engine, discrete_gradient, delta_x)

        for i in range(dim):
            self.assertTrue(np.allclose(x[i], placement_c[i, :, :, :]))

    def test_discrete_integrate_vector_2d_no_homogeneous(self):
        """
        Test the correct integration of a first-rank tensor gradient field,
        like the electrostatic field, using discrete integration. The mean
        gradient for this test is zero.
        """
        res = [23, 45]
        lens = [1.4, 2.3]
        dim = len(res)
        delta_x = [l/r for l, r in zip(lens, res)]
        # Create a random displacement field
        x = np.random.random(res)-0.5
        x -= x.mean()
        # Create grid positions
        x_n, x_c = \
            µ.gradient_integration.make_grid(np.array(lens), np.array(res))
        # Gradient
        g = np.zeros([dim] + res)
        for i in range(dim):
            g[i, :, :] = (np.roll(x, -1, axis=i) - x)/delta_x[i]

        fft_engine = muFFT.FFT(res)
        fft_engine.create_plan(dim**0)
        fft_engine.create_plan(dim**1)
        dy = muFFT.DiscreteDerivative([0, 0], [[-1, 1]])
        dx = dy.rollaxes(-1)
        discrete_gradient = [dx, dy]
        int_x = µ.gradient_integration.integrate_vector(
            g, fft_engine, discrete_gradient, delta_x)

        self.assertTrue(np.allclose(x, int_x[0,:-1, :-1]))

    def test_discrete_integrate_vector_3d(self):
        """
        Test the correct integration of a first-rank tensor gradient field,
        like the electrostatic field, using discrete integration.
        """
        F0 = np.array([1.1, 0.2, 0.7])
        res = [23, 45, 17]
        lens = [1.4, 2.3, 1.7]
        res1 = [r+1 for r in res]
        dim = len(res)
        delta_x = [l/r for l, r in zip(lens, res)]
        # Create a random displacement field
        x = np.random.random(res)-0.5
        x -= x.mean()  # mean of random field should be zero
        x = µ.gradient_integration.complement_periodically(x, 3)
        # Create grid positions
        nodal_positions, center_positions = \
            µ.gradient_integration.make_grid(np.array(lens), np.array(res))
        x += np.einsum('j,jxyz->xyz', F0, nodal_positions)
        # Gradient
        g = np.zeros([dim] + res)
        for i in range(dim):
            g[i, :, :, :] = (np.roll(x, -1, axis=i) -
                             x)[:-1, :-1, :-1]/delta_x[i]

        fft_engine = muFFT.FFT(res)
        fft_engine.create_plan(dim**0)
        fft_engine.create_plan(dim**1)
        dz = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]])
        dy = dz.rollaxes(-1)
        dx = dy.rollaxes(-1)
        discrete_gradient = [dx, dy, dz]
        int_x = µ.gradient_integration.integrate_vector(
            g, fft_engine, discrete_gradient, delta_x)

        self.assertTrue(np.allclose(x, int_x))

    def test_compute_placement(self):
        """Test the computation of placements and the original positions."""
        ### shear of a homogeneous material ###
        res = [3, 11]  # nb_grid_pts
        lens = [10, 10]  # lengths
        dim = len(res)  # dimension
        x_n = µ.gradient_integration.make_grid(np.array(lens), np.array(res))[0]

        # finite strain
        formulation = µ.Formulation.finite_strain
        cell = µ.Cell(res, lens, formulation)
        mat = µ.material.MaterialLinearElastic1_2d.make(cell, "material",
                                                        Young=10, Poisson=0.3)
        for pixel_id in cell.pixel_indices:
            mat.add_pixel(pixel_id)
        cell.initialise()
        DelF = np.array([[0, 0.05],
                         [0, 0   ]])
        # analytic
        placement_ana = np.copy(x_n)
        placement_ana[0, :, :] += DelF[0, 1]*x_n[1, :, :]

        # µSpectre solution
        fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
        solver = µ.solvers.KrylovSolverCG(cell, tol=1e-6, maxiter=100,
                                          verbose=µ.Verbosity.Silent)
        result = µ.solvers.newton_cg(cell, DelF, solver, newton_tol=1e-6,
                                     equil_tol=1e-6,
                                     verbose=µ.Verbosity.Silent)
        for rr in [(cell.strain.array(), 'using cell.strain'),
                   (result, 'using OptimizeResult'),
                   (result.grad, 'using gradient')]:
            r, msg = rr
            # check input of result=OptimiseResult and result=np.ndarray
            placement, x = µ.gradient_integration.compute_placement(
                r, lens, res, fourier_gradient,
                formulation=µ.Formulation.finite_strain)
            self.assertLess(np.linalg.norm(placement_ana -
                                           placement), 1e-12, msg=msg)
            self.assertTrue((x_n == x).all(), msg=msg)


    def test_compute_placement_small_strain(self):
        """Test the computation of placements and the original positions."""
        ### shear of a homogeneous material ###
        res = [3, 11]  # nb_grid_pts
        lens = [10, 12]  # lengths
        dim = len(res)  # dimension
        x_n = µ.gradient_integration.make_grid(np.array(lens), np.array(res))[0]

        # finite strain
        formulation = µ.Formulation.small_strain
        cell = µ.Cell(res, lens, formulation)
        mat = µ.material.MaterialLinearElastic1_2d.make(cell, "material",
                                                        Young=10, Poisson=0.3)
        for pixel_id in cell.pixel_indices:
            mat.add_pixel(pixel_id)
        cell.initialise()
        DelE = 1e-4 * np.array([[0   , 0.05],
                                [0.05, 0   ]])

        # analytic
        placement_ana = np.copy(x_n)
        placement_ana[0, :, :] += DelE[0, 1] * x_n[1, :, :]
        placement_ana[1, :, :] += DelE[1, 0] * x_n[0, :, :]

        # µSpectre solution
        fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
        solver = \
            µ.solvers.KrylovSolverCG(cell, tol=1e-6, maxiter=100,
                                     verbose=µ.Verbosity.Silent)
        result = µ.solvers.newton_cg(cell, DelE, solver,
                                     newton_tol=1e-6, equil_tol=1e-6,
                                     verbose=µ.Verbosity.Silent)
        for r in [result, result.grad]:
            # check input of result=OptimiseResult and result=np.ndarray
            u_n, x = µ.gradient_integration.compute_placement(
                r, lens, res, fourier_gradient,
                formulation=µ.Formulation.small_strain)
            self.assertLess(np.linalg.norm(placement_ana -
                                           u_n), 1e-12)
            self.assertTrue((x_n == x).all())


    def test_compare_small_strain_finite_strain(self):
        """Tests the equality of the displacement field obtained from
        integration of small strain field Vs. finite strain field of a cell."""
        F_amp = 1e-4
        lengths = [1.0, 1.0, 1.0]
        nb_grid_pts = [11, 11, 3]
        dim = len(nb_grid_pts)
        Nx, Ny, Nz = nb_grid_pts
        formulation = µ.Formulation.small_strain
        Young = [10, 20]  # Youngs modulus for each phase
        Poisson = [0.3, 0.4]  # Poissons ratio for each phase

        # solver
        newton_tol = 1e-8  # tolerance for newton algo
        cg_tol = 1e-10  # tolerance for cg algo
        equil_tol = 1e-8  # tolerance for equilibrium
        maxiter = 1000
        verbose = µ.Verbosity.Silent

        # sinusoidal bump
        d = int(nb_grid_pts[1]*0.2)
        l = Nx//3  # length of bump
        h = Ny//5  # height of bump
        low_y = (Ny-d)//2  # lower y-boundary of phase
        high_y = low_y+d  # upper y-boundary of phase
        left_x = (Nx-l)//2  # boundaries in x direction left
        right_x = left_x + l  # boundaries in x direction right
        x = np.arange(l)
        p_y = h*np.sin(np.pi/(l-1)*x)  # bump function
        xy_bump = np.ones((l, h, Nz))  # grid representing bumpx

        for i, threshold in enumerate(np.round(p_y)):
            xy_bump[i, int(threshold):, :] = 0

        phase = np.zeros(nb_grid_pts, dtype=int)  # 0 for surrounding matrix
        phase[:, low_y:high_y, :] = 1
        phase[left_x:right_x, high_y:high_y+h, :] = xy_bump

        ### Run muSpectre ###
        #-------------------#
        fourier_gradient =\
            [µ.FourierDerivative(dim , i) for i in range(dim)]
        ######finite strain:
        DelF = F_amp * np.array([[+0.50, +0.20, +0.00],
                                 [+0.00, -0.03, +0.15],
                                 [+0.00, +0.15, +0.40]])
        F = np.identity(DelF.shape[0]) + DelF

        cell_finite =\
            µ.Cell(nb_grid_pts, lengths, µ.Formulation.finite_strain,
                   fourier_gradient)
        mat_finite =\
            µ.material.MaterialLinearElastic4_3d.make(cell_finite,
                                                      "material_finite")
        for pixel_id, pixel in cell_finite.pixels.enumerate():
            # add Young and Poisson depending on the material index
            m_i = phase.flatten(order='F')[pixel_id]
            mat_finite.add_pixel(pixel_id, Young[m_i], Poisson[m_i])

        cell_finite.initialise()  # initialization of fft to make faster fft

        solver_newton_finite = \
            µ.solvers.KrylovSolverCG(cell_finite, cg_tol, maxiter, verbose)
        result_finite = µ.solvers.newton_cg(cell_finite, DelF,
                                            solver_newton_finite,
                                            newton_tol, equil_tol, verbose)
        ######small strain:
        DelE = 0.5 * ((F.T).dot(F) - np.identity(DelF.shape[0]))

        cell_small =\
            µ.Cell(nb_grid_pts, lengths, µ.Formulation.small_strain,
                   fourier_gradient)
        mat_small =\
            µ.material.MaterialLinearElastic4_3d.make(cell_small,
                                                      "material_small")
        for pixel_id, pixel in cell_small.pixels.enumerate():
            # add Young and Poisson depending on the material index
            m_i = phase.flatten(order='F')[pixel_id]
            mat_small.add_pixel(pixel_id, Young[m_i], Poisson[m_i])

        cell_small.initialise()  # initialization of fft to make faster fft
        solver_newton_small = \
            µ.solvers.KrylovSolverCG(cell_small, cg_tol, maxiter, verbose)
        result_small = µ.solvers.newton_cg(cell_small, DelE,
                                           solver_newton_small,
                                           newton_tol, equil_tol, verbose)

        #-------------------#
        # integration of the deformation gradient field
        placement_n_finite, x = \
            µ.gradient_integration.compute_placement(
                result_finite, lengths, nb_grid_pts, fourier_gradient,
                formulation = result_finite.formulation)

        placement_n_small, x = \
            µ.gradient_integration.compute_placement(
                result_small, lengths, nb_grid_pts, fourier_gradient,
                formulation = result_small.formulation)

        err_norm = np.linalg.norm(placement_n_small - placement_n_finite)
        self.assertLess(err_norm, F_amp * 5)


    def test_vacuum(self):
        form = µ.Formulation.finite_strain
        Poisson = 0.3
        nb_pts = 9
        delF_lat = -0.10
        delF_surf = 0.50

        for dim in [2, 3]:
            if dim == 2:
                discrete_gradient = [Stencils2D.d_10_00, Stencils2D.d_01_00,
                                     Stencils2D.d_11_01, Stencils2D.d_11_10]
                # We are compressing 10% in lateral and 50% in normal direction
                DelF = np.array([[delF_lat, 0.0],
                                 [0.0, delF_surf]])
                mat = µ.material.MaterialLinearElastic1_2d
            else:
                dz = muFFT.DiscreteDerivative([0, 0, 0],
                                              [[[-0.25, 0.25], [-0.25, 0.25]],
                                               [[-0.25, 0.25], [-0.25, 0.25]]])
                dy = dz.rollaxes(-1)
                dx = dy.rollaxes(-1)
                discrete_gradient = [dx, dy, dz]
                # We are compressing 10% in lateral and 50% in normal direction
                DelF = np.array([[delF_lat, 0.0, 0.0],
                                 [0.0, delF_lat, 0.0],
                                 [0.0, 0.0, delF_surf]])
                mat = µ.material.MaterialLinearElastic1_3d

            fourier_gradient = [µ.FourierDerivative(dim, d) for d in range(dim)]

            lengths = [1.]*dim
            nb_grid_pts = [nb_pts]*dim

            for k, gradient_op in enumerate([fourier_gradient,
                                             discrete_gradient]):
                cell = µ.Cell(nb_grid_pts, lengths, form, gradient_op)

                mat_vac = mat.make(cell, "vacuum", 0, 0)
                mat_sol = mat.make(cell, "el", 1, Poisson)

                for i, pixel in enumerate(cell.pixels):
                    if np.array(pixel)[-1] == nb_grid_pts[-1]-1:
                        mat_vac.add_pixel(i)
                    else:
                        mat_sol.add_pixel(i)

                # Solver
                newton_tol = 1e-8  # tolerance for newton algo
                cg_tol = 1e-8  # tolerance for cg algo
                equil_tol = 1e-8  # tolerance for equilibrium
                maxiter = 1000
                verbose = µ.Verbosity.Silent
                #verbose = µ.Verbosity.Full

                solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter,
                                                  verbose)
                cell.initialise()

                result = µ.solvers.newton_cg(cell, DelF, solver,
                                             newton_tol, equil_tol, verbose)

                F = cell.strain.array(muGrid.IterUnit.Pixel)
                self.assertTrue(np.allclose(F[0, 0], 1+delF_lat))
                self.assertTrue(np.allclose(F[1, 0], 0))
                self.assertTrue(np.allclose(F[0, 1], 0))
                if dim == 3:
                    self.assertTrue(np.allclose(F[1, 1], 1+delF_lat))
                    self.assertTrue(np.allclose(F[2, 0], 0))
                    self.assertTrue(np.allclose(F[0, 2], 0))
                    self.assertTrue(np.allclose(F[2, 1], 0))
                    self.assertTrue(np.allclose(F[1, 2], 0))

                displ, r = µ.gradient_integration.compute_placement(
                    F, lengths, nb_grid_pts, gradient_op,
                    formulation=µ.Formulation.finite_strain)

                if dim == 2:
                    x, y = displ
                    self.assertTrue(np.allclose(x,
                                                np.arange(nb_pts + 1).reshape(
                                                    -1, 1) / (nb_pts + 1)))
                    if k == 0:
                        # Fourier gradient
                        self.assertAlmostEqual(
                            y[0, -1] - y[0, -2],
                            1.5-(nb_pts-1)/nb_pts*(1 + Poisson*0.1*dim),
                            delta=0.05)
                    else:
                        # discrete gradient
                        self.assertAlmostEqual(
                            y[0, -1] - y[0, -2],
                            1.5-(nb_pts-1)/nb_pts*(1 + Poisson*0.1*dim),
                            delta=0.03)
                else:
                    x, y, z = displ
                    self.assertTrue(np.allclose(x,
                                                np.arange(nb_pts + 1).reshape(
                                                    -1, 1, 1) / (nb_pts + 1)))
                    self.assertTrue(np.allclose(y,
                                                np.arange(nb_pts + 1).reshape(
                                                    1, -1, 1) / (nb_pts + 1)))
                    if k == 0:
                        # Fourier gradient
                        self.assertAlmostEqual(
                            z[0, 0, -1] - z[0, 0, -2],
                            1.5-(nb_pts-1)/nb_pts*(1 + Poisson*0.1*dim),
                            delta=0.05)
                    else:
                        # discrete gradient
                        self.assertAlmostEqual(
                            z[0, 0, -1] - z[0, 0, -2],
                            1.5-(nb_pts-1)/nb_pts*(1 + Poisson*0.1*dim),
                            delta=0.015)




if __name__ == '__main__':
    unittest.main()
