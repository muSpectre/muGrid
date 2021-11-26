#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_mpi_stochastic_plasticity_search_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   19 Sep 2019

@brief  parallel tests for stochastic_plasticity_search.py

Copyright © 2019 Till Junge

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

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import unittest
import time
import warnings
import os
import numpy as np

from python_test_imports import µ
from muFFT import Stencils3D
import muSpectre.stochastic_plasticity_search as sps
from python_stochastic_plasticity_search_test import \
    stiffness_matrix,  green_lagrangian_strain_vector, PK2_tensor,  sigma_eq, \
    init_cg_solver, init_file_io_object, eps_voigt_vector


def init_cell(res, lens, formulation, gradient, fft, comm):
    return µ.Cell(res, lens, formulation, gradient, fft, comm)


def init_mat(cell, young, poisson, yield_stress,
             plastic_increment, eigen_strain):
    mat = µ.material.MaterialStochasticPlasticity_3d.make(cell, 'test_mat')
    dim = cell.dim
    nb_quad = cell.nb_quad_pts
    # init pixels
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel(pixel_id, young, poisson,
                      np.array([plastic_increment]*nb_quad),
                      yield_stress[tuple(pixel)],
                      eigen_strain.reshape((1, dim, dim))
                      .repeat(nb_quad)
                      .reshape((nb_quad, dim**2)))
    mat.initialise()

    return mat


def compute_local_quad_pt_id_from_pixel(global_pixel, subdomain_location,
                                        nb_domain_grid_pts, nb_quad_pts,
                                        pixel_quad_pt):
    dim = len(nb_domain_grid_pts)

    if dim == 3:
        nx, ny, nz = nb_domain_grid_pts
        px, py, pz = global_pixel - subdomain_location
        return (px + py * nx + pz * nx * ny) * nb_quad_pts + pixel_quad_pt

    elif dim == 2:
        nx, ny = nb_domain_grid_pts
        px, py = global_pixel - subdomain_location
        return (px + py * nx) * nb_quad_pts + pixel_quad_pt

    elif dim == 1:
        nx = nb_domain_grid_pts
        px = global_pixel - subdomain_location
        return px * nb_quad_pts + pixel_quad_pt

    else:
        raise ValueError("Not supported dimension D={}. The function "
                         "'compute_quad_pt_id_from_pixel' is only available "
                         "for one two or three dimensional systems."
                         .format(dim))


def compute_global_quad_pt_id_from_pixel(global_pixel, nb_domain_grid_pts,
                                         nb_quad_pts, pixel_quad_pt):
    dim = len(nb_domain_grid_pts)

    if dim == 3:
        nx, ny, nz = nb_domain_grid_pts
        px, py, pz = global_pixel
        return (px + py * nx + pz * nx * ny) * nb_quad_pts + pixel_quad_pt

    elif dim == 2:
        nx, ny = nb_domain_grid_pts
        px, py = global_pixel
        return (px + py * nx) * nb_quad_pts + pixel_quad_pt

    elif dim == 1:
        nx = nb_domain_grid_pts
        px = global_pixel
        return px * nb_quad_pts + pixel_quad_pt

    else:
        raise ValueError("Not supported dimension D={}. The function "
                         "'compute_quad_pt_id_from_pixel' is only available "
                         "for one two or three dimensional systems."
                         .format(dim))


class StochasticPlasticitySearch_Check(unittest.TestCase):
    """
    Test the stochastic plasticity algorithm for correctness
    """

    def setUp(self):
        # parallel environment
        self.comm = MPI.COMM_WORLD
        self.fft = "mpi"

        # cell parameters
        self.res = [3, 3, 3]
        self.dim = len(self.res)
        self.lens = [1, 1.2, 1]
        self.formulation = µ.Formulation.small_strain
        self.gradient = Stencils3D.linear_finite_elements
        self.nb_quad_pts = int(len(self.gradient) / self.dim)

        # material parameters
        self.young = 210
        self.poisson = 0.33
        self.plastic_increment = 1e-5
        self.eigen_strain = np.zeros((self.dim, self.dim))
        np.random.seed(18*(self.comm.rank+1))
        self.yield_stress = \
            np.random.random(tuple(self.res) + (self.nb_quad_pts, ))
        # uniform distribution [a,b]
        mean = np.mean(self.yield_stress)
        std = np.std(self.yield_stress)
        a = mean - std
        b = mean + std
        self.inverse_cumulative_dist_func = lambda z: a + (b-a)*z

        # solver parameter
        self.newton_tol = 1e-6
        self.cg_tol = 1e-6  # tolerance for cg algo
        self.newton_equil_tol = 1e-6  # tolerance for equilibrium
        self.maxiter = 100
        self.verbose = µ.Verbosity.Silent
        self.verbose_sps = 0

        # stochastic plasticity maximum and accuracy parameters
        self.yield_surface_accuracy = 1e-6
        self.n_max_strain_loop = 20
        self.n_max_bracket_search = 30
        self.n_max_avalanche = int(np.prod(np.array(self.res))*2/3)

        # set timing = True for timing information
        self.timing = False
        self.startTime = time.time()

    def tearDown(self):
        if self.timing:
            if self.comm.rank == 0:
                t = time.time() - self.startTime
                print("{}:\n{:.3f} seconds".format(self.id(), t))

    def test_update_eigen_strain(self):
        """Test if the eigen strain is updated properly on each core"""
        cell = init_cell(self.res, self.lens,
                         self.formulation, self.gradient,
                         self.fft, self.comm)
        mat = init_mat(cell, self.young, self.poisson, self.yield_stress,
                       self.plastic_increment, self.eigen_strain)

        pixel = np.array(cell.subdomain_locations)
        quad_pt = 2
        quad_pt_id = compute_local_quad_pt_id_from_pixel(
            pixel, np.array(cell.subdomain_locations), cell.nb_domain_grid_pts,
            cell.nb_quad_pts, pixel_quad_pt=quad_pt)

        # read initial eigen strain
        init_strain = np.copy(mat.get_eigen_strain(quad_pt_id))

        # set stress on quad_pt_0 corresponding to pixel [0,0,0] (only rank 0)
        if self.comm.rank == 0:
            stress = np.zeros((self.dim, self.dim))
            stress_1 = 0.3
            stress[0, 1] = stress_1
            sps.update_eigen_strain(mat, quad_pt_id, stress, self.dim)

            # read out updated eigen strain and proof
            updated_strain = mat.get_eigen_strain(quad_pt_id)
            analytic_strain = init_strain
            analytic_strain[0, 1] = \
                self.plastic_increment * stress_1 / np.sqrt(3/2 * stress_1**2)
            self.assertLess(
                np.linalg.norm(updated_strain - analytic_strain), 1e-8)

        else:
            # updated_strain = np.zeros(dim,dim) on all ranks != 0
            updated_strain = mat.get_eigen_strain(quad_pt_id)

        # update eigen strain
        stress = np.zeros((self.dim, self.dim))
        stress_2 = 0.4
        stress[1, 0] = stress_2
        sps.update_eigen_strain(mat, quad_pt_id, stress, self.dim)

        # read out updated eigen strain and proof
        updated_strain_2 = mat.get_eigen_strain(quad_pt_id)
        analytic_strain = updated_strain
        analytic_strain[1, 0] = \
            self.plastic_increment * stress_2 / np.sqrt(3/2 * stress_2**2)
        self.assertLess(
            np.linalg.norm(updated_strain_2 - analytic_strain), 1e-8)

    def test_set_new_threshold(self):
        cell = init_cell(self.res, self.lens, self.formulation,
                         self.gradient, self.fft, self.comm)
        mat = init_mat(cell, self.young, self.poisson, self.yield_stress,
                       self.plastic_increment, self.eigen_strain)

        pixel = np.array(cell.subdomain_locations)
        quad_pt = 4
        quad_pt_id = compute_local_quad_pt_id_from_pixel(
            pixel, np.array(cell.subdomain_locations), cell.nb_domain_grid_pts,
            cell.nb_quad_pts, pixel_quad_pt=quad_pt)
        # uniform distribution on the interval (a,b)
        a = 10
        b = 14
        inv_cum_dist_func = lambda z: a + (b-a)*z

        # write first time a threshold on the pixel
        seed_1 = 19092019 * (self.comm.rank+1)
        np.random.seed(seed_1)
        sps.set_new_threshold(mat, quad_pt_id,
                              inverse_cumulative_dist_func=inv_cum_dist_func)
        np.random.seed(seed_1)
        threshold_expected = inv_cum_dist_func(np.random.random())
        threshold_read = mat.get_stress_threshold(quad_pt_id)
        self.assertLess(threshold_expected - threshold_read, 1e-8)

        # write second time a threshold on the pixel
        seed_2 = 2019 * (self.comm.rank+1)
        np.random.seed(seed_2)
        sps.set_new_threshold(mat, quad_pt_id,
                              inverse_cumulative_dist_func=inv_cum_dist_func)
        np.random.seed(seed_2)
        threshold_expected = inv_cum_dist_func(np.random.random())
        threshold_read = mat.get_stress_threshold(quad_pt_id)
        self.assertLess(threshold_expected - threshold_read, 1e-8)

    def test_propagate_avalanche_step(self):
        """
        Check if a single overloaded pixel (on each proc) breaks, at the right
        strain load.
        """
        cell = init_cell(self.res, self.lens, self.formulation,
                         self.gradient, self.fft, self.comm)
        strain_xy = 0.1
        weak_pixel = np.array(tuple(cell.subdomain_locations)) \
            + np.array([1, 0, 0])*self.comm.rank
        quad_pt = np.array([self.comm.rank % self.nb_quad_pts])
        weak_quad_pt_id = compute_global_quad_pt_id_from_pixel(
            weak_pixel, cell.nb_domain_grid_pts, cell.nb_quad_pts,
            pixel_quad_pt=quad_pt)

        weak_pixels = np.empty(self.comm.Get_size()*3, dtype=int)
        self.comm.Allgather([weak_pixel, MPI.INT], [weak_pixels, MPI.INT])
        weak_pixels = weak_pixels.reshape((self.comm.Get_size(), 3))

        quad_pts = np.empty(self.comm.Get_size(), dtype=int)
        self.comm.Allgather([quad_pt, MPI.INT], [quad_pts, MPI.INT])

        # analytic
        # analytic compute the equivalent stress for a given strain 'strain_xy'
        C = stiffness_matrix(self.young, self.poisson, self.dim)
        if self.formulation == µ.Formulation.finite_strain:
            F = np.eye(self.dim)
            F[0, 1] = strain_xy
            E = green_lagrangian_strain_vector(F, self.dim)
            PK2_analytic = PK2_tensor(C, E)
            # analytic computed equivalent stress
            eq_stress = sigma_eq(PK2_analytic)
        elif self.formulation == µ.Formulation.small_strain:
            eps = np.zeros((self.dim, self.dim))
            eps[0, 1] = strain_xy / 2
            eps[1, 0] = eps[0, 1]
            eps_v = eps_voigt_vector(eps, self.dim)
            cauchy_analytic = PK2_tensor(C, eps_v)
            # analytic computed equivalent stress
            eq_stress = sigma_eq(cauchy_analytic)

        # numeric
        # set the analytic computed equivalent stress reduced by a tiny amount
        # as threshold for one "weak pixel" and the other thresholds to
        # slightly higher values.
        fixed_yield_stress = np.ones(tuple(self.res) + (self.nb_quad_pts,),
                                     order='F') * eq_stress * (1 + 1e-8)
        for index, qpt in zip(weak_pixels, quad_pts):
            fixed_yield_stress[tuple(index) + (qpt,)] = eq_stress * (1 - 1e-8)

        # init material
        mat = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
                       self.plastic_increment, self.eigen_strain)

        cg_solver = init_cg_solver(cell, self.cg_tol,
                                   self.maxiter, self.verbose)
        # set the eigen strain field to the previous fixed deformation
        # 'strain_xy'
        strain_field = cell.strain.array()
        if self.formulation == µ.Formulation.finite_strain:
            for i in range(self.dim):
                strain_field[i, i, ...] = 1.0
            strain_field[0, 1, ...] = strain_xy
        elif self.formulation == µ.Formulation.small_strain:
            for i in range(self.dim):
                strain_field[i, i, ...] = 0.0
            strain_field[0, 1, ...] = strain_xy/2
            strain_field[1, 0, ...] = strain_field[0, 1, ...]

        # check if you can find the yielding quad point
        overloaded_quad_pts = sps.propagate_avalanche_step(
            mat, cell, self.dim, cg_solver, self.newton_tol,
            self.newton_equil_tol, self.verbose)

        # one overloaded quad point per core!
        self.assertTrue(len(overloaded_quad_pts) == 1)
        self.assertTrue(overloaded_quad_pts == weak_quad_pt_id)

    def test_bracket_search(self):
        """
        Tests:
        1. Test if bracket search find the exact yield point for one pixel with
           a lower yield threshold than the others.
        2. Test exception for two/n pixels with very close yield criterion.
           Thus they should break together (hence an avalanche can start for
           n>= 2 pixel)
        3. Test if an error is raised when the maximum allowed bracket steps
           are reached.
        """
        # ------- 1. ------- #
        # init data
        low_yield_stress = 9.5
        yield_surface_accuracy = 1e-7
        n_max_bracket_search = 3

        # set the initial deformation close to the final deformation to reduce
        # the needed bracket search steps
        if self.formulation == µ.Formulation.finite_strain:
            g_01 = 0.06941878  # final deformation
            strain_init = np.array([[1, g_01 - yield_surface_accuracy*1.75, 0],
                                    [0,  1,  0],
                                    [0,  0,  1]])
        elif self.formulation == µ.Formulation.small_strain:
            g_01 = 0.03473725  # approx. g_01/2 of finite strain
            strain_init = \
                np.array([[0, g_01 - yield_surface_accuracy/2*1.75, 0],
                          [g_01 - yield_surface_accuracy/2*1.75, 0, 0],
                          [0,  0,  0]])

        fixed_yield_stress = np.ones(tuple(self.res) + (self.nb_quad_pts,),
                                     order='F')*14  # high threshold
        weak_pixel = [0, 0, 0]
        weak_quad_pt = 5
        # global quad point id which is easy to compute for pixel [0,0,0]
        weak_quad_pt_id = 5
        fixed_yield_stress[tuple(weak_pixel) + (weak_quad_pt,)] = \
            low_yield_stress
        cell = init_cell(self.res, self.lens, self.formulation,
                         self.gradient, self.fft, self.comm)
        mat = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
                       self.plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol,
                                   self.maxiter, self.verbose)

        DelF_initial = np.zeros((self.dim, self.dim))
        if self.formulation == µ.Formulation.finite_strain:
            DelF_initial[0, 1] = yield_surface_accuracy
        elif self.formulation == µ.Formulation.small_strain:
            DelF_initial[0, 1] = yield_surface_accuracy/2
            DelF_initial[1, 0] = DelF_initial[0, 1]

        # initialize cell with unit-matrix deformation gradient
        cell_strain = cell.strain.array()
        cell_strain[:] = np.tensordot(
            strain_init, np.ones((self.nb_quad_pts,)
                                 + tuple(cell.nb_subdomain_grid_pts)), axes=0)

        # if self.comm.rank == 0:
        next_DelF_guess, PK2, F, breaking_quad_pts = \
            sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
                               self.newton_equil_tol, yield_surface_accuracy,
                               n_max_bracket_search, DelF_initial,
                               self.verbose_sps)

        # Is it exactly one breaking pixel and if yes is it pixel [0,0,0]?
        self.assertEqual(len(breaking_quad_pts), 1)
        self.assertTrue(breaking_quad_pts == weak_quad_pt_id)

        # plug in the numeric result into the analytic formula and see if one
        # gets out the exact yield stress (low_yield_stress)
        F = F.reshape((self.dim, self.dim, self.nb_quad_pts,
                       *cell.nb_subdomain_grid_pts), order='f')
        PK2 = PK2.reshape((self.dim, self.dim, self.nb_quad_pts,
                           *cell.nb_subdomain_grid_pts), order='f')
        F_numeric = F[:, :, weak_quad_pt, 0, 0, 0]
        stress_numeric = PK2[:, :, weak_quad_pt, 0, 0, 0]
        C = stiffness_matrix(self.young, self.poisson, self.dim)
        if self.formulation == µ.Formulation.finite_strain:
            E = green_lagrangian_strain_vector(F_numeric, self.dim)
            PK2_analytic = PK2_tensor(C, E)
            eq_stress = sigma_eq(PK2_analytic)
            stress_analytic = PK2_analytic
        elif self.formulation == µ.Formulation.small_strain:
            eps_v = eps_voigt_vector(F_numeric, self.dim)
            cauchy_analytic = PK2_tensor(C, eps_v)
            eq_stress = sigma_eq(cauchy_analytic)
            stress_analytic = cauchy_analytic

        # Is the analytic yield stress equivalent to yield stress of
        # weak_quad_pt at pix(0,0,0)?
        self.assertLess(np.abs(low_yield_stress - eq_stress), 5.9e-6)

        # Is the computed deformation gradient F_numeric correct?
        if self.formulation == µ.Formulation.finite_strain:
            F_yield10 = np.array([[1, g_01, 0],
                                  [0,  1,   0],
                                  [0,  0,   1]])
        elif self.formulation == µ.Formulation.small_strain:
            F_yield10 = np.array([[0,  g_01, 0],
                                  [g_01, 0,  0],
                                  [0,    0,  0]])
        self.assertLess(np.linalg.norm(F_yield10 - F_numeric),
                        yield_surface_accuracy)
        # compare the computed stress (here PK1 vs PK2 because stochastic
        # plasticity computes PK1 up to now but has to be changed to PK2)
        self.assertLess(np.linalg.norm(stress_analytic - stress_numeric), 1e-8)

        # ------- 2. ------- #
        # init data
        low_yield_stress = 9.5
        yield_surface_accuracy = 1e-8
        small_yield_difference = \
            low_yield_stress * yield_surface_accuracy**2 * 1e-2
        # set the initial deformation close to the final deformation to reduce
        # the needed bracket search steps
        if self.formulation == µ.Formulation.finite_strain:
            g_01 = 0.06941878  # final deformation
            strain_init = \
                np.array([[1, g_01 - yield_surface_accuracy**2*1.75, 0],
                          [0,  1,  0],
                          [0,  0,  1]])
        elif self.formulation == µ.Formulation.small_strain:
            g_01 = 0.03473726  # approx. g_01/2 of finite strain
            strain_init = \
                np.array([[0, g_01 - yield_surface_accuracy**2/2*1.75, 0],
                          [g_01 - yield_surface_accuracy**2/2*1.75, 0, 0],
                          [0,  0,  0]])
        n_max_bracket_search = 4

        fixed_yield_stress = np.ones(tuple(self.res) + (self.nb_quad_pts,),
                                     order='F')*14  # high threshold
        weak_pix_1 = [0, 0, 0]
        w_qpt_1 = 3
        w_qpt_1_global_id = compute_global_quad_pt_id_from_pixel(
            weak_pix_1, self.res, self.nb_quad_pts, w_qpt_1)

        weak_pix_2 = [i//2+1 for i in self.res]
        w_qpt_2 = 4
        w_qpt_2_global_id = compute_global_quad_pt_id_from_pixel(
            weak_pix_2, self.res, self.nb_quad_pts, w_qpt_2)

        fixed_yield_stress[tuple(weak_pix_1) + (w_qpt_1,)] = low_yield_stress
        fixed_yield_stress[tuple(weak_pix_2) + (w_qpt_2,)] = \
            low_yield_stress + small_yield_difference
        cell = init_cell(self.res, self.lens, self.formulation,
                         self.gradient, self.fft, self.comm)
        mat = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
                       self.plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol,
                                   self.maxiter, self.verbose)

        DelF_initial = np.zeros((self.dim, self.dim))
        if self.formulation == µ.Formulation.finite_strain:
            DelF_initial[0, 1] = yield_surface_accuracy**2
        elif self.formulation == µ.Formulation.small_strain:
            DelF_initial[0, 1] = yield_surface_accuracy**2/2
            DelF_initial[1, 0] = DelF_initial[0, 1]

        # initialize cell with deformation gradient for fast convergence
        cell_strain = cell.strain.array()
        cell_strain[:] = np.tensordot(
            strain_init, np.ones((self.nb_quad_pts,)
                                 + tuple(cell.nb_subdomain_grid_pts)), axes=0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # all warnings be triggered.
            next_DelF_guess, PK2, F, breaking_quad_pts = \
                sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
                                   self.newton_equil_tol,
                                   yield_surface_accuracy,
                                   n_max_bracket_search,
                                   DelF_initial, verbose=1)
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
            self.assertTrue("bracket_search found 2 quad points overcoming "
                            "their yield threshold for the final deformation. "
                            "To initialise the avalanche think about using the"
                            " parameter 'single_quad_pt_start' of "
                            "'propagate_avalanche()' to start the avalanche "
                            "from a single quad point!" == str(w[-1].message))

        # Are there exactly two breaking pixels, [0,0,0] and
        # [nx//2,ny//2,nz//2] corresponding to the quad points 0 and 26?
        self.assertEqual(len(breaking_quad_pts), 2)
        self.assertTrue((breaking_quad_pts ==
                         [w_qpt_1_global_id, w_qpt_2_global_id]).all())

        # ------- 3. ------- #
        # use the initalization from the last test
        n_max_bracket_search = 2
        DelF_initial = np.zeros((self.dim, self.dim))
        if self.formulation == µ.Formulation.finite_strain:
            DelF_initial[0, 1] = 0.1
        elif self.formulation == µ.Formulation.small_strain:
            DelF_initial[0, 1] = 0.1 / 2
            DelF_initial[1, 0] = DelF_initial[0, 1]
        with self.assertRaises(RuntimeError):
            sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
                               self.newton_equil_tol, yield_surface_accuracy,
                               n_max_bracket_search, DelF_initial,
                               self.verbose_sps)

    def test_propagate_avalanche(self):
        """
        Tests:
        1. Test if plastic deformations are done in the right order and at the
           right place!
        2. Test if initially_overloaded_pixels behaves right.
        3. Test the parameter 'single_pixel_start'
        """
        # ------- 1. ------- #
        # init parameters
        res = [3, 3, 3]
        lens = [1, 1, 1]
        dim = len(res)
        strain_xy_1 = 0.01
        plastic_increment = strain_xy_1 * 12

        # analytic compute eq_stress_1 for a given strain 'strain_xy_1'
        C = stiffness_matrix(self.young, self.poisson, self.dim)
        if self.formulation == µ.Formulation.finite_strain:
            F = np.eye(self.dim)
            F[0, 1] = strain_xy_1
            E = green_lagrangian_strain_vector(F, self.dim)
            PK2_analytic = PK2_tensor(C, E)
            eq_stress = sigma_eq(PK2_analytic)
        elif self.formulation == µ.Formulation.small_strain:
            eps = np.zeros((self.dim, self.dim))
            eps[0, 1] = strain_xy_1 / 2
            eps[1, 0] = eps[0, 1]
            eps_v = eps_voigt_vector(eps, self.dim)
            cauchy_analytic = PK2_tensor(C, eps_v)
            eq_stress = sigma_eq(cauchy_analytic)
            plastic_increment /= 2

        eq_stress_1 = eq_stress
        eq_stress_2 = eq_stress * 1.0002
        eq_stress_3 = eq_stress * 1.0008

        pixel_1 = [1, 1, 1]
        qpt_1 = 5
        pixel_2 = [2, 2, 1]
        qpt_2 = 0
        pixel_3 = [1, 2, 1]
        qpt_3 = 3

        # init material, with fine tuned order of stress thresholds
        fixed_yield_stress = np.ones(tuple(res) + (self.nb_quad_pts, ))*17
        fixed_yield_stress[tuple(pixel_1) + (qpt_1,)] = eq_stress_1
        fixed_yield_stress[tuple(pixel_2) + (qpt_2,)] = eq_stress_2
        fixed_yield_stress[tuple(pixel_3) + (qpt_3,)] = eq_stress_2
        if self.comm.Get_size() == 2:
            pixel_4 = [2, 1, 2]
            qpt_4 = 0
            fixed_yield_stress[tuple(pixel_4) + (qpt_4,)] = eq_stress_3
        cell = init_cell(res, lens, self.formulation,
                         self.gradient, self.fft, self.comm)
        mat = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
                       plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol,
                                   self.maxiter, self.verbose)

        # compute quad_pt_ids
        quad_pt_1 = compute_global_quad_pt_id_from_pixel(
            pixel_1, cell.nb_domain_grid_pts, cell.nb_quad_pts,
            pixel_quad_pt=qpt_1)
        quad_pt_2 = compute_global_quad_pt_id_from_pixel(
            pixel_2, cell.nb_domain_grid_pts, cell.nb_quad_pts,
            pixel_quad_pt=qpt_2)
        quad_pt_3 = compute_global_quad_pt_id_from_pixel(
            pixel_3, cell.nb_domain_grid_pts, cell.nb_quad_pts,
            pixel_quad_pt=qpt_3)
        if self.comm.Get_size() == 2:
            quad_pt_4 = compute_global_quad_pt_id_from_pixel(
                pixel_4, cell.nb_domain_grid_pts, cell.nb_quad_pts,
                pixel_quad_pt=qpt_4)

        # expected avalanche histories
        if self.comm.Get_size() == 1:
            expected_ava_history = np.array([[quad_pt_1],
                                             [quad_pt_3],
                                             [quad_pt_2]])
        if self.comm.Get_size() == 2:
            expected_ava_history = np.array([[quad_pt_1, np.nan],
                                             [quad_pt_3, quad_pt_4],
                                             [quad_pt_2, np.nan]])

        # overload one pixel which breaks and by its plastic increment
        # overloads two additional pixels.

        # propagate the avalanche
        yield_surface_accuracy = 1e-8
        n_max_bracket_search = 5
        # set the eigen strain field to the slightly modified previous fixed
        # deformation 'strain_xy' for faster convergence
        strain_field = cell.strain.array()
        DelF_init = np.zeros((dim, dim))
        if self.formulation == µ.Formulation.finite_strain:
            for i in range(self.dim):
                strain_field[i, i, ...] = 1.0
            strain_field[0, 1, ...] = strain_xy_1 - yield_surface_accuracy*1.25
            DelF_init[0, 1] = yield_surface_accuracy
        elif self.formulation == µ.Formulation.small_strain:
            for i in range(self.dim):
                strain_field[i, i, ...] = 0.0
            strain_field[0, 1, ...] = \
                strain_xy_1/2 - yield_surface_accuracy/2*1.25
            strain_field[1, 0, ...] = strain_field[0, 1, ...]
            DelF_init[0, 1] = yield_surface_accuracy/2
            DelF_init[1, 0] = DelF_init[0, 1]

        n_max_avalanche = 10
        i_cdf = lambda z: 17  # constant value

        DelF, PK2, F, breaking_pixel = \
            sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
                               self.newton_equil_tol, yield_surface_accuracy,
                               n_max_bracket_search, DelF_init,
                               self.verbose_sps)

        def save_and_test_ava(n_strain_loop, ava_history):
            # check avalanche history
            self.assertTrue(np.isclose(ava_history, expected_ava_history,
                                       equal_nan=True).all())

        def create_save_fields(f_name, cell):
            # Clean up
            if self.comm.rank == 0:
                pwd = os.getcwd() + "/"
                if os.path.exists(pwd + f_name):
                    os.remove(pwd + f_name)
            MPI.COMM_WORLD.Barrier()

            # create new file
            if self.comm.size > 1:
                file_io_object = init_file_io_object(f_name, cell,
                                                     cell.communicator)
            else:
                file_io_object = init_file_io_object(f_name, cell)

            def save_fields(cell, n_strain_loop, before_avalanche):
                if before_avalanche:
                    file_io_object.append_frame().write(["stress", "strain"])

                if not before_avalanche:
                    file_io_object.append_frame().write(["stress", "strain"])

                return 0

            return save_fields

        # initial quadpoint is qpt_1 on pixel [1,1,1] with global id quad_pt_1
        self.assertTrue((breaking_pixel == quad_pt_1).all())

        f_name = "stochastic-plasticity-parallel-test-file.nc"
        save_fields = create_save_fields(f_name, cell)

        sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
                                self.newton_equil_tol, PK2, F, n_max_avalanche,
                                self.verbose_sps,
                                inverse_cumulative_dist_func=i_cdf,
                                save_avalanche=save_and_test_ava,
                                save_fields=save_fields,
                                n_strain_loop=0)

        # ------- 2. ------- #
        cell = init_cell(res, lens, self.formulation,
                         self.gradient, self.fft, self.comm)
        mat = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
                       plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol,
                                   self.maxiter, self.verbose)

        strain_field = cell.strain.array()
        if self.formulation == µ.Formulation.finite_strain:
            for i in range(self.dim):
                strain_field[i, i, ...] = 1.0
            strain_field[0, 1, ...] = strain_xy_1 - yield_surface_accuracy*1.25
        elif self.formulation == µ.Formulation.small_strain:
            for i in range(self.dim):
                strain_field[i, i, ...] = 0.0
            strain_field[0, 1, ...] = \
                strain_xy_1/2 - yield_surface_accuracy/2*1.25
            strain_field[1, 0, ...] = strain_field[0, 1, ...]

        DelF, PK2, F, breaking_quad_pts = \
            sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
                               self.newton_equil_tol, yield_surface_accuracy,
                               n_max_bracket_search, DelF_init,
                               self.verbose_sps)

        # initial quad point is qpt_1 on pixel_1
        # which corresponds to global quad_pt_1
        self.assertTrue(breaking_quad_pts == quad_pt_1)

        f_name = "stochastic-plasticity-parallel-test-file.nc"
        save_fields = create_save_fields(f_name, cell)

        sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
                                self.newton_equil_tol, PK2, F, n_max_avalanche,
                                self.verbose_sps,
                                initially_overloaded_quad_pts=breaking_quad_pts,
                                inverse_cumulative_dist_func=i_cdf,
                                save_avalanche=save_and_test_ava,
                                save_fields=save_fields,
                                n_strain_loop=0)

        # CLEAN UP
        if self.comm.rank == 0:
            pwd = os.getcwd() + "/"
            os.remove(pwd + f_name)

        # ------- 3. ------- #
        # init material, with two pixels of the same yield strength
        pixel_1 = [1, 1, 1]
        qpt_1 = 5
        pixel_2 = [1, 1, 2]
        qpt_2 = 0

        fixed_yield_stress = np.ones(tuple(res) + (self.nb_quad_pts, ))*17
        fixed_yield_stress[tuple(pixel_1) + (qpt_1,)] = eq_stress_1
        fixed_yield_stress[tuple(pixel_2) + (qpt_2,)] = eq_stress_1
        yield_surface_accuracy = 1e-8
        n_max_bracket_search = 5

        plastic_increment = strain_xy_1 * 10
        DelF_init = np.zeros((dim, dim))
        if self.formulation == µ.Formulation.finite_strain:
            DelF_init[0, 1] = yield_surface_accuracy**2
        elif self.formulation == µ.Formulation.small_strain:
            DelF_init[0, 1] = yield_surface_accuracy**2/2
            DelF_init[1, 0] = DelF_init[0, 1]

        def setup_material():
            # helper function to set up the material which is needed four times
            cell = init_cell(res, lens, self.formulation,
                             self.gradient, self.fft, self.comm)
            mat = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
                           plastic_increment, self.eigen_strain)
            cg_solver = init_cg_solver(cell, self.cg_tol,
                                       self.maxiter, self.verbose)

            # set the eigen strain field to the previous fixed deformation
            strain_field = cell.strain.array()
            if self.formulation == µ.Formulation.finite_strain:
                for i in range(self.dim):
                    strain_field[i, i, ...] = 1.0
                    strain_field[0, 1, ...] = \
                        strain_xy_1 - yield_surface_accuracy**2*1.25
            elif self.formulation == µ.Formulation.small_strain:
                for i in range(self.dim):
                    strain_field[i, i, ...] = 0.0
                    strain_field[0, 1, ...] = \
                        strain_xy_1/2 - yield_surface_accuracy**2/2*1.25
                    strain_field[1, 0, ...] = strain_field[0, 1, ...]

            with warnings.catch_warnings():
                # suppress warnings of bracket_search()
                warnings.simplefilter("ignore")
                DelF, PK2, F, breaking_quad_pts = \
                    sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
                                       self.newton_equil_tol,
                                       yield_surface_accuracy,
                                       n_max_bracket_search, DelF_init,
                                       self.verbose_sps)
            return mat, cell, cg_solver, PK2, F, breaking_quad_pts

        def sa_2break(n_strain_loop, ava_history):
            self.assertTrue(ava_history.shape == (1, 2))

        def sa_1break(n_strain_loop, ava_history):
            self.assertTrue(ava_history.shape == (2, 1))

        # Check for all combinations of initially_overloaded_pixels and
        # single_quad_pt_start
        mat, cell, cg_solver, PK2, F, breaking_quad_pts = setup_material()
        # compute quad_pt_ids
        quad_pt_1 = compute_global_quad_pt_id_from_pixel(
            pixel_1, cell.nb_domain_grid_pts, cell.nb_quad_pts, qpt_1)
        quad_pt_2 = compute_global_quad_pt_id_from_pixel(
            pixel_2, cell.nb_domain_grid_pts, cell.nb_quad_pts, qpt_2)
        self.assertTrue((breaking_quad_pts == [quad_pt_1, quad_pt_2]).all())
        sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
                                self.newton_equil_tol, PK2, F, n_max_avalanche,
                                self.verbose_sps,
                                initially_overloaded_quad_pts=None,
                                single_quad_pt_start=False,
                                inverse_cumulative_dist_func=i_cdf,
                                save_avalanche=sa_2break,
                                save_fields=None, n_strain_loop=0)

        mat, cell, cg_solver, PK2, F, breaking_quad_pts = setup_material()
        sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
                                self.newton_equil_tol, PK2, F, n_max_avalanche,
                                self.verbose_sps,
                                initially_overloaded_quad_pts=breaking_quad_pts,
                                single_quad_pt_start=False,
                                inverse_cumulative_dist_func=i_cdf,
                                save_avalanche=sa_2break,
                                save_fields=None, n_strain_loop=0)

        mat, cell, cg_solver, PK2, F, breaking_quad_pts = setup_material()
        sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
                                self.newton_equil_tol, PK2, F, n_max_avalanche,
                                self.verbose_sps,
                                initially_overloaded_quad_pts=breaking_quad_pts,
                                single_quad_pt_start=True,
                                inverse_cumulative_dist_func=i_cdf,
                                save_avalanche=sa_1break,
                                save_fields=None, n_strain_loop=0)

        mat, cell, cg_solver, PK2, F, breaking_quad_pts = setup_material()
        sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
                                self.newton_equil_tol, PK2, F, n_max_avalanche,
                                self.verbose_sps,
                                initially_overloaded_quad_pts=None,
                                single_quad_pt_start=True,
                                inverse_cumulative_dist_func=i_cdf,
                                save_avalanche=sa_1break,
                                save_fields=None, n_strain_loop=0)

    def test_strain_cell(self):
        """
        Tests:
        1. Test if the function reaches the required deformation
        2. Small deformation with only one avalanche. Check:
            - avalanche pixel index
            - PK2, stress field
            - F, deformation gradient field
        """
        # ------- 1. ------- #
        DelF = np.zeros((self.dim, self.dim))
        DelF[0, 1] = 0.0001
        if self.formulation == µ.Formulation.small_strain:
            DelF[0, 1] /= 2
            DelF[1, 0] = DelF[0, 1]

        if self.formulation == µ.Formulation.finite_strain:
            F_tot = np.eye(self.dim)
            F_tot[0, 1] = 0.0002
        elif self.formulation == µ.Formulation.small_strain:
            F_tot = np.zeros((self.dim, self.dim))
            F_tot[0, 1] = 0.0002
            F_tot[1, 0] = F_tot[0, 1]

        cell = init_cell(self.res, self.lens, self.formulation,
                         self.gradient, self.fft, self.comm)
        mat = init_mat(cell, self.young, self.poisson, self.yield_stress,
                       self.plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter,
                                   self.verbose)
        F_fin = sps.strain_cell(mat, cell, cg_solver, self.newton_tol,
                                self.newton_equil_tol, DelF, F_tot,
                                self.yield_surface_accuracy,
                                self.n_max_strain_loop,
                                self.n_max_bracket_search,
                                self.n_max_avalanche, self.verbose_sps, False,
                                self.inverse_cumulative_dist_func,
                                save_avalanche=None,
                                save_fields=None,
                                is_strain_initialised=False)
        # is the reached deformation larger or almost equal to the required one
        self.assertTrue(((F_fin-F_tot) > -1e-14).all())

    def test_parallel_vs_serial(self):
        """
        Check if the parallel and serial version of stochastic plasticity
        lead to the same results.
        """
        if self.comm.size == 1:
            # execute test_parallel_vs_serial only for size > 1.
            # It works for size = 1 but we save the time for this execution.
            return 0

        # setup
        res = [3, 3, 3]
        yield_stress = 1.5
        # set the same random seed on each core, otherwise the serial and
        # parallel versions dont come to the same result because they have
        # different random numbers.
        np.random.seed(12345)
        fixed_yield_stress = 0.90 * yield_stress \
            + np.random.random(tuple(res) + (self.nb_quad_pts,)) \
            * 0.20 * yield_stress
        a = yield_stress * 0.90
        b = yield_stress * 1.10
        i_cdf = lambda z: a + (b-a)*z
        DelF = np.zeros((self.dim, self.dim))
        plastic_increment = 4e-3  # 1e-3 leads to different avalanche

        if self.formulation == µ.Formulation.finite_strain:
            g_01 = 0.00989125
            strain_init = \
                np.array([[1, g_01 - self.yield_surface_accuracy*1.75, 0],
                          [0,  1,  0],
                          [0,  0,  1]])
            DelF[0, 1] = self.yield_surface_accuracy
            F_tot = np.eye(self.dim)
            F_tot[0, 1] = 0.01
        elif self.formulation == µ.Formulation.small_strain:
            g_01 = 0.00494575  # approx. 0.00989125 / 2
            strain_init = \
                np.array([[1, g_01 - self.yield_surface_accuracy/2*1.75, 0],
                          [g_01 - self.yield_surface_accuracy/2*1.75, 1, 0],
                          [0,  0,  1]])
            DelF[0, 1] = self.yield_surface_accuracy / 2
            DelF[1, 0] = DelF[0, 1]
            F_tot = np.zeros((self.dim, self.dim))
            F_tot[0, 1] = 0.01 / 2
            F_tot[1, 0] = F_tot[0, 1]
            plastic_increment /= 2

        n_max_bracket_search = 20
        n_max_strain_loop = 7
        ava_serial = []
        ava_parallel = []

        def save_and_test_ava_s(n_strain_loop, ava_history):
            ava_serial.append(ava_history)

        def save_and_test_ava_p(n_strain_loop, ava_history):
            ava_parallel.append(ava_history)

        # serial
        # create a new communicator with only one rank to make a effective
        # (cell with comm that has only one rank leads to a serial
        # computation, no mpi) serial computation.
        comm_s = self.comm.Split(color=self.comm.rank)
        if self.comm.rank == 0:
            fft_s = "fftw"
            cell_s = init_cell(res, self.lens, self.formulation,
                               self.gradient, fft_s, comm_s)
            mat_s = init_mat(cell_s, self.young, self.poisson,
                             fixed_yield_stress, plastic_increment,
                             self.eigen_strain)
            cg_solver_s = init_cg_solver(cell_s, self.cg_tol,
                                         self.maxiter, self.verbose)
            c_strain = cell_s.strain.array()
            c_strain[:] = np.tensordot(
                strain_init, np.ones((self.nb_quad_pts,)
                                     + tuple(cell_s.nb_domain_grid_pts)),
                axes=0)

            DelF, PK2, F, breaking_quad_pts = \
                sps.bracket_search(mat_s, cell_s, cg_solver_s, self.newton_tol,
                                   self.newton_equil_tol,
                                   self.yield_surface_accuracy,
                                   n_max_bracket_search, DelF, self.verbose_sps)

            F_fin_s = sps.strain_cell(mat_s, cell_s, cg_solver_s,
                                      self.newton_tol, self.newton_equil_tol,
                                      DelF, F_tot, self.yield_surface_accuracy,
                                      n_max_strain_loop, n_max_bracket_search,
                                      self.n_max_avalanche,
                                      self.verbose_sps, False,
                                      i_cdf, save_avalanche=save_and_test_ava_s,
                                      save_fields=None,
                                      is_strain_initialised=True)

        # parallel
        fft_p = "fftwmpi"
        comm_p = self.comm
        cell_p = init_cell(res, self.lens, self.formulation,
                           self.gradient, fft_p, comm_p)
        mat_p = init_mat(cell_p, self.young, self.poisson, fixed_yield_stress,
                         plastic_increment, self.eigen_strain)
        cg_solver_p = init_cg_solver(cell_p, self.cg_tol,
                                     self.maxiter, self.verbose)
        c_strain = cell_p.strain.array()
        c_strain[:] = np.tensordot(
            strain_init, np.ones((self.nb_quad_pts,)
                                 + tuple(cell_p.nb_subdomain_grid_pts)),
            axes=0)

        DelF, PK2, F, breaking_quad_pts = \
            sps.bracket_search(mat_p, cell_p,
                               cg_solver_p, self.newton_tol,
                               self.newton_equil_tol,
                               self.yield_surface_accuracy,
                               n_max_bracket_search,
                               DelF, self.verbose_sps)

        F_fin_p = sps.strain_cell(mat_p, cell_p, cg_solver_p, self.newton_tol,
                                  self.newton_equil_tol, DelF, F_tot,
                                  self.yield_surface_accuracy,
                                  n_max_strain_loop, n_max_bracket_search,
                                  self.n_max_avalanche, self.verbose_sps, False,
                                  i_cdf, save_avalanche=save_and_test_ava_p,
                                  save_fields=None,
                                  is_strain_initialised=True)

        # comparison
        if self.comm.rank == 0:
            self.assertTrue(
                np.array([np.isclose(s, p, equal_nan=True).all()
                          for s, p in zip(ava_serial, ava_parallel)]).all())
            self.assertTrue((abs(F_fin_s - F_fin_p) < 1e-8).all())

    def test_gather_of_avalanches(self):
        """
        Test if a avalanche of "general" size can be gathered properly by the
        stochastic plasticity module, i.e. we have different sizes of the
        avalanche on the different cores.
        1. One quad point breaking on rank 0 and two quadpoints breaking
           on rank 1.
        """
        if self.comm.size != 2:
            # make this tests only for parallel evaluations on two processors
            return 0

        # ------- 1. ------- #
        # init parameters
        res = [3, 3, 3]
        lens = [1, 1, 1]
        dim = len(res)
        strain_xy_1 = 0.01
        plastic_increment = 1e-5

        # analytic compute eq_stress_1 for a given strain 'strain_xy_1'
        C = stiffness_matrix(self.young, self.poisson, self.dim)
        if self.formulation == µ.Formulation.finite_strain:
            F = np.eye(self.dim)
            F[0, 1] = strain_xy_1
            E = green_lagrangian_strain_vector(F, self.dim)
            PK2_analytic = PK2_tensor(C, E)
            eq_stress = sigma_eq(PK2_analytic)
        elif self.formulation == µ.Formulation.small_strain:
            eps = np.zeros((self.dim, self.dim))
            eps[0, 1] = strain_xy_1 / 2
            eps[1, 0] = eps[0, 1]
            eps_v = eps_voigt_vector(eps, self.dim)
            cauchy_analytic = PK2_tensor(C, eps_v)
            eq_stress = sigma_eq(cauchy_analytic)
            plastic_increment /= 2

        pixel_1 = [1, 1, 1]
        qpt_1 = 5
        pixel_2 = [0, 1, 2]
        qpt_2 = 0
        pixel_3 = [1, 2, 2]
        qpt_3 = 3

        # init material, with fine tuned order of stress thresholds
        fixed_yield_stress = np.ones(tuple(res) + (self.nb_quad_pts, ))*17
        fixed_yield_stress[tuple(pixel_1) + (qpt_1,)] = eq_stress
        fixed_yield_stress[tuple(pixel_2) + (qpt_2,)] = eq_stress
        fixed_yield_stress[tuple(pixel_3) + (qpt_3,)] = eq_stress
        cell = init_cell(res, lens, self.formulation,
                         self.gradient, self.fft, self.comm)
        mat = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
                       plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol,
                                   self.maxiter, self.verbose)

        # compute quad_pt_ids
        quad_pt_1 = compute_global_quad_pt_id_from_pixel(
            pixel_1, cell.nb_domain_grid_pts, cell.nb_quad_pts,
            pixel_quad_pt=qpt_1)
        quad_pt_2 = compute_global_quad_pt_id_from_pixel(
            pixel_2, cell.nb_domain_grid_pts, cell.nb_quad_pts,
            pixel_quad_pt=qpt_2)
        quad_pt_3 = compute_global_quad_pt_id_from_pixel(
            pixel_3, cell.nb_domain_grid_pts, cell.nb_quad_pts,
            pixel_quad_pt=qpt_3)

        # expected avalanche histories
        expected_ava_history = np.array([quad_pt_1, quad_pt_2, quad_pt_3])

        # propagate the avalanche
        yield_surface_accuracy = 1e-8  # for fast convergence
        n_max_bracket_search = 2
        # set the eigen strain field to the slightly modified previous fixed
        # deformation 'strain_xy' for faster convergence
        strain_field = cell.strain.array()
        DelF_init = np.zeros((dim, dim))
        if self.formulation == µ.Formulation.finite_strain:
            for i in range(self.dim):
                strain_field[i, i, ...] = 1.0

            strain_field[0, 1, ...] = \
                strain_xy_1 - yield_surface_accuracy**2*1.25
            DelF_init[0, 1] = yield_surface_accuracy**2
        elif self.formulation == µ.Formulation.small_strain:
            for i in range(self.dim):
                strain_field[i, i, ...] = 0.0

            strain_field[0, 1, ...] = \
                strain_xy_1/2 - yield_surface_accuracy**2/2*1.25
            strain_field[1, 0, ...] = strain_field[0, 1, ...]
            DelF_init[0, 1] = yield_surface_accuracy**2/2
            DelF_init[1, 0] = DelF_init[0, 1]

        n_max_avalanche = 2
        i_cdf = lambda z: 17  # constant value

        with warnings.catch_warnings():
            # suppress warnings of bracket_search()
            warnings.simplefilter("ignore")
            DelF, PK2, F, breaking_pixel = \
                sps.bracket_search(mat, cell, cg_solver,
                                   self.newton_tol,
                                   self.newton_equil_tol,
                                   yield_surface_accuracy,
                                   n_max_bracket_search,
                                   DelF_init, self.verbose_sps)

        def save_and_test_ava(n_strain_loop, ava_history):
            # check avalanche history
            self.assertTrue((ava_history == expected_ava_history).all())

        def save_fields(cell, n_strain_loop, before_avalanche):
            return 0

        sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
                                self.newton_equil_tol, PK2, F, n_max_avalanche,
                                self.verbose_sps,
                                inverse_cumulative_dist_func=i_cdf,
                                save_avalanche=save_and_test_ava,
                                save_fields=save_fields,
                                n_strain_loop=0)

    def test_empty_processors(self):
        """
        Tests:
        1. Test if stochastic plasticity search
           crashes if one processor is empty.
        """
        # ------- 1. ------- #
        # TODO(RLeute): this test should run for small strain!
        #     However, this might be not possible due to geometric constraints
        #     in small-strain. One might have to carfully choos the
        #     deformation and the yield thresholds to get this running.
        formulation = µ.Formulation.finite_strain

        DelF = np.zeros((self.dim, self.dim))
        if formulation == µ.Formulation.finite_strain:
            DelF[0, 1] = 0.0001
            F_tot = np.eye(self.dim)
            F_tot[0, 1] = 0.0002
        elif formulation == µ.Formulation.small_strain:
            DelF[0, 1] = 0.0001 / 2
            DelF[1, 0] = DelF[0, 1]
            F_tot = np.zeros((self.dim, self.dim))
            F_tot[0, 1] = 0.0002 / 2
            F_tot[1, 0] = F_tot[0, 1]

        res = [3, 3, 1]
        cell = init_cell(res, self.lens, formulation,
                         self.gradient, self.fft, self.comm)

        mat = init_mat(cell, self.young, self.poisson, self.yield_stress,
                       self.plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter,
                                   self.verbose)

        def save_avalanche(n_strain_loop, ava_history):
            return 0

        def save_fields(cell, n_strain_loop, before_avalanche):
            return 0

        sps.strain_cell(mat, cell, cg_solver, self.newton_tol,
                        self.newton_equil_tol, DelF, F_tot,
                        self.yield_surface_accuracy,
                        self.n_max_strain_loop,
                        self.n_max_bracket_search,
                        self.n_max_avalanche, self.verbose_sps, False,
                        self.inverse_cumulative_dist_func,
                        save_avalanche=save_avalanche,
                        save_fields=save_fields,
                        is_strain_initialised=False)


if __name__ == '__main__':
    unittest.main()
