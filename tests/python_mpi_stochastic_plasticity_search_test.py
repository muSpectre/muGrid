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

from python_test_imports import µ, muFFT
import muSpectre.stochastic_plasticity_search as sps
from python_stochastic_plasticity_search_test import \
    stiffness_matrix,  green_lagrangian_strain_vector, PK2_tensor,  sigma_eq, \
    init_cg_solver


def init_cell(res, lens, formulation, fft, comm):
    dim = len(res)
    grad = [µ.FourierDerivative(dim, i) for i in range(dim)] #fourier gradient
    return µ.Cell(res, lens, formulation, grad, fft, comm)

def init_mat(cell, young, poisson, yield_stress, plastic_increment, eigen_strain):
    mat = µ.material.MaterialStochasticPlasticity_3d.make(cell.wrapped_cell,
                                                          'test_mat')
    #init pixels
    for pixel in cell:
        mat.add_pixel(pixel, young, poisson, plastic_increment,
                      yield_stress[tuple(pixel)], eigen_strain)
    return mat


class StochasticPlasticitySearch_Check(unittest.TestCase):
    """
    Test the stochastic plasticity algorithm for correctness
    """
    def setUp(self):
        #parallel environment
        self.comm = MPI.COMM_WORLD
        self.fft = "fftwmpi"

        #cell parameters
        self.res  = [3, 3, 3]
        self.dim  = len(self.res)
        self.lens = [1, 1.2, 1]
        self.formulation = µ.Formulation.finite_strain

        #material parameters
        self.young = 210
        self.poisson = 0.33
        self.plastic_increment = 1e-5
        self.eigen_strain = np.zeros((self.dim, self.dim))
        np.random.seed(18*(self.comm.rank+1))
        self.yield_stress = np.random.random(tuple(self.res))
        #uniform distribution [a,b]
        mean = np.mean(self.yield_stress); std = np.std(self.yield_stress)
        a = mean - std; b = mean + std
        self.inverse_cumulative_dist_func = lambda z: a + (b-a)*z

        #solver parameter
        self.newton_tol       = 1e-6
        self.cg_tol           = 1e-6 #tolerance for cg algo
        self.newton_equil_tol = 1e-6 #tolerance for equilibrium
        self.maxiter          = 100
        self.verbose          = 0

        #stochastic plasticity maximum and accuracy parameters
        self.yield_surface_accuracy = 1e-6
        self.n_max_strain_loop      = 20
        self.n_max_bracket_search   = 30
        self.n_max_avalanche        = int(np.prod(np.array(self.res))*2/3)

        #set timing = True for timing information
        self.timing = False
        self.startTime = time.time()

    def tearDown(self):
        if self.timing:
            if self.comm.rank == 0:
                t = time.time() - self.startTime
                print("{}:\n{:.3f} seconds".format(self.id(), t))

    # def test_update_eigen_strain(self):
    #     """Test if the eigen strain is updated properly on each core"""
    #     cell = init_cell(self.res, self.lens,
    #                      self.formulation, self.fft, self.comm)
    #     mat  = init_mat(cell, self.young, self.poisson, self.yield_stress,
    #                     self.plastic_increment, self.eigen_strain)

    #     pixel = cell.subdomain_locations

    #     #read initial eigen strain
    #     init_strain = np.copy(mat.get_eigen_strain(pixel))

    #     #set stress on pixel [0,0,0] (only rank 0)
    #     if self.comm.rank == 0:
    #         stress = np.zeros((self.dim, self.dim))
    #         stress_1 = 0.3
    #         stress[0,1] = stress_1
    #         sps.update_eigen_strain(mat, pixel, stress, self.dim)

    #         #read out updated eigen strain and proof
    #         updated_strain = mat.get_eigen_strain(pixel)
    #         analytic_strain = init_strain
    #         analytic_strain[0,1] = \
    #             self.plastic_increment * stress_1 / np.sqrt(3/2 * stress_1**2)
    #         self.assertLess(
    #             np.linalg.norm(updated_strain - analytic_strain), 1e-8)

    #     else:
    #         # updated_strain = np.zeros(dim,dim) on all ranks != 0
    #         updated_strain = mat.get_eigen_strain(pixel)

    #     #update eigen strain
    #     stress = np.zeros((self.dim, self.dim))
    #     stress_2 = 0.4
    #     stress[1,0] = stress_2
    #     sps.update_eigen_strain(mat, pixel, stress, self.dim)

    #     #read out updated eigen strain and proof
    #     updated_strain_2 = mat.get_eigen_strain(pixel)
    #     analytic_strain = updated_strain
    #     analytic_strain[1,0] = \
    #         self.plastic_increment * stress_2 / np.sqrt(3/2 * stress_2**2)
    #     self.assertLess(
    #         np.linalg.norm(updated_strain_2 - analytic_strain), 1e-8)

    # def test_set_new_threshold(self):
    #     cell = init_cell(self.res, self.lens,
    #                      self.formulation, self.fft, self.comm)
    #     mat  = init_mat(cell, self.young, self.poisson, self.yield_stress,
    #                     self.plastic_increment, self.eigen_strain)

    #     pixel = cell.subdomain_locations
    #     # uniform distribution on the interval (a,b)
    #     a = 10
    #     b = 14
    #     inv_cum_dist_func = lambda z: a + (b-a)*z

    #     ### write first time a threshold on the pixel
    #     seed_1 = 19092019 * (self.comm.rank+1)
    #     np.random.seed(seed_1)
    #     sps.set_new_threshold(mat, pixel,
    #                           inverse_cumulative_dist_func = inv_cum_dist_func)
    #     np.random.seed(seed_1)
    #     threshold_expected = inv_cum_dist_func(np.random.random())
    #     threshold_read = mat.get_stress_threshold(pixel)
    #     self.assertLess(threshold_expected - threshold_read, 1e-8)

    #     ### write second time a threshold on the pixel
    #     seed_2 = 2019 * (self.comm.rank+1)
    #     np.random.seed(seed_2)
    #     sps.set_new_threshold(mat, pixel,
    #                           inverse_cumulative_dist_func = inv_cum_dist_func)
    #     np.random.seed(seed_2)
    #     threshold_expected = inv_cum_dist_func(np.random.random())
    #     threshold_read = mat.get_stress_threshold(pixel)
    #     self.assertLess(threshold_expected - threshold_read, 1e-8)

    # def test_propagate_avalanche_step(self):
    #     """
    #     Check if a single overloaded pixel breaks, at the right strain load.
    #     """
    #     cell = init_cell(self.res, self.lens,
    #                      self.formulation, self.fft, self.comm)
    #     strain_xy = 0.1
    #     weak_pixel = np.array(tuple(cell.subdomain_locations))
    #     weak_pixels = np.empty(self.comm.Get_size()*3, dtype = int)
    #     self.comm.Allgather([weak_pixel, MPI.INT], [weak_pixels, MPI.INT])
    #     weak_pixels = weak_pixels.reshape((self.comm.Get_size(), 3))

    #     ### analytic
    #     #analytic compute the equivalent stress for a given strain 'strain_xy'
    #     F = np.eye(self.dim)
    #     F[0,1] = strain_xy
    #     C = stiffness_matrix(self.young, self.poisson, self.dim)
    #     E = green_lagrangian_strain_vector(F, self.dim)
    #     PK2_analytic = PK2_tensor(C, E)
    #     PK1_analytic = np.dot(F, PK2_analytic)
    #     eq_stress = sigma_eq(PK1_analytic) #analytic computed equivalent stress

    #     ### numeric
    #     #set the analytic computed equivalent stress reduced by a tiny amount as
    #     #threshold for one "weak pixel" and the other thresholds to slightly
    #     #higher values.
    #     fixed_yield_stress = np.ones(tuple(self.res)) * eq_stress * (1 + 1e-8)
    #     for index in weak_pixels:
    #         fixed_yield_stress[tuple(index)] = eq_stress * (1 - 1e-8)

    #     #init material
    #     mat  = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
    #                     self.plastic_increment, self.eigen_strain)

    #     cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter, self.verbose)
    #     #set the eigen strain field to the previous fixed deformation 'strain_xy'
    #     strain_field = cell.strain
    #     for i in range(self.dim):
    #         strain_field[i,i,...] = 1.0
    #     strain_field[0,1,...] = strain_xy
    #     #check if you can find the yielding pixel
    #     overloaded_pixels = \
    #         sps.propagate_avalanche_step(mat, cell.wrapped_cell, self.dim,
    #                                      cg_solver, self.newton_tol,
    #                                      self.newton_equil_tol, self.verbose)

    #     # one overloaded pixel per core
    #     self.assertTrue(len(overloaded_pixels) == 1)
    #     self.assertTrue((overloaded_pixels == weak_pixel).all())

    # def test_bracket_search(self):
    #     """
    #     Tests:
    #     1. Test if bracket search find the exact yield point for one pixel with
    #        a lower yield threshold than the others.
    #     2. Test exception for two/n pixels with very close yield criterion. Thus
    #        they should break together(hence avalanche can start for n>= 2 pixel)
    #     3. Test if an error is raised when the maximum allowed bracket steps are
    #        reached.
    #     """
    #     ### ------- 1. ------- ###
    #     # init data
    #     low_yield_stress = 10.0
    #     yield_surface_accuracy = 1e-7
    #     n_max_bracket_search   = 4

    #     #set the initial deformation close to the final deformation to reduce
    #     #the needed bracket search steps
    #     g_01 = 0.07268800332367435 #final deformation
    #     strain_init = np.array([[1, g_01 - yield_surface_accuracy*1.75, 0],
    #                             [0,  1  ,0],
    #                             [0,  0  ,1]])

    #     fixed_yield_stress = np.ones(tuple(self.res))*14 #high threshold
    #     fixed_yield_stress[0,0,0] = low_yield_stress
    #     cell = init_cell(self.res, self.lens,
    #                      self.formulation, self.fft, self.comm)
    #     mat  = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
    #                     self.plastic_increment, self.eigen_strain)
    #     cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter, self.verbose)

    #     DelF_initial = np.zeros((self.dim, self.dim))
    #     DelF_initial[0,1] = yield_surface_accuracy

    #     #initialize cell with unit-matrix deformation gradient
    #     cell_strain = cell.strain
    #     cell_strain[:] = np.tensordot(strain_init,
    #                                   np.ones(cell.nb_subdomain_grid_pts),
    #                                   axes=0)

    #     #if self.comm.rank == 0:
    #     next_DelF_guess, PK2, F, breaking_pixel = \
    #             sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
    #                                self.newton_equil_tol,
    #                                yield_surface_accuracy, n_max_bracket_search,
    #                                DelF_initial, self.verbose)

    #     # #Is it exactly one breaking pixel and if yes is it pixel [0,0,0]?
    #     self.assertEqual(len(breaking_pixel), 1)
    #     self.assertTrue((breaking_pixel[0] == [0, 0, 0]).all())

    #     ### plug in the numeric result into the analytic formula and see if one
    #     #   gets out the exact yield stress (low_yield_stress)
    #     F_numeric = µ.gradient_integration.reshape_gradient(
    #         F, cell.nb_subdomain_grid_pts)[(0,)*self.dim]
    #     PK2_numeric = µ.gradient_integration.reshape_gradient(
    #         PK2, cell.nb_subdomain_grid_pts)[(0,)*self.dim]
    #     C = stiffness_matrix(self.young, self.poisson, self.dim)
    #     E = green_lagrangian_strain_vector(F_numeric, self.dim)
    #     PK2_analytic = PK2_tensor(C, E)
    #     #TODO(RLeute): change to PK2 if material_stochastic_plasticity returns PK2
    #     #eq_pk2    = sigma_eq(PK2_analytic)
    #     #print("eq_stress PK2: ", eq_pk2)
    #     PK1_analytic = np.dot(F_numeric, PK2_analytic)
    #     eq_stress = sigma_eq(PK1_analytic)

    #     #Is the analytic yield stress equivalent to yield stress of pix(0,0,0)?
    #     self.assertLess(np.abs(low_yield_stress - eq_stress), 1e-7)

    #     #Is the computed deformation gradient F_numeric correct?
    #     F_yield10 = np.array([[1, g_01,0],
    #                           [0,  1  ,0],
    #                           [0,  0  ,1]])
    #     self.assertLess(np.linalg.norm(F_yield10-F_numeric),
    #                       yield_surface_accuracy)
    #     # compare the computed stress (here PK1 vs PK2 because stochastic
    #     # plasticity computes PK1 up to now but has to be changed to PK2)
    #     self.assertLess(np.linalg.norm(PK1_analytic-PK2_numeric), 1e-8)

    #     ### ------- 2. ------- ###
    #     # init data
    #     low_yield_stress = 10.0
    #     yield_surface_accuracy = 1e-8
    #     small_yield_difference = \
    #                 low_yield_stress * yield_surface_accuracy**2 * 1e-2
    #     #set the initial deformation close to the final deformation to reduce
    #     #the needed bracket search steps
    #     g_01 = 0.07268800332367393591 #final deformation
    #     strain_init = np.array([[1, g_01 - yield_surface_accuracy**2*1.75, 0],
    #                             [0,  1  ,0],
    #                             [0,  0  ,1]])
    #     n_max_bracket_search   = 4

    #     fixed_yield_stress = np.ones(tuple(self.res))*14 #high threshold
    #     fixed_yield_stress[0,0,0] = low_yield_stress
    #     fixed_yield_stress[tuple([i//2+1 for i in self.res])] = \
    #         low_yield_stress + small_yield_difference
    #     cell = init_cell(self.res, self.lens,
    #                      self.formulation, self.fft, self.comm)
    #     mat  = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
    #                     self.plastic_increment, self.eigen_strain)
    #     cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter, self.verbose)
    #     DelF_initial = np.zeros((self.dim, self.dim))
    #     DelF_initial[0,1] = yield_surface_accuracy**2

    #     #initialize cell with deformation gradient for fast convergence
    #     cell_strain = cell.strain
    #     cell_strain[:] = np.tensordot(strain_init,
    #                                   np.ones(cell.nb_subdomain_grid_pts),
    #                                   axes=0)

    #     with warnings.catch_warnings(record=True) as w:
    #         warnings.simplefilter("always") #all warnings be triggered.
    #         next_DelF_guess, PK2, F, breaking_pixel = \
    #             sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
    #                                self.newton_equil_tol,
    #                                yield_surface_accuracy, n_max_bracket_search,
    #                                DelF_initial, self.verbose)
    #         self.assertTrue(len(w) == 1)
    #         self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
    #         self.assertTrue("bracket_search found 2 pixels overcoming their "
    #                 "yield threshold for the final deformation. To initialise "
    #                 "the avalanche think about using the parameter 'single_pixel"
    #                 "_start' of 'propagate_avalanche()' to start the avalanche "
    #                 "from a single pixel!" == str(w[-1].message))

    #     #Are there exactly two breaking pixels, [0,0,0] and [nx//2,ny//2,nz//2]?
    #     self.assertEqual(len(breaking_pixel), 2)
    #     self.assertTrue(
    #         (breaking_pixel == [[0,0,0], [i//2+1 for i in self.res]]).all())

    #     ### ------- 3. ------- ###
    #     # use the initalization from the last test
    #     n_max_bracket_search = 2
    #     DelF_initial = np.zeros((self.dim, self.dim))
    #     DelF_initial[0,1] = 0.1
    #     with self.assertRaises(RuntimeError):
    #         sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
    #                            self.newton_equil_tol, yield_surface_accuracy,
    #                            n_max_bracket_search, DelF_initial, self.verbose)

    # def test_propagate_avalanche(self):
    #     """
    #     Tests:
    #     1. Test if plastic deformations are done in the right order and at the
    #        right place!
    #     2. Test if initially_overloaded_pixels behaves right.
    #     3. Test the parameter "single_pixel_start"
    #     """
    #     ### ------- 1. ------- ###
    #     ### init parameters
    #     res  = [3,3,3]
    #     lens = [1,1,1]
    #     dim  = len(res)
    #     strain_xy_1 = 0.01
    #     plastic_increment = strain_xy_1 * 10
    #     if self.comm.Get_size() == 1:
    #         expected_ava_history = np.array([[[ 1.,  1.,  1.],
    #                                           [np.nan, np.nan, np.nan]],
    #                                          [[ 1.,  2.,  1.],
    #                                           [ 2.,  2.,  1.]]])
    #     if self.comm.Get_size() == 2:
    #         expected_ava_history = np.array([[[ 1.,  1.,  1.],
    #                                           [np.nan, np.nan, np.nan]],
    #                                          [[ 1.,  2.,  1.],
    #                                           [ 2.,  2.,  1.]],
    #                                          [[  2.  ,   1.  ,   2.  ],
    #                                           [np.nan, np.nan, np.nan]]])

    #     ### analytic compute eq_stress_1 for a given strain 'strain_xy_1'
    #     F = np.eye(self.dim)
    #     F[0,1] = strain_xy_1
    #     C = stiffness_matrix(self.young, self.poisson, self.dim)
    #     E = green_lagrangian_strain_vector(F, self.dim)
    #     PK2_analytic = PK2_tensor(C, E)
    #     PK1_analytic = np.dot(F, PK2_analytic)
    #     eq_stress = sigma_eq(PK1_analytic) #analytic computed equivalent stress

    #     eq_stress_1 = eq_stress
    #     eq_stress_2 = eq_stress * 1.05
    #     eq_stress_3 = eq_stress * 1.50

    #     ### init material, with fine tuned order of stress thresholds
    #     fixed_yield_stress = np.ones(tuple(res))*17 #high threshold
    #     fixed_yield_stress[1,1,1] = eq_stress_1
    #     fixed_yield_stress[2,2,1] = eq_stress_2
    #     fixed_yield_stress[1,2,1] = eq_stress_2
    #     if self.comm.Get_size() == 2:
    #         fixed_yield_stress[2,1,2] = eq_stress_3
    #     cell = init_cell(res, lens, self.formulation, self.fft, self.comm)
    #     mat  = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
    #                     plastic_increment, self.eigen_strain)
    #     cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter, self.verbose)

    #     ### overload one pixel which breaks and by its plastic increment
    #     ### overloads two additional pixels.

    #     #propagate the avalanche
    #     yield_surface_accuracy = 1e-8
    #     n_max_bracket_search = 5
    #     #set the eigen strain field to the slightly modified previous fixed
    #     #deformation 'strain_xy' for faster convergence
    #     strain_field = cell.strain
    #     for i in range(self.dim):
    #         strain_field[i,i] = 1.0
    #     strain_field[0,1] = strain_xy_1 - yield_surface_accuracy*1.25
    #     DelF_init = np.zeros((dim, dim))
    #     DelF_init[0,1] = yield_surface_accuracy
    #     n_max_avalanche = 10
    #     i_cdf = lambda z: 17 #constant value

    #     DelF, PK2, F, breaking_pixel = \
    #         sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
    #                            self.newton_equil_tol, yield_surface_accuracy,
    #                            n_max_bracket_search, DelF_init, self.verbose)

    #     def save_and_test_ava(n_strain_loop, ava_history):
    #         # check avalanche history
    #         self.assertTrue(np.isclose(ava_history, expected_ava_history,
    #                                    equal_nan=True).all())

    #     nc_save_1 = muFFT.NCStructuredGrid('stored_stress_strain_1.nc', 'w',
    #                     nb_domain_grid_pts = cell.nb_domain_grid_pts,
    #                     decomposition = 'subdomain',
    #                     subdomain_locations = cell.subdomain_locations,
    #                     nb_subdomain_grid_pts = cell.nb_subdomain_grid_pts,
    #                     communicator = self.comm)
    #     def save_stress_strain_1(n_strain_loop, PK2_initial, F_initial,
    #                              PK2_final, F_final, cell):
    #         #Collect the data from the processors (subdomains) and write it into
    #         #a global (domain) stress and strain field.
    #         nc_save_1[n_strain_loop].F_initial = F_initial
    #         nc_save_1[n_strain_loop].F_final = F_final
    #         nc_save_1[n_strain_loop].PK2_initial = PK2_initial
    #         nc_save_1[n_strain_loop].PK2_final = PK2_final
    #         nc_save_1.close()

    #     #initial pixel is [1,1,1]
    #     self.assertTrue((breaking_pixel == [[1,1,1]]).all())

    #     sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
    #         self.newton_equil_tol, PK2, F, n_max_avalanche, self.verbose,
    #         inverse_cumulative_dist_func = i_cdf,
    #         save_avalanche = save_and_test_ava,
    #         save_stress_strain = save_stress_strain_1,
    #         n_strain_loop = 0)

    #     #Check if at least stress and strain have the right dimensions after
    #     #collecting it in "save_stress_and_strain"
    #     nc_read_1 = muFFT.NCStructuredGrid('stored_stress_strain_1.nc', 'r')[0]
    #     PK2_in_1 = nc_read_1.PK2_initial
    #     F_in_1   = nc_read_1.F_initial
    #     PK2_fi_1 = nc_read_1.PK2_final
    #     F_fi_1   = nc_read_1.F_final
    #     self.assertTrue(PK2_in_1.shape == tuple(res) + (dim,)*2)
    #     self.assertTrue(F_in_1.shape   == tuple(res) + (dim,)*2)
    #     self.assertTrue(PK2_fi_1.shape == tuple(res) + (dim,)*2)
    #     self.assertTrue(F_fi_1.shape   == tuple(res) + (dim,)*2)
    #     #check the initial stress and the initial strain
    #     loc = cell.subdomain_locations
    #     nb_pts = cell.nb_subdomain_grid_pts
    #     self.assertTrue((µ.gradient_integration.reshape_gradient(PK2, nb_pts) ==
    #                      PK2_in_1[loc[0]:loc[0]+nb_pts[0],
    #                               loc[1]:loc[1]+nb_pts[1],
    #                               loc[2]:loc[2]+nb_pts[2]]).all())
    #     self.assertTrue((µ.gradient_integration.reshape_gradient(F, nb_pts) ==
    #                      F_in_1[loc[0]:loc[0]+nb_pts[0],
    #                             loc[1]:loc[1]+nb_pts[1],
    #                             loc[2]:loc[2]+nb_pts[2]]).all())

    #     ### ------- 2. ------- ###
    #     cell = init_cell(res, lens, self.formulation, self.fft, self.comm)
    #     mat  = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
    #                     plastic_increment, self.eigen_strain)
    #     cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter, self.verbose)

    #     strain_field = cell.strain
    #     for i in range(self.dim):
    #         strain_field[i,i] = 1.0
    #     strain_field[0,1] = strain_xy_1 - yield_surface_accuracy*1.25

    #     DelF, PK2, F, breaking_pixel = \
    #         sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
    #                            self.newton_equil_tol, yield_surface_accuracy,
    #                            n_max_bracket_search, DelF_init, self.verbose)

    #     nc_save_2 = muFFT.NCStructuredGrid('stored_stress_strain_2.nc', 'w',
    #                     nb_domain_grid_pts = cell.nb_domain_grid_pts,
    #                     decomposition = 'subdomain',
    #                     subdomain_locations = cell.subdomain_locations,
    #                     nb_subdomain_grid_pts = cell.nb_subdomain_grid_pts,
    #                     communicator = self.comm)
    #     def save_stress_strain_2(n_strain_loop, PK2_initial, F_initial,
    #                              PK2_final, F_final, cell):
    #         #Collect the data from the processors (subdomains) and write it into
    #         #a global (domain) stress and strain field.
    #         nc_save_2[n_strain_loop].F_initial = F_initial
    #         nc_save_2[n_strain_loop].F_final = F_final
    #         nc_save_2[n_strain_loop].PK2_initial = PK2_initial
    #         nc_save_2[n_strain_loop].PK2_final = PK2_final
    #         nc_save_2.close()

    #     #initial pixel is [1,1,1]
    #     self.assertTrue((breaking_pixel == [[1,1,1]]).all())

    #     sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
    #         self.newton_equil_tol, PK2, F, n_max_avalanche, self.verbose,
    #         initially_overloaded_pixels = breaking_pixel,
    #         inverse_cumulative_dist_func = i_cdf,
    #         save_avalanche = save_and_test_ava,
    #         save_stress_strain = save_stress_strain_2,
    #         n_strain_loop = 0)

    #     #Check if at least stress and strain have the right dimensions after
    #     #collecting it in "save_stress_and_strain"
    #     nc_read_2 = muFFT.NCStructuredGrid('stored_stress_strain_2.nc', 'r')[0]
    #     PK2_in_2 = nc_read_2.PK2_initial
    #     F_in_2   = nc_read_2.F_initial
    #     PK2_fi_2 = nc_read_2.PK2_final
    #     F_fi_2   = nc_read_2.F_final
    #     self.assertTrue(PK2_in_2.shape == tuple(res) + (dim,)*2)
    #     self.assertTrue(F_in_2.shape   == tuple(res) + (dim,)*2)
    #     self.assertTrue(PK2_fi_2.shape == tuple(res) + (dim,)*2)
    #     self.assertTrue(F_fi_2.shape   == tuple(res) + (dim,)*2)
    #     #check the initial stress and the initial strain
    #     loc = cell.subdomain_locations
    #     nb_pts = cell.nb_subdomain_grid_pts
    #     self.assertTrue((µ.gradient_integration.reshape_gradient(PK2, nb_pts) ==
    #                      PK2_in_2[loc[0]:loc[0]+nb_pts[0],
    #                               loc[1]:loc[1]+nb_pts[1],
    #                               loc[2]:loc[2]+nb_pts[2]]).all())
    #     self.assertTrue((µ.gradient_integration.reshape_gradient(F, nb_pts) ==
    #                      F_in_2[loc[0]:loc[0]+nb_pts[0],
    #                             loc[1]:loc[1]+nb_pts[1],
    #                             loc[2]:loc[2]+nb_pts[2]]).all())

    #     #Check if you have the same results in the two nc files
    #     self.assertTrue((PK2_in_1 == PK2_in_2).all())
    #     self.assertTrue((F_in_1 == F_in_2).all())
    #     self.assertTrue((PK2_fi_1 == PK2_fi_2).all())
    #     self.assertTrue((F_fi_1 == F_fi_2).all())

    #     ### CLEAN UP
    #     if self.comm.rank == 0:
    #         pwd = os.getcwd()
    #         os.remove(pwd + "/stored_stress_strain_1.nc")
    #         os.remove(pwd + "/stored_stress_strain_2.nc")

    #     ### ------- 3. ------- ###
    #     ### init material, with two pixels of the same yield strength
    #     fixed_yield_stress = np.ones(tuple(res))*17 #high threshold
    #     fixed_yield_stress[1,1,1] = eq_stress_1
    #     fixed_yield_stress[2,2,2] = eq_stress_1
    #     yield_surface_accuracy = 1e-8
    #     n_max_bracket_search = 5
    #     DelF_init = np.zeros((dim, dim))
    #     DelF_init[0,1] = yield_surface_accuracy**2
    #     def setup_material():
    #         #helper function to set up the material which is needed four times
    #         cell = init_cell(res, lens, self.formulation, self.fft, self.comm)
    #         mat  = init_mat(cell, self.young, self.poisson, fixed_yield_stress,
    #                         plastic_increment, self.eigen_strain)
    #         cg_solver = init_cg_solver(cell, self.cg_tol,
    #                                    self.maxiter, self.verbose)

    #         #set the eigen strain field to the previous fixed deformation
    #         strain_field = cell.strain
    #         for i in range(self.dim):
    #             strain_field[i,i,...] = 1.0
    #         strain_field[0,1,...] = strain_xy_1 - yield_surface_accuracy**2*1.25

    #         with warnings.catch_warnings():
    #             #suppress warnings of bracket_search()
    #             warnings.simplefilter("ignore")
    #             DelF, PK2, F, breaking_pixel = \
    #                 sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
    #                             self.newton_equil_tol, yield_surface_accuracy,
    #                             n_max_bracket_search, DelF_init, self.verbose)
    #         return mat, cell, cg_solver, PK2, F, breaking_pixel

    #     def sa_2break(n_strain_loop, ava_history):
    #         self.assertTrue(ava_history.shape == (1,2,3))

    #     def sa_1break(n_strain_loop, ava_history):
    #         self.assertTrue(ava_history.shape == (2,1,3))

    #     #Check for all combinations of initially_overloaded_pixels and
    #     #single_pixel_start
    #     mat, cell, cg_solver, PK2, F, breaking_pixel = setup_material()
    #     self.assertTrue((breaking_pixel == [[1,1,1], [2,2,2]]).all())
    #     sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
    #         self.newton_equil_tol, PK2, F, n_max_avalanche, self.verbose,
    #         initially_overloaded_pixels = None, single_pixel_start = False,
    #         inverse_cumulative_dist_func = i_cdf, save_avalanche = sa_2break,
    #         save_stress_strain = None, n_strain_loop = 0)

    #     mat, cell, cg_solver, PK2, F, breaking_pixel = setup_material()
    #     sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
    #         self.newton_equil_tol, PK2, F, n_max_avalanche, self.verbose,
    #         initially_overloaded_pixels = breaking_pixel,
    #         single_pixel_start = False, inverse_cumulative_dist_func = i_cdf,
    #         save_avalanche = sa_2break, save_stress_strain = None,
    #         n_strain_loop = 0)

    #     mat, cell, cg_solver, PK2, F, breaking_pixel = setup_material()
    #     sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
    #         self.newton_equil_tol, PK2, F, n_max_avalanche, self.verbose,
    #         initially_overloaded_pixels = breaking_pixel,
    #         single_pixel_start = True, inverse_cumulative_dist_func = i_cdf,
    #         save_avalanche = sa_1break, save_stress_strain = None,
    #         n_strain_loop = 0)

    #     mat, cell, cg_solver, PK2, F, breaking_pixel = setup_material()
    #     sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
    #         self.newton_equil_tol, PK2, F, n_max_avalanche, self.verbose,
    #         initially_overloaded_pixels = None, single_pixel_start = True,
    #         inverse_cumulative_dist_func = i_cdf, save_avalanche = sa_1break,
    #         save_stress_strain = None, n_strain_loop = 0)

    # def test_strain_cell(self):
    #     """
    #     Tests:
    #     1. Test if the function reaches the required deformation
    #     2. Small deformation with only one avalanche. Check:
    #         - avalanche pixel index
    #         - PK2, stress field
    #         - F, deformation gradient field
    #     """
    #     ### ------- 1. ------- ###
    #     DelF      = np.zeros((self.dim,self.dim))
    #     DelF[0,1] = 0.0001
    #     F_tot      = np.eye(self.dim)
    #     F_tot[0,1] = 0.0002
    #     cell = init_cell(self.res, self.lens, self.formulation, self.fft,
    #                      self.comm)
    #     mat  = init_mat(cell, self.young, self.poisson, self.yield_stress,
    #                     self.plastic_increment, self.eigen_strain)
    #     cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter,
    #                                self.verbose)
    #     F_fin = sps.strain_cell(mat, cell, cg_solver, self.newton_tol,
    #                             self.newton_equil_tol, DelF, F_tot,
    #                             self.yield_surface_accuracy,
    #                             self.n_max_strain_loop,
    #                             self.n_max_bracket_search,
    #                             self.n_max_avalanche, self.verbose, False,
    #                             self.inverse_cumulative_dist_func,
    #                             save_avalanche = None,
    #                             save_stress_strain = None,
    #                             is_strain_initialised = False)
    #     #is the reached deformation larger or almost equal to the required one
    #     self.assertTrue(((F_fin-F_tot) > -1e-16).all())

    # def test_parallel_vs_serial(self):
    #     """
    #     Check if the parallel and serial version of stochastic plasticity lead
    #     to the same results.
    #     """
    #     if self.comm.size == 1:
    #         #execute test_parallel_vs_serial only for size > 1.
    #         #It works for size = 1 but we save the time for this execution.
    #         return 0

    #     ### setup
    #     res = [3,3,3]
    #     yield_stress = 1.5
    #     #set the same random seed on each core, otherwise the serial and
    #     #parallel versions dont come to the same result because they have
    #     #different random numbers.
    #     np.random.seed(12345)
    #     fixed_yield_stress = 0.90 * yield_stress \
    #                          + np.random.random(res) * 0.20
    #     a = yield_stress * 0.90; b = yield_stress * 1.10
    #     i_cdf = lambda z: a + (b-a)*z

    #     g_01 = 0.00988425
    #     strain_init = np.array([[1, g_01 - self.yield_surface_accuracy*1.75, 0],
    #                             [0,  1  ,0],
    #                             [0,  0  ,1]])
    #     DelF      = np.zeros((self.dim,self.dim))
    #     DelF[0,1] = self.yield_surface_accuracy
    #     F_tot      = np.eye(self.dim)
    #     F_tot[0,1] = 0.01

    #     n_max_bracket_search = 20
    #     n_max_strain_loop = 5
    #     plastic_increment = 4e-3 #1e-3 leads to different avalanche
    #     ava_serial = []
    #     ava_parallel = []
    #     def save_and_test_ava_s(n_strain_loop, ava_history):
    #         ava_serial.append(ava_history)
    #     def save_and_test_ava_p(n_strain_loop, ava_history):
    #         ava_parallel.append(ava_history)

    #     ### serial
    #     #create a new communicator with only one rank to make a effective
    #     #(cell with comm that has only one rank leads to a serial
    #     #computation, no mpi) serial computation.
    #     comm_s = self.comm.Split(color = self.comm.rank)
    #     if self.comm.rank == 0:
    #         fft_s = "fftw"
    #         cell_s = init_cell(res, self.lens, self.formulation, fft_s, comm_s)
    #         mat_s  = init_mat(cell_s, self.young, self.poisson,
    #                           fixed_yield_stress, plastic_increment,
    #                           self.eigen_strain)
    #         cg_solver_s = init_cg_solver(cell_s, self.cg_tol,
    #                                      self.maxiter, self.verbose)
    #         c_strain = cell_s.strain
    #         c_strain[:] = np.tensordot(
    #             strain_init, np.ones(cell_s.nb_domain_grid_pts), axes=0)
    #         F_fin_s = sps.strain_cell(mat_s, cell_s, cg_solver_s,
    #                                   self.newton_tol, self.newton_equil_tol,
    #                                   DelF, F_tot, self.yield_surface_accuracy,
    #                                   n_max_strain_loop, n_max_bracket_search,
    #                                   self.n_max_avalanche, self.verbose, False,
    #                                   i_cdf, save_avalanche = save_and_test_ava_s,
    #                                   save_stress_strain = None,
    #                                   is_strain_initialised = True)

    #     ### parallel
    #     fft_p = "fftwmpi"
    #     comm_p = self.comm
    #     cell_p = init_cell(res, self.lens, self.formulation, fft_p, comm_p)
    #     mat_p  = init_mat(cell_p, self.young, self.poisson, fixed_yield_stress,
    #                       plastic_increment, self.eigen_strain)
    #     cg_solver_p = init_cg_solver(cell_p, self.cg_tol,
    #                                  self.maxiter, self.verbose)
    #     c_strain = cell_p.strain
    #     c_strain[:] = np.tensordot(
    #         strain_init, np.ones(cell_p.nb_subdomain_grid_pts), axes=0)
    #     F_fin_p = sps.strain_cell(mat_p, cell_p, cg_solver_p, self.newton_tol,
    #                               self.newton_equil_tol, DelF, F_tot,
    #                               self.yield_surface_accuracy,
    #                               n_max_strain_loop, n_max_bracket_search,
    #                               self.n_max_avalanche, self.verbose, False,
    #                               i_cdf, save_avalanche = save_and_test_ava_p,
    #                               save_stress_strain = None,
    #                               is_strain_initialised = True)

    #     ### comaparison
    #     if self.comm.rank == 0:
    #         self.assertTrue(np.array([np.isclose(s, p, equal_nan=True).all()
    #             for s,p in zip(ava_serial, ava_parallel)]).all())
    #         self.assertTrue((abs(F_fin_s - F_fin_p) < 1e-8).all())

    # def test_empty_processors(self):
    #     """
    #     Tests:
    #     1. Test if stochastic plasticity search crashes if one processor is empty
    #     """
    #     ### ------- 1. ------- ###
    #     #TODO(rleute):
    #     #check this test when the parallel version of muSpectre runs again
    #     DelF      = np.zeros((self.dim,self.dim))
    #     DelF[0,1] = 0.0001
    #     F_tot      = np.eye(self.dim)
    #     F_tot[0,1] = 0.0002
    #     res = [3, 3, 1]
    #     cell = init_cell(res, self.lens, self.formulation, self.fft, self.comm)
    #     mat  = init_mat(cell, self.young, self.poisson, self.yield_stress,
    #                     self.plastic_increment, self.eigen_strain)
    #     cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter,
    #                                self.verbose)
    #     def save_avalanche(n_strain_loop, ava_history):
    #         return 0
    #     def save_stress_strain(n_strain_loop, PK2_initial, F_initial,
    #                            PK2_final, F_final, cell):
    #         return 0
    #     F_fin = sps.strain_cell(mat, cell, cg_solver, self.newton_tol,
    #                             self.newton_equil_tol, DelF, F_tot,
    #                             self.yield_surface_accuracy,
    #                             self.n_max_strain_loop,
    #                             self.n_max_bracket_search,
    #                             self.n_max_avalanche, self.verbose, False,
    #                             self.inverse_cumulative_dist_func,
    #                             save_avalanche = save_avalanche,
    #                             save_stress_strain = save_stress_strain,
    #                             is_strain_initialised = False)

if __name__ == '__main__':
    unittest.main()
