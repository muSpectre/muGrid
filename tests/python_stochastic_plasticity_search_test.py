#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_stochastic_plasticity_search_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   19 Mär 2019

@brief  test for the stochastic plasticity search algorithm in file
        stochastic_plasticity_search.py

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

import unittest
import time
import warnings
import numpy as np

from python_test_imports import µ
import muSpectre.stochastic_plasticity_search as sps

def init_cell(res, lens, formulation):
    return µ.Cell(res, lens, formulation)

def init_material(res, lens, formulation, young, poisson,
                  yield_stress, plastic_increment, eigen_strain):
    cell = init_cell(res, lens, formulation)
    mat = µ.material.MaterialStochasticPlasticity_3d.make(cell,
                                                          'test_mat')
    #init pixels
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel(pixel_id, young, poisson, plastic_increment,
                      yield_stress[tuple(pixel)], eigen_strain)
    cell.initialise()
    return cell, mat

def stiffness_matrix(young, poisson, dim):
    nu = poisson
    if dim == 2:
        C = young/((1+nu)*(1-2*nu)) *\
            np.array([[1-nu, nu  , 0         ],
                      [nu  , 1-nu, 0         ],
                      [0   , 0   , (1-2*nu)/2]])
    if dim == 3:
        C = young/((1+nu)*(1-2*nu)) *\
            np.array([[1-nu, nu  , nu  , 0         , 0         , 0         ],
                      [nu  , 1-nu, nu  , 0         , 0         , 0         ],
                      [nu  , nu  , 1-nu, 0         , 0         , 0         ],
                      [0   , 0   , 0   , (1-2*nu)/2, 0         , 0         ],
                      [0   , 0   , 0   , 0         , (1-2*nu)/2, 0         ],
                      [0   , 0   , 0   , 0         ,          0, (1-2*nu)/2]])
    return C

def green_lagrangian_strain_vector(F, dim):
    green_tensor = np.dot(F.T,F)
    gl_tensor = 1/2 * (green_tensor - np.eye(dim))
    gl_vector = np.append(np.diagonal(gl_tensor), 2*gl_tensor[0, 1:])
    if dim == 3:
        gl_vector = np.append(gl_vector, 2*gl_tensor[1,2])
    return gl_vector

def PK2_tensor(C, E):
    PK2_vector = np.dot(C, E)
    dim = int(len(E)/2)
    PK2 = np.zeros((dim, dim))
    np.fill_diagonal(PK2, PK2_vector[:dim])
    PK2[0,1] = PK2_vector[dim]
    PK2[1,0] = PK2_vector[dim]
    if dim == 3:
        PK2[0,2] = PK2_vector[dim+1]
        PK2[2,0] = PK2_vector[dim+1]
        PK2[1,2] = PK2_vector[dim+2]
        PK2[2,1] = PK2_vector[dim+2]

    return PK2

def sigma_eq(sigma):
    dim = sigma.shape[0]
    if dim == 3:
        sigma_dev = sigma - 1/dim * np.trace(sigma)*np.eye(dim)
        sigma_eq  = np.sqrt(3/2 * np.tensordot(sigma_dev, sigma_dev))
    elif dim == 2:
        sigma_eq = np.sqrt(sigma[0,0]**2 + sigma[1,1]**2
                           - sigma[0,0]*sigma[1,1] + 3 * sigma[0,1]**2)
    elif dim == 1:
        sigma_eq = sigma[0,0]
    else:
        raise RuntimeError("The von Mises equivalent stress is not defined for "
                           "{}D".format(dim))
    return sigma_eq

def init_cg_solver(cell, cg_tol, maxiter, verbose):
    return µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)


class StochasticPlasticitySearch_Check(unittest.TestCase):
    """
    Test the stochastic plasticity algorithm for correctness
    """

    def setUp(self):
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
        np.random.seed(18)
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
        self.verbose          = µ.Verbosity.Silent

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
            t = time.time() - self.startTime
            print("{}:\n{:.3f} seconds".format(self.id(), t))

    def test_compute_deviatoric_stress(self):
        for dim in [2,3]:
            p = 2*np.eye(dim)
            fluctuations = np.random.random((dim, dim))
            np.fill_diagonal(fluctuations, 0)
            stress = fluctuations + p
            dev_stress = sps.compute_deviatoric_stress(stress, dim)

            self.assertLess(np.linalg.norm(dev_stress - fluctuations), 1e-8)

    def test_compute_equivalent_von_Mises_stress(self):
        dim = 3
        gamma_01 = 0.1
        stress = np.eye(dim)
        stress[0,1] = gamma_01
        eq_stress = sps.compute_equivalent_von_Mises_stress(stress, dim)

        self.assertLess(eq_stress - np.sqrt(3/2*gamma_01**2), 1e-8)

    def test_compute_strain_direction(self):
        dim = 3
        gamma_12 = 0.2
        stress = np.eye(dim)
        stress[1,2] = gamma_12
        strain_direc_analytic = np.zeros((dim, dim))
        strain_direc_analytic[1,2] = gamma_12 / np.sqrt(3/2 * gamma_12**2)
        strain_direc = sps.compute_strain_direction(stress, dim)

        self.assertLess(
            np.linalg.norm(strain_direc - strain_direc_analytic), 1e-8)

    def test_update_eigen_strain(self):
        cell, mat = init_material(self.res, self.lens, self.formulation,
                                  self.young, self.poisson, self.yield_stress,
                                  self.plastic_increment, self.eigen_strain)
        pixel = list(range(self.dim))

        pixel_id = 0
        #read or set initial eigen strain
        init_strain = np.copy(mat.get_eigen_strain(pixel_id))

        #update eigen strain
        stress = np.zeros((self.dim, self.dim))
        stress_1 = 0.3
        stress[0,1] = stress_1
        #TODO rleute
        # sps.update_eigen_strain(mat, pixel, stress, self.dim)

        # #read out updated eigen strain and proof
        # updated_strain = mat.get_eigen_strain(pixel)
        # analytic_strain = init_strain
        # analytic_strain[0,1] = \
        #     self.plastic_increment * stress_1 / np.sqrt(3/2 * stress_1**2)
        # self.assertLess(np.linalg.norm(updated_strain - analytic_strain), 1e-8)

        # #update eigen strain
        # stress = np.zeros((self.dim, self.dim))
        # stress_2 = 0.4
        # stress[1,0] = stress_2
        # sps.update_eigen_strain(mat, pixel, stress, self.dim)

        # #read out updated eigen strain and proof
        # updated_strain_2 = mat.get_eigen_strain(pixel)
        # analytic_strain = updated_strain
        # analytic_strain[1,0] = \
        #     self.plastic_increment * stress_2 / np.sqrt(3/2 * stress_2**2)
        # self.assertLess(
        #     np.linalg.norm(updated_strain_2 - analytic_strain), 1e-8)

    def test_set_new_threshold(self):
        cell, mat = init_material(self.res, self.lens, self.formulation,
                                  self.young, self.poisson, self.yield_stress,
                                  self.plastic_increment, self.eigen_strain)
        pixel = [0,1,2][:self.dim]
        pixel_id = 0
        # uniform distribution on the interval (a,b)
        a = 10
        b = 14
        inv_cum_dist_func = lambda z: a + (b-a)*z

        ### write first time a threshold on the pixel
        np.random.seed()
        seed = int(np.random.random())
        np.random.seed(seed)
        #TODO rleute
        # sps.set_new_threshold(mat, pixel_id
        #                       inverse_cumulative_dist_func = inv_cum_dist_func)
        # np.random.seed(seed)
        # threshold_expected = inv_cum_dist_func(np.random.random())
        # threshold_read = mat.get_stress_threshold(pixel)

        # self.assertLess(threshold_expected - threshold_read, 1e-8)

        # ### write second time a threshold on the pixel
        # np.random.seed()
        # seed = int(np.random.random())
        # np.random.seed(seed)
        # sps.set_new_threshold(mat, pixel,
        #                       inverse_cumulative_dist_func = inv_cum_dist_func)
        # np.random.seed(seed)
        # threshold_expected = inv_cum_dist_func(np.random.random())
        # threshold_read = mat.get_stress_threshold(pixel)

        # self.assertLess(threshold_expected - threshold_read, 1e-8)

    def test_propagate_avalanche_step(self):
        """
        Check if a single overloaded pixel breaks, at the right strain load.
        """
        strain_xy = 0.1
        weak_pixel = (0,)*self.dim

        ### analytic
        #analytic compute the equivalent stress for a given strain 'strain_xy'
        F = np.eye(self.dim)
        F[0,1] = strain_xy
        C = stiffness_matrix(self.young, self.poisson, self.dim)
        E = green_lagrangian_strain_vector(F, self.dim)
        PK2_analytic = PK2_tensor(C, E)
        PK1_analytic = np.dot(F, PK2_analytic)
        eq_stress = sigma_eq(PK1_analytic) #analytic computed equivalent stress

        ### numeric
        #set the analytic computed equivalent stress reduced by a tiny amount as
        #threshold for one "weak pixel" and the other thresholds to a little bit
        #higher values.
        fixed_yield_stress = np.ones(tuple(self.res)) * eq_stress * (1 + 1e-8)
        fixed_yield_stress[weak_pixel] = eq_stress * (1 - 1e-8)
        #init material
        cell, mat = init_material(self.res, self.lens, self.formulation,
                                  self.young, self.poisson, fixed_yield_stress,
                                  self.plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol,
                                   self.maxiter, self.verbose)
        #set the eigen strain field to the previous fixed deformation 'strain_xy'
        strain_field = cell.strain.array((self.dim, self.dim))
        for i in range(self.dim):
            strain_field[i,i,...] = 1.0
        strain_field[0,1,...] = strain_xy
        #check if you can find the yielding pixel
        overloaded_pixels = \
            sps.propagate_avalanche_step(mat, cell, self.dim, cg_solver,
                                         self.newton_tol, self.newton_equil_tol,
                                         self.verbose)

        #TODO rleute        # self.assertTrue(len(overloaded_pixels) == 1)
        # self.assertTrue(overloaded_pixels[0] == list(weak_pixel))

    def test_reshape_avalanche_history(self):
        """
        Reshapes the nested avalanche history list with nana
        """
        dim = 3
        ava_hist_list = [np.array([[13, 12, 6]]),
                         np.array([[13, 12, 6]]),
                         np.array([[13, 12, 6], [13, 13, 5]])]
        ava_ha = sps.reshape_avalanche_history(ava_hist_list, dim)

        expected = np.array([[[    13,     12,      6],
                              [np.nan, np.nan, np.nan]],
                             [[    13,     12,      6],
                              [np.nan, np.nan, np.nan]],
                             [[    13,     12,      6],
                              [    13,     13,      5]]])

        self.assertIsInstance(ava_ha, np.ndarray)
        self.assertTrue(((ava_ha == expected) |
                         (np.isnan(ava_ha) & np.isnan(expected))).all())

    def test_compute_average_deformation_gradient(self):
        """
        Tests if the "field average" is correctly computed.
        """
        for d in range(2,4):
            ### 1. ###
            one_field = np.ones((d,d) + (4,)*d)
            average = sps.compute_average_deformation_gradient(one_field, d)
            expected_average = np.ones((d,d))
            self.assertTrue((average == expected_average).all())

            ### 2. ###
            expected = np.arange(d**2).reshape((d,)*2)
            field = np.tensordot(expected, np.ones(([5,6,7][:d])), axes=0)
            np.random.seed(1568794)
            noise = (np.random.random((d,)*2 + tuple([5,6,7][:d])) - 0.5)*1e-3
            noise_field = field + noise
            average = sps.compute_average_deformation_gradient(noise_field, d)
            self.assertLess(np.linalg.norm(expected-average),
                            [1.6e-4,7.4e-5][d-2])

    def test_bracket_search(self):
        """
        Tests:
        1. Test if bracket search find the exact yield point for one pixel with
           a lower yield threshold than the others.
        2. Test exception for two/n pixels with very close yield criterion. Thus
           they should break together(hence avalanche can start for n>= 2 pixel)
        3. Test if an error is raised when the maximum allowed bracket steps are
           reached.
        """
        ### ------- 1. ------- ###
        # init data
        low_yield_stress = 10.0
        yield_surface_accuracy = 1e-7 #low accuracy for short test times
        n_max_bracket_search   = 4

        g_01 = 0.07268800332367435 #final deformation
        strain_init = np.array([[1, g_01 - yield_surface_accuracy*1.75, 0],
                                [0,  1  ,0],
                                [0,  0  ,1]])

        fixed_yield_stress = np.ones(tuple(self.res))*14 #high threshold
        fixed_yield_stress[0,0,0] = low_yield_stress
        cell, mat = init_material(self.res, self.lens, self.formulation,
                                  self.young, self.poisson, fixed_yield_stress,
                                  self.plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol,
                                   self.maxiter, self.verbose)
        DelF_initial = np.zeros((self.dim, self.dim))
        DelF_initial[0,1] = yield_surface_accuracy

        #initialize cell with unit-matrix deformation gradient
        cell_strain = cell.strain.array((self.dim, self.dim))
        np.squeeze(cell_strain)[:] = np.tensordot(
            strain_init,
            np.ones(cell.nb_subdomain_grid_pts),
            axes=0)

        #TODO(rleute) Bracket search forever
        # next_DelF_guess, PK2, F, breaking_pixel = \
        #     sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
        #                        self.newton_equil_tol, yield_surface_accuracy,
        #                        n_max_bracket_search, DelF_initial, self.verbose,
        #                        test_mode = True)

        # #Is it exactly one breaking pixel and if yes is it pixel [0,0,0]?
        # self.assertEqual(len(breaking_pixel), 1)
        # self.assertTrue((breaking_pixel[0] == [0, 0, 0]).all())

        # ### plug in the numeric result into the analytic formula and see if one
        # #   gets out the exact yield stress (low_yield_stress)
        # F_numeric = µ.gradient_integration.reshape_gradient(
        #     F, self.res)[(0,)*self.dim]
        # C = stiffness_matrix(self.young, self.poisson, self.dim)
        # E = green_lagrangian_strain_vector(F_numeric, self.dim)
        # PK2_analytic = PK2_tensor(C, E)
        # #TODO(RLeute): change to PK2 if material_stochastic_plasticity returns PK2
        # #eq_pk2    = sigma_eq(PK2_analytic)
        # #print("eq_stress PK2: ", eq_pk2)
        # PK1_analytic = np.dot(F_numeric, PK2_analytic)
        # eq_stress = sigma_eq(PK1_analytic)

        # #Is the analytic yield stress equivalent to yield stress of pix(0,0,0)?
        # self.assertLess(np.abs(low_yield_stress - eq_stress), 4e-3)

        # #Is the computed deformation gradient F_numeric correct?
        # F_yield10 = np.array([[1, g_01,0],
        #                       [0,  1  ,0],
        #                       [0,  0  ,1]])
        # self.assertLess(np.linalg.norm(F_yield10-F_numeric),
        #                   yield_surface_accuracy)

        # ### ------- 2. ------- ###
        # # init data
        # low_yield_stress = 10.0
        # yield_surface_accuracy = 1e-8 #low accuracy for short test times
        # small_yield_difference = \
        #             low_yield_stress * yield_surface_accuracy**2 * 1e-2
        # #set the initial deformation close to the final deformation to reduce
        # #the needed bracket search steps
        # g_01 = 0.07268800332367393591 #final deformation
        # strain_init = np.array([[1, g_01 - yield_surface_accuracy**2*1.75, 0],
        #                         [0,  1  ,0],
        #                         [0,  0  ,1]])
        # n_max_bracket_search   = 4

        # fixed_yield_stress = np.ones(tuple(self.res))*14 #high threshold
        # fixed_yield_stress[0,0,0] = low_yield_stress
        # fixed_yield_stress[tuple([i//2 for i in self.res])] = \
        #     low_yield_stress + small_yield_difference
        # cell, mat = init_material(self.res, self.lens, self.formulation,
        #                           self.young, self.poisson, fixed_yield_stress,
        #                           self.plastic_increment, self.eigen_strain)
        # cg_solver = init_cg_solver(cell, self.cg_tol,
        #                            self.maxiter, self.verbose)
        # DelF_initial = np.zeros((self.dim, self.dim))
        # DelF_initial[0,1] = yield_surface_accuracy**2

        # #initialize cell with deformation gradient for fast convergence
        # cell_strain = cell.strain.array((self.dim, self.dim))
        # np.squeeze(cell_strain)[:] = np.tensordot(strain_init,
        #                               np.ones(cell.nb_subdomain_grid_pts),
        #                               axes=0)

        # with warnings.catch_warnings(record=True) as w:
        #     warnings.simplefilter("always") #all warnings be triggered.
        #     next_DelF_guess, PK2, F, breaking_pixel = \
        #         sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
        #                            self.newton_equil_tol, yield_surface_accuracy,
        #                            n_max_bracket_search, DelF_initial,
        #                            self.verbose, test_mode = True)
        #     self.assertTrue(len(w) == 1)
        #     self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
        #     self.assertTrue("The avalanche starts with 2 initially "
        #                     "plastically deforming pixels, instead of starting "
        #                     "with one pixel!" == str(w[-1].message))
        # #Are there exactly two breaking pixels, [0,0,0] and [nx//2,ny//2,nz//2]?
        # self.assertEqual(len(breaking_pixel), 2)
        # self.assertTrue(
        #     (breaking_pixel == [[0,0,0], [i//2 for i in self.res]]).all())

        # ### ------- 3. ------- ###
        # #use the initalization from the last test
        # n_max_bracket_search = 2
        # DelF_initial = np.zeros((self.dim, self.dim))
        # DelF_initial[0,1] = 0.1
        # with self.assertRaises(RuntimeError):
        #     sps.bracket_search(mat, cell, cg_solver,
        #                        self.newton_tol, self.newton_equil_tol,
        #                        yield_surface_accuracy, n_max_bracket_search,
        #                        DelF_initial, self.verbose)

    def test_propagate_avalanche(self):
        """
        Test if plastic deformations are done in the right order and at the
        right place!
        """
        ### init parameters
        res  = [5,5,5]
        lens = [1,1,1]
        dim  = len(res)
        strain_xy_1 = 0.01
        plastic_increment = strain_xy_1 * 10
        expected_ava_history = np.array([[[ 3.,  3.,  3.],
                                          [np.nan, np.nan, np.nan]],
                                         [[ 2.,  2.,  3.],
                                          [ 4.,  4.,  3.]]])

        ### analytic compute eq_stress_1 for a given strain 'strain_xy_1'
        F = np.eye(self.dim)
        F[0,1] = strain_xy_1
        C = stiffness_matrix(self.young, self.poisson, self.dim)
        E = green_lagrangian_strain_vector(F, self.dim)
        PK2_analytic = PK2_tensor(C, E)
        PK1_analytic = np.dot(F, PK2_analytic)
        eq_stress = sigma_eq(PK1_analytic) #analytic computed equivalent stress

        eq_stress_1 = eq_stress
        eq_stress_2 = eq_stress * 1.05

        ### init material, with fine tuned order of stress thresholds
        fixed_yield_stress = np.ones(tuple(res))*17 #high threshold
        fixed_yield_stress[3,3,3] = eq_stress_1
        fixed_yield_stress[4,4,3] = eq_stress_2
        fixed_yield_stress[2,2,3] = eq_stress_2
        cell, mat = init_material(res, lens, self.formulation,
                                  self.young, self.poisson, fixed_yield_stress,
                                  plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol,
                                   self.maxiter, self.verbose)

        ### overload one pixel which breaks and by its plastic increment
        ### overloads two additional pixels.

        #propagate the avalanche
        yield_surface_accuracy = 1e-8
        n_max_bracket_search = 5
        #set the eigen strain field to the previous fixed deformation 'strain_xy'
        strain_field = cell.strain.array((self.dim, self.dim))
        for i in range(self.dim):
            strain_field[i,i,...] = 1.0
        strain_field[0,1,...] = strain_xy_1 - yield_surface_accuracy*1.25
        DelF_init = np.zeros((dim, dim))
        DelF_init[0,1] = yield_surface_accuracy
        n_max_avalanche = 10
        i_cdf = lambda z: 17 #constant value
        def save_and_test_ava(n_strain_loop, ava_history, PK2_initial,
                              F_initial, PK2_final, F_final, communicator):
            self.assertTrue(
                np.isclose(ava_history, expected_ava_history, equal_nan=True)
                .all())

        # TODO rleute (bracket search forever)
        # DelF, PK2, F, breaking_pixel = \
        #     sps.bracket_search(mat, cell, cg_solver, self.newton_tol,
        #                        self.newton_equil_tol, yield_surface_accuracy,
        #                        n_max_bracket_search, DelF_init, self.verbose,
        #                        test_mode = True)
        # #initial pixel is [3,3,3]
        # self.assertTrue((breaking_pixel == [[3,3,3]]).all())

        
        # sps.propagate_avalanche(mat, cell, cg_solver, self.newton_tol,
        #     self.newton_equil_tol, PK2, F, n_max_avalanche, self.verbose,
        #     inverse_cumulative_dist_func = i_cdf,
        #     save_avalanche = save_and_test_ava, n_strain_loop = 0)


    def test_strain_cell(self):
        """
        Tests:
        1. Test if the function reaches the required deformation
        2. Small deformation with only one avalanche. Check:
            - avalanche pixel index
            - PK2, stress field
            - F, deformation gradient field
        """
        ### ------- 1. ------- ###
        DelF      = np.zeros((self.dim,self.dim))
        DelF[0,1] = 0.0001
        F_tot      = np.eye(self.dim)
        F_tot[0,1] = 0.0002
        cell, mat = init_material(self.res, self.lens, self.formulation,
                                  self.young, self.poisson,
                                  self.yield_stress,
                                  self.plastic_increment, self.eigen_strain)
        cg_solver = init_cg_solver(cell, self.cg_tol, self.maxiter,
                                   self.verbose)
        #TODO rleute
        # F_fin = sps.strain_cell(mat, cell, cg_solver, self.newton_tol,
        #                         self.newton_equil_tol, DelF, F_tot,
        #                         self.yield_surface_accuracy,
        #                         self.n_max_strain_loop,
        #                         self.n_max_bracket_search,
        #                         self.n_max_avalanche, self.verbose,
        #                         self.inverse_cumulative_dist_func,
        #                         save_avalanche = None)
        # #is the reached deformation larger or almost equal to the required one
        # self.assertTrue(((F_fin-F_tot) > -1e-16).all())
