#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_sensitivity_analysis_test.py

@author Indre Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   22 Apr 2020

@brief  tests for the sensitivity analysis in file sensitivity_analysis.py

Copyright © 2020 Till Junge

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
import numpy as np

from python_test_imports import µ
from python_test_imports import muFFT
import muSpectre.sensitivity_analysis as sa

class SensitivityAnalysis_Check(unittest.TestCase):
    """
    Test the sensitivity analysis
    """
    #########
    # SetUp #
    #########
    def setUp(self):
        ### ----- Parameters ----- ###
        # Cell parameters
        self.nb_grid_pts = [3, 3]
        self.dim = len(self.nb_grid_pts)
        self.lengths = [1, 1]
        self.formulation = µ.Formulation.finite_strain

        # Material parameters
        self.Young1 = 10
        self.Poisson1 = 0.3
        self.Young2 = 30
        self.Poisson2 = 0.35

        # Material distribution
        x = np.linspace(0, self.lengths[0], self.nb_grid_pts[0], endpoint=False)
        x = x + 0.5 * self.lengths[0]/self.nb_grid_pts[0]
        self.phase = np.empty(self.nb_grid_pts)
        for j in range(self.nb_grid_pts[1]):
            self.phase[:, j] = 0.5*np.sin(2*np.pi/self.lengths[0]*x) + 0.5

        # Solver parameters
        self.newton_tol       = 1e-6
        self.cg_tol           = 1e-6 # tolerance for cg algo
        self.equil_tol        = 1e-6 # tolerance for equilibrium
        self.maxiter          = 100
        self.verbose          = µ.Verbosity.Silent

        # Macroscopic strain
        self.DelFs = [np.array([[1.1, 0], [0, 1]]),
                      np.array([[1, 0.1], [0, 1]])]
        self.DelF = [self.DelFs[0]]

        # Additional parameters for the aim function
        self.args = ()

        # Discrete Gradient
        self.gradient = [muFFT.Stencils2D.d_10_00, muFFT.Stencils2D.d_01_00,
                         muFFT.Stencils2D.d_11_01, muFFT.Stencils2D.d_11_10]

        ### ----- Aim functions ----- ###
        # Aim function = 1/Lx/Ly * int(stress_00)
        def aim_function(phase, strains, stresses, cell, args):
            stress = stresses[0]
            f = np.average(stress[0, 0])
            return f
        self.aim_function = aim_function

        # Partial derivatives of the aim function
        def dfdstrain(phase, strains, stress, cell, args):
            dim = cell.nb_domain_grid_pts.dim
            shape = [dim, dim, cell.nb_quad_pts, *cell.nb_domain_grid_pts]
            strain = strains[0].reshape(shape, order='F')
            stress, tangent = cell.evaluate_stress_tangent(strain)
            deriv = tangent[0, 0] / cell.nb_pixels / cell.nb_quad_pts
            return [deriv.flatten(order='F')]
        self.dfdstrain = dfdstrain

        def dfdphase(phase, strains, stresses, cell, Young, delta_Young,
                     Poisson, delta_Poisson, dstress_dphase, args):
            deriv = dstress_dphase[0][0, 0] / cell.nb_pixels
            deriv = np.average(deriv, axis=0)
            return deriv.flatten(order='F')
        self.dfdphase = dfdphase

        # Aim function = 1/Lx/Ly * int(stress1_00) + 1/Lx/Ly * int(stress2_00)
        def aim_function_2strains(phase, strains, stresses, cell, args):
            stress1 = stresses[0]
            stress2 = stresses[1]
            f = np.average(stress1[0, 0])
            f += np.average(stress2[0, 0])
            return f
        self.aim_function_2strains = aim_function_2strains

        # Partial derivatives of the second aim function
        def df2dstrain(phase, strains, stress, cell, args):
            dim = cell.nb_domain_grid_pts.dim
            shape = [dim, dim, cell.nb_quad_pts, *cell.nb_domain_grid_pts]
            # Derivative with respect to the first strain
            strain = strains[0].reshape(shape, order='F')
            stress1, tangent1 = cell.evaluate_stress_tangent(strain)
            deriv1 = tangent1[0, 0] / cell.nb_pixels / cell.nb_quad_pts
            # Derivative with respect to the second strain
            strain = strains[1].reshape(shape, order='F')
            stress2, tangent2 = cell.evaluate_stress_tangent(strain)
            deriv2 = tangent1[0, 0] / cell.nb_pixels / cell.nb_quad_pts
            return [deriv1.flatten(order='F'), deriv2.flatten(order='F')]
        self.df2dstrain = df2dstrain

        def df2dphase(phase, strains, stresses, cell, Young, delta_Young,
                      Poisson, delta_Poisson, dstress_dphase, args):
            deriv = dstress_dphase[0][0, 0] + dstress_dphase[1][0, 0]
            deriv = np.average(deriv, axis=0) / cell.nb_pixels
            return deriv.flatten(order='F')
        self.df2dphase = df2dphase

        ### ----- Equilibrium calculations ----- ###
        # Cell construction
        self.cell = µ.Cell(self.nb_grid_pts, self.lengths, self.formulation)
        phase = self.phase.flatten(order='F')
        delta_Young = self.Young2 - self.Young1
        delta_Poisson = self.Poisson2 - self.Poisson1
        Young = delta_Young*phase + self.Young1
        Poisson = delta_Poisson*phase + self.Poisson1
        mat = µ.material.MaterialLinearElastic4_2d.make(self.cell, "material")
        for pixel_id, pixel in self.cell.pixels.enumerate():
            mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

        # Solver
        self.krylov_solver = µ.solvers.KrylovSolverCG(self.cell, self.cg_tol,
                                                 self.maxiter, self.verbose)

        # Equilibrium calculations
        dim = self.dim
        self.strains_list = []
        self.stresses_list = []
        self.shape = [dim, dim, 1, *self.nb_grid_pts]
        #print('Equilibrium calculations (without discrete gradient)')
        for DelF in self.DelFs:
            result = µ.solvers.newton_cg(self.cell, DelF, self.krylov_solver,
                                     self.newton_tol, self.equil_tol,
                                     verbose=self.verbose)
            strain = result.grad.reshape(self.shape, order='F')
            self.strains_list.append(strain)
            stress = self.cell.evaluate_stress(strain).copy()
            self.stresses_list.append(stress)

        ### ----- Equilibrium calculations (fin. diff. gradient) ----- ###
        # Cell construction
        self.cell_gradient = µ.Cell(self.nb_grid_pts, self.lengths,
                                    self.formulation, self.gradient)
        mat = µ.material.MaterialLinearElastic4_2d.make(self.cell_gradient,
                                                        "material")
        for pixel_id, pixel in self.cell_gradient.pixels.enumerate():
            mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

        # Solver
        self.krylov_solver_gradient \
            = µ.solvers.KrylovSolverCG(self.cell_gradient, self.cg_tol,
                                       self.maxiter, self.verbose)

        # Equilibrium calculations
        self.strains_gradient_list = []
        self.stresses_gradient_list = []
        self.shape_gradient = [dim, dim, self.cell_gradient.nb_quad_pts,
                               *self.nb_grid_pts]
        #print('Equilibrium calculations (with discrete gradient)')
        for DelF in self.DelFs:
            result = µ.solvers.newton_cg(self.cell_gradient, DelF,
                                         self.krylov_solver_gradient,
                                         self.newton_tol, self.equil_tol,
                                         verbose=self.verbose)
            strain = result.grad.reshape(self.shape_gradient, order='F')
            self.strains_gradient_list.append(strain)
            stress = self.cell_gradient.evaluate_stress(strain).copy()
            self.stresses_gradient_list.append(stress)

    #############################
    # Test sensitivity analysis #
    #############################
    def test_sensitivity_analysis(self):
        """
        Test the sensitivity_analysis by comparison with finite differences.
        """
        strains = [self.strains_list[0]]
        stresses = [self.stresses_list[0]]

        ### ----- Sensitivity analysis with muSpectre ----- ###
        S_muSpectre \
            = sa.sensitivity_analysis(self.dfdstrain, self.dfdphase, self.phase,
                                      self.Young1, self.Poisson1, self.Young2,
                                      self.Poisson2, self.cell,
                                      self.krylov_solver, strains, stresses,
                                      args=self.args)

        ### ----- Sensitivity analysis with finite differences ----- ###
        delta_phase = 1e-6
        phase = self.phase.flatten(order='F')
        phase_dist = phase.copy()
        S_fin_diff = np.empty(S_muSpectre.size)

        # Initial aim function
        f_ini = self.aim_function(phase, strains, stresses, self.cell,
                                  self.args)

        for i in range(S_fin_diff.size):
            # Disturb phase
            phase_dist[i] += delta_phase

            # Disturbed aim function
            Young = (self.Young2 - self.Young1)*phase_dist + self.Young1
            Poisson = (self.Poisson2 - self.Poisson1)*phase_dist + self.Poisson1
            cell = µ.Cell(self.nb_grid_pts, self.lengths, self.formulation)
            solver = µ.solvers.KrylovSolverCG(cell, self.cg_tol, self.maxiter,
                                          self.verbose)
            mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
            for pixel_id, pixel in cell.pixels.enumerate():
                mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
            result = µ.solvers.newton_cg(cell, self.DelF[0], solver,
                                         self.newton_tol, self.equil_tol,
                                         verbose=self.verbose)
            strain = result.grad.reshape(self.shape, order='F')
            stress = cell.evaluate_stress(strain)
            f_dist = self.aim_function(phase_dist, [strain], [stress], cell,
                                       self.args)

            # Sensitivity at pixel i
            S_fin_diff[i] = (f_dist - f_ini) / delta_phase

            phase_dist[i] = phase_dist[i] - delta_phase

        ### ----- Comparison ----- ###
        self.assertTrue(np.allclose(S_muSpectre.flatten(order='F'), S_fin_diff))

        print()
        print('Test of sensitivity analysis done.')

    #########################################
    # Test sensitivity analysis (2 strains) #
    #########################################
    def test_sensitivity_analysis_2strains(self):
        """
        Test the sensitivity_analysis of an aim function depending on two
        strains by comparison with finite differences.
        """
        strains = self.strains_list
        stresses = self.stresses_list

        ### ----- Sensitivity analysis with muSpectre ----- ###
        S_muSpectre \
            = sa.sensitivity_analysis(self.df2dstrain, self.df2dphase,
                                      self.phase, self.Young1, self.Poisson1,
                                      self.Young2, self.Poisson2, self.cell,
                                      self.krylov_solver, strains, stresses,
                                      args=self.args)

        ### ----- Sensitivity analysis with finite differences ----- ###
        delta_phase = 1e-6
        phase = self.phase.flatten(order='F')
        phase_dist = phase.copy()
        S_fin_diff = np.empty(S_muSpectre.size)

        # Initial aim function
        f_ini = self.aim_function_2strains(phase, strains, stresses, self.cell,
                                           self.args)

        for i in range(S_fin_diff.size):
            # Disturb phase
            phase_dist[i] += delta_phase

            # Disturbed aim function
            Young = (self.Young2 - self.Young1)*phase_dist + self.Young1
            Poisson = (self.Poisson2 - self.Poisson1)*phase_dist + self.Poisson1
            cell = µ.Cell(self.nb_grid_pts, self.lengths, self.formulation)
            solver = µ.solvers.KrylovSolverCG(cell, self.cg_tol, self.maxiter,
                                          self.verbose)
            mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
            for pixel_id, pixel in cell.pixels.enumerate():
                mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
            result1 = µ.solvers.newton_cg(cell, self.DelFs[0], solver,
                                         self.newton_tol, self.equil_tol,
                                         verbose=self.verbose)
            result2 = µ.solvers.newton_cg(cell, self.DelFs[1], solver,
                                         self.newton_tol, self.equil_tol,
                                         verbose=self.verbose)
            strains_dist = [result1.grad, result2.grad]
            stresses_dist = []
            for strain_dist in strains_dist:
                strain_dist = strain_dist.reshape(self.shape, order='F')
                stress_dist = cell.evaluate_stress(strain_dist)
                stresses_dist.append(stress_dist.copy())
            f_dist = self.aim_function_2strains(phase_dist, strains_dist,
                                                stresses_dist, cell, self.args)

            # Sensitivity at pixel i
            S_fin_diff[i] = (f_dist - f_ini) / delta_phase

            phase_dist[i] = phase_dist[i] - delta_phase

        ### ----- Comparison ----- ###
        self.assertTrue(np.allclose(S_muSpectre.flatten(order='F'), S_fin_diff))

        print()
        print('Test of sensitivity analysis with aim function depending on ' +
              'two strains done.')

    #######################
    # Test dstress_dphase #
    #######################
    def test_dstress_dphase(self):
        """
        Test the calculation of the partial derivative of the stress with
        respect to the phase.
        """
        strains = [self.strains_list[0]]
        stresses = [self.stresses_list[0]]

        delta_Young = self.Young2 - self.Young1
        delta_Poisson = self.Poisson2 - self.Poisson1
        LinMat = µ.material.MaterialLinearElastic4_2d

        ### ----- Analytical calculation ----- ###
        Young = delta_Young*self.phase.flatten(order='F') + self.Young1
        Poisson = delta_Poisson*self.phase.flatten(order='F') + self.Poisson1
        dstress_dphase_ana = sa.calculate_dstress_dphase(self.cell, strains, Young,
                                                      delta_Young, Poisson,
                                                      delta_Poisson)[0]
        dstress_dphase_ana = dstress_dphase_ana.reshape((self.dim, self.dim, 1,
                                                 self.cell.nb_pixels), order='F')

        ### ----- Finite difference calculation ----- ###
        delta_rho = 1e-6
        # Initial stress
        stress_ini = stresses[0]
        for pixel_id in self.cell.pixel_indices:
            # Construct cell with disturbed material properties
            helper_cell = µ.Cell(self.nb_grid_pts, self.lengths,
                                 self.formulation)
            helper_mat = LinMat.make(helper_cell, "helper material")
            for iter_pixel, pixel in helper_cell.pixels.enumerate():
                if pixel_id == iter_pixel:
                    Young_dist = Young[pixel_id] + delta_Young*delta_rho
                    Poisson_dist = Poisson[pixel_id] + delta_Poisson*delta_rho
                    helper_mat.add_pixel(iter_pixel, Young_dist, Poisson_dist)
                else:
                    helper_mat.add_pixel(iter_pixel, Young[pixel_id],
                                         Poisson[pixel_id])

            # Disturbed stress
            stress_dist = helper_cell.evaluate_stress(strains[0])
            # Partial derivative dstress_dphase
            dstress_dphase_fd = (stress_dist - stress_ini)/delta_rho
            dstress_dphase_fd = dstress_dphase_fd.reshape(dstress_dphase_ana.shape,
                                                          order='F')

            ### ----- Comparison ----- ###
            self.assertTrue(np.allclose(dstress_dphase_ana[:, :, :, pixel_id],
                                        dstress_dphase_fd[:, :, :, pixel_id]))

        print('Test dstress_dphase is done.')
        print()

    ###################################
    # Test dstress_dphase (2 strains) #
    ###################################
    def test_dstress_dphase_two_strains(self):
        """
        Test the calculation of the partial derivative of the stress with
        respect to the phase for a list of two strains.
        """
        strains = self.strains_list
        stresses = self.stresses_list

        LinMat = µ.material.MaterialLinearElastic4_2d
        delta_Young = self.Young2 - self.Young1
        delta_Poisson = self.Poisson2 - self.Poisson1

        ### ----- Analytical calculation ----- ###
        Young = delta_Young*self.phase.flatten(order='F') + self.Young1
        Poisson = delta_Poisson*self.phase.flatten(order='F') + self.Poisson1
        dstress_dphase_ana \
            = sa.calculate_dstress_dphase(self.cell, strains, Young,
                                          delta_Young, Poisson, delta_Poisson)
        shape = [self.dim, self.dim, 1, self.cell.nb_pixels]
        dstress_0_dphase_ana = dstress_dphase_ana[0].reshape(shape, order='F')
        dstress_1_dphase_ana = dstress_dphase_ana[1].reshape(shape, order='F')

        ### ----- Finite difference calculation ----- ###
        delta_rho = 1e-6
        # Initial stresses
        stress_0_ini = stresses[0]
        stress_1_ini = stresses[1]
        for pixel_id in self.cell.pixel_indices:
            # Construct cell with disturbed material properties
            helper_cell = µ.Cell(self.nb_grid_pts, self.lengths,
                                 self.formulation)
            helper_mat = LinMat.make(helper_cell, "helper material")
            for iter_pixel, pixel in helper_cell.pixels.enumerate():
                if pixel_id == iter_pixel:
                    Young_dist = Young[pixel_id] + delta_Young*delta_rho
                    Poisson_dist = Poisson[pixel_id] + delta_Poisson*delta_rho
                    helper_mat.add_pixel(iter_pixel, Young_dist, Poisson_dist)
                else:
                    helper_mat.add_pixel(iter_pixel, Young[pixel_id],
                                         Poisson[pixel_id])

            # Disturbed stresses
            stress_0_dist = helper_cell.evaluate_stress(strains[0]).copy()
            stress_1_dist = helper_cell.evaluate_stress(strains[1]).copy()
            # Partial derivatives dstress_dphase
            dstress_0_dphase_fd = (stress_0_dist - stress_0_ini) / delta_rho
            dstress_1_dphase_fd = (stress_1_dist - stress_1_ini) / delta_rho
            dstress_0_dphase_fd = dstress_0_dphase_fd.reshape(shape, order='F')
            dstress_1_dphase_fd = dstress_1_dphase_fd.reshape(shape, order='F')

            ### ----- Comparison ----- ###
            self.assertTrue(np.allclose(dstress_0_dphase_ana[:, :, :, pixel_id],
                                        dstress_0_dphase_fd[:, :, :, pixel_id]))
            self.assertTrue(np.allclose(dstress_1_dphase_ana[:, :, :, pixel_id],
                                        dstress_1_dphase_fd[:, :, :, pixel_id]))

        print('Test dstress_dphase for two strains is done.')
        print()

    ########################################
    # Test Partial derivatives calculation #
    ########################################
    def test_partial_derivatives_finite_diff(self):
        """
        Test the calculation of partial derivatives with finite differences
        in muSpectre.
        """
        strains = [self.strains_list[0]]
        stresses = [self.stresses_list[0]]

        ### ----- Partial derivatives with muSpectre ----- ###
        krylov_solver_args = (self.cg_tol, self.maxiter, self.verbose)
        solver_args = (self.newton_tol, self.equil_tol, self.verbose)
        derivatives \
            = sa.partial_derivatives_finite_diff(self.aim_function, self.phase,
                                                 self.Young1, self.Poisson1,
                                                 self.Young2, self.Poisson2,
                                                 self.nb_grid_pts, self.lengths,
                                                 self.formulation, self.DelF,
                                                 krylov_solver_args=
                                                 krylov_solver_args,
                                                 solver_args=solver_args,
                                                 args=self.args, delta=1e-6)
        df_dstrain_fin_diff = derivatives[0][0]
        df_dphase_fin_diff = derivatives[1]

        ### ----- Analytical partial derivatives ----- ###
        delta_Young = self.Young2 - self.Young1
        delta_Poisson = self.Poisson2 - self.Poisson1
        Young = delta_Young*self.phase.flatten(order='F') + self.Young1
        Poisson = delta_Poisson*self.phase.flatten(order='F') + self.Poisson1
        df_dstrain_ana = self.dfdstrain(self.phase, strains, stresses,
                                        self.cell, self.args)
        df_dstrain_ana \
            = df_dstrain_ana[0].reshape([self.dim, self.dim, 1,
                                         *self.nb_grid_pts], order='F')

        dstress_dphase = sa.calculate_dstress_dphase(self.cell, strains, Young,
                                                         delta_Young, Poisson,
                                                         delta_Poisson)
        df_dphase_ana = self.dfdphase(self.phase, strains, stresses, self.cell,
                                      Young, delta_Young, Poisson,
                                      delta_Poisson, dstress_dphase, self.args)

        ### ----- Comparison ----- ###
        self.assertTrue(np.allclose(df_dstrain_fin_diff, df_dstrain_ana,
                                    atol=1e-5))
        self.assertTrue(np.allclose(df_dphase_ana, df_dphase_fin_diff))

        print('Test test_partial_deriv is done.')

    ####################################################
    # Test Partial derivatives calculation (2 strains) #
    ####################################################
    def test_partial_derivatives_finite_diff_two_strains(self):
        """
        Test if the calculation of partial derivatives with finite differences
        in muSpectre.
        """
        strains = self.strains_list
        stresses = self.stresses_list

        ### ----- Partial derivatives with muSpectre ----- ###
        krylov_solver_args = (self.cg_tol, self.maxiter, self.verbose)
        solver_args = (self.newton_tol, self.equil_tol, self.verbose)
        derivatives \
            = sa.partial_derivatives_finite_diff(self.aim_function_2strains,
                                                 self.phase,
                                                 self.Young1, self.Poisson1,
                                                 self.Young2, self.Poisson2,
                                                 self.nb_grid_pts, self.lengths,
                                                 self.formulation, self.DelFs,
                                                 krylov_solver_args=
                                                 krylov_solver_args,
                                                 solver_args=solver_args,
                                                 args=self.args, delta=1e-6)
        df_dstrain_0_fin_diff = derivatives[0][0]
        df_dstrain_1_fin_diff = derivatives[0][1]
        df_dphase_fin_diff = derivatives[1]

        ### ----- Analytical partial derivatives ----- ###
        shape = [self.dim, self.dim, 1, *self.nb_grid_pts]
        delta_Young = self.Young2 - self.Young1
        delta_Poisson = self.Poisson2 - self.Poisson1
        Young = delta_Young*self.phase.flatten(order='F') + self.Young1
        Poisson = delta_Poisson*self.phase.flatten(order='F') + self.Poisson1

        df_dstrain_ana = self.df2dstrain(self.phase, strains, stresses,
                                         self.cell, self.args)
        df_dstrain_0_ana = df_dstrain_ana[0].reshape(shape, order='F')
        df_dstrain_1_ana = df_dstrain_ana[1].reshape(shape, order='F')

        dstress_dphase = sa.calculate_dstress_dphase(self.cell, strains, Young,
                                                         delta_Young, Poisson,
                                                         delta_Poisson)
        df_dphase_ana = self.df2dphase(self.phase, strains, stresses, self.cell,
                                       Young, delta_Young, Poisson,
                                       delta_Poisson, dstress_dphase, self.args)

        ### ----- Comparison ----- ###
        self.assertTrue(np.allclose(df_dstrain_0_fin_diff, df_dstrain_0_ana,
                                    atol=1e-5))
        self.assertTrue(np.allclose(df_dstrain_1_fin_diff, df_dstrain_1_ana,
                                    atol=1e-5))
        self.assertTrue(np.allclose(df_dphase_ana, df_dphase_fin_diff))

        print('Test test_partial_deriv for two strains is done.')

    ####################################
    # Test dstress_dphase (2 quad_pts) #
    ####################################
    def test_dstress_dphase_two_quad_pts(self):
        """
        Test the calculation of the partial derivative of the stress with
        respect to the phase for two quadrature points.
        """
        strains = [self.strains_gradient_list[0]]
        stresses = [self.stresses_gradient_list[0]]

        nb_quad_pts = self.cell_gradient.nb_quad_pts
        shape = [self.dim, self.dim, nb_quad_pts, self.cell_gradient.nb_pixels]

        ### ----- Analytical calculation ----- ###
        delta_Young = self.Young2 - self.Young1
        delta_Poisson = self.Poisson2 - self.Poisson1
        Young = delta_Young*self.phase.flatten(order='F') + self.Young1
        Poisson = delta_Poisson*self.phase.flatten(order='F') + self.Poisson1
        dstress_dphase_ana \
            = sa.calculate_dstress_dphase(self.cell_gradient, strains, Young,
                                          delta_Young, Poisson, delta_Poisson,
                                          self.gradient)[0]

        dstress_dphase_ana = dstress_dphase_ana.reshape(shape, order='F')

        ### ----- Finite difference calculation ----- ###
        LinMat = µ.material.MaterialLinearElastic4_2d
        delta_rho = 1e-6
        # Initial stress
        stress_ini = stresses[0]
        for pixel_id in self.cell_gradient.pixel_indices:
            # Cell with disturbed material properties
            helper_cell = µ.Cell(self.nb_grid_pts, self.lengths,
                                 self.formulation, self.gradient)
            helper_mat = LinMat.make(helper_cell, "helper material")
            for iter_pixel, pixel in helper_cell.pixels.enumerate():
                if pixel_id == iter_pixel:
                    Young_dist = Young[pixel_id] + delta_Young*delta_rho
                    Poisson_dist = Poisson[pixel_id] + delta_Poisson*delta_rho
                    helper_mat.add_pixel(iter_pixel, Young_dist, Poisson_dist)
                else:
                    helper_mat.add_pixel(iter_pixel, Young[pixel_id],
                                         Poisson[pixel_id])

            # Disturbed stress
            stress_dist = helper_cell.evaluate_stress(strains[0])

            # Partial derivative
            dstress_dphase_fd = (stress_dist - stress_ini)/delta_rho
            dstress_dphase_fd = dstress_dphase_fd.reshape(shape, order='F')

            ### ----- Comparison ----- ###
            self.assertTrue(np.allclose(dstress_dphase_ana[:, :, :, pixel_id],
                                        dstress_dphase_fd[:, :, :, pixel_id]))

        print('Test dstress_dphase for two quadrature pixels is done.')
        print()

    #####################################################
    # Test Partial derivatives calculation (2 quad_pts) #
    #####################################################
    def test_partial_derivatives_finite_diff_gradient(self):
        """
        Test the calculation of partial derivatives with finite differences
        in muSpectre for two quadrature points.
        """
        strains = [self.strains_gradient_list[0]]
        stresses = [self.stresses_gradient_list[0]]

        ### ----- Partial derivatives with muSpectre ----- ###
        krylov_solver_args = (self.cg_tol, self.maxiter, self.verbose)
        solver_args = (self.newton_tol, self.equil_tol, self.verbose)
        derivatives \
            = sa.partial_derivatives_finite_diff(self.aim_function, self.phase,
                                                 self.Young1, self.Poisson1,
                                                 self.Young2, self.Poisson2,
                                                 self.nb_grid_pts, self.lengths,
                                                 self.formulation, self.DelF,
                                                 krylov_solver_args=
                                                 krylov_solver_args,
                                                 solver_args = solver_args,
                                                 args=self.args, delta=1e-6,
                                                 gradient = self.gradient)
        df_dstrain_fin_diff = derivatives[0][0]
        df_dphase_fin_diff = derivatives[1]

        ### ----- Analytical partial derivatives ----- ###
        delta_Young = self.Young2 - self.Young1
        delta_Poisson = self.Poisson2 - self.Poisson1
        Young = delta_Young*self.phase.flatten(order='F') + self.Young1
        Poisson = delta_Poisson*self.phase.flatten(order='F') + self.Poisson1
        df_dstrain_ana = self.dfdstrain(self.phase, strains, stresses,
                                        self.cell_gradient, self.args)
        shape = [self.dim, self.dim, self.cell_gradient.nb_quad_pts,
                 *self.nb_grid_pts]
        df_dstrain_ana \
            = df_dstrain_ana[0].reshape(shape, order='F')

        dstress_dphase \
            = sa.calculate_dstress_dphase(self.cell_gradient, strains, Young,
                                          delta_Young, Poisson, delta_Poisson,
                                          self.gradient)
        df_dphase_ana = self.dfdphase(self.phase, strains, stresses,
                                      self.cell_gradient, Young, delta_Young,
                                      Poisson, delta_Poisson, dstress_dphase,
                                      self.args)

        ### ----- Comparison ----- ###
        self.assertTrue(np.allclose(df_dstrain_fin_diff, df_dstrain_ana,
                                    atol=1e-5))
        self.assertTrue(np.allclose(df_dphase_ana, df_dphase_fin_diff))

        print('Test test_partial_deriv for two quad_ptsd is done.')


    ##########################################
    # Test sensitivity analysis (2 quad_pts) #
    ##########################################
    def test_sensitivity_analysis_gradient(self):
        """
        Test the sensitivity_analysis by comparison with finite differences
        for two quadrature points.
        """
        strains = [self.strains_gradient_list[0]]
        stresses = [self.stresses_gradient_list[0]]

        ### ----- Sensitivity analysis with muSpectre ----- ###
        S_muSpectre \
            = sa.sensitivity_analysis(self.dfdstrain, self.dfdphase, self.phase,
                                      self.Young1, self.Poisson1, self.Young2,
                                      self.Poisson2, self.cell_gradient,
                                      self.krylov_solver_gradient, strains,
                                      stresses, gradient=self.gradient,
                                      args=self.args)

        ### ----- Sensitivity analysis with finite differences ----- ###
        delta_phase = 1e-6
        phase = self.phase.flatten(order='F')
        phase_dist = phase.copy()
        S_fin_diff = np.empty(S_muSpectre.size)

        # Initial aim function
        f_ini = self.aim_function(phase, strains, stresses, self.cell,
                                  self.args)

        for i in range(S_fin_diff.size):
            # Disturb phase
            phase_dist[i] += delta_phase

            # Disturbed aim function
            Young = (self.Young2 - self.Young1)*phase_dist + self.Young1
            Poisson = (self.Poisson2 - self.Poisson1)*phase_dist + self.Poisson1
            cell = µ.Cell(self.nb_grid_pts, self.lengths, self.formulation)
            solver = µ.solvers.KrylovSolverCG(cell, self.cg_tol, self.maxiter,
                                          self.verbose)
            mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
            for pixel_id, pixel in cell.pixels.enumerate():
                mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
            result = µ.solvers.newton_cg(cell, self.DelF[0], solver,
                                         self.newton_tol, self.equil_tol,
                                         verbose=self.verbose)
            strain = result.grad.reshape(self.shape, order='F')
            stress = cell.evaluate_stress(strain)
            f_dist = self.aim_function(phase_dist, [strain], [stress], cell,
                                       self.args)

            # Sensitivity at pixel i
            S_fin_diff[i] = (f_dist - f_ini) / delta_phase

            phase_dist[i] = phase_dist[i] - delta_phase

        ### ----- Comparison ----- ###
        self.assertTrue(np.allclose(S_muSpectre.flatten(order='F'), S_fin_diff))

        print()
        print('Test of sensitivity analysis done.')
