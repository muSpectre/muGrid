#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_strain_stress_control_test.py

@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   09 Jul 2021

@brief  Testing the outcome of mean strain/stress control in
        projection based solvers

Copyright © 2021 Till Junge

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
from python_test_imports import muSpectre
from muSpectre import cell


class StrainStressControlCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [11, 11]
        self.domain_lens = [7., 5.]
        self.center = np.array([r // 2 for r in self.nb_grid_pts])
        self.incl = self.nb_grid_pts[0] // 5
        self.cell_data_strain_control =\
            cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cell_data_stress_control =\
            cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cell_data_strain_control.nb_quad_pts = 1
        self.cell_data_stress_control.nb_quad_pts = 1
        self.cg_tol = 1e-8
        self.newton_tol = 1e-6
        self.equil_tol = 1e-10
        self.maxiter = self.cell_data_strain_control.spatial_dim * 20
        self.verbose = muSpectre.Verbosity.Silent
        self.del0 = 1e-6
        self.load_strain_control = np.array(
            [[self.del0/2.512, self.del0/6.157],
             [self.del0/6.157, self.del0/1.247]])
        self.control_strain_control = muSpectre.solvers.MeanControl.strain_control
        self.control_stress_control = muSpectre.solvers.MeanControl.stress_control
        self.formulation = muSpectre.Formulation.finite_strain
        self.E0 = 1.0e+1

    def test_strain_stress_control(self):
        for form in (muSpectre.Formulation.finite_strain,
                     muSpectre.Formulation.small_strain):
            if form == muSpectre.Formulation.finite_strain:
                strain_corrector = np.identity(2)
                runner_strain_stress_control(from, strain_corrector)
            else:
                strain_corrector = np.zeros(2)
                runner_strain_stress_control(from, strain_corrector)

    def runner_strain_stress_control(self, formulation, strain_corrector):
        hard_strain_control = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_strain_control, "hard", 10.*self.E0, .3)
        soft_strain_control = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_strain_control, "soft", 0.1*self.E0, .3)
        # making a ""circular"" inclusion
        for i, pixel in self.cell_data_strain_control.pixels.enumerate():
            if np.linalg.norm(self.center - np.array(pixel), 2) < self.incl:
                soft_strain_control.add_pixel(i)
            else:
                hard_strain_control.add_pixel(i)
        krylov_solver_strain_control = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                                 self.maxiter,
                                                                 self.verbose)
        newton_solver_strain_control = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_strain_control, krylov_solver_strain_control,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter,
            self.control_strain_control)
        newton_solver_strain_control.formulation = formulation
        newton_solver_strain_control.initialise_cell()

        newton_solver_strain_control_homo = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_strain_control, krylov_solver_strain_control,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter)
        newton_solver_strain_control_homo.formulation = formulation
        newton_solver_strain_control_homo.initialise_cell()

        result_strain_control = \
            newton_solver_strain_control.solve_load_increment(self.load_strain_control)

        C_eff_strain_control = newton_solver_strain_control_homo.compute_effective_stiffness()

        stress_strain_control = result_strain_control.stress
        grad_strain_control = result_strain_control.grad
        stress_strain_control = stress_strain_control.reshape(2, 2, *self.nb_grid_pts)
        grad_strain_control = grad_strain_control.reshape(2, 2, *self.nb_grid_pts)

        load_stress_control = stress_strain_control.mean(axis=(2, 3))
        hard_stress_control = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_stress_control, "hard", 10.*self.E0, .3)
        soft_stress_control = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_stress_control, "soft", 0.1*self.E0, .3)
        # making a ""circular"" inclusion
        for i, pixel in self.cell_data_stress_control.pixels.enumerate():
            if np.linalg.norm(self.center - np.array(pixel), 2) < self.incl:
                soft_stress_control.add_pixel(i)
            else:
                hard_stress_control.add_pixel(i)
        krylov_solver_stress_control = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                                  self.maxiter,
                                                                  self.verbose)
        newton_solver_stress_control = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_stress_control, krylov_solver_stress_control,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter,
            self.control_stress_control)
        newton_solver_stress_control.formulation = formulation
        newton_solver_stress_control.initialise_cell()

        newton_solver_stress_control_homo = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_stress_control, krylov_solver_stress_control,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter)

        newton_solver_stress_control_homo.formulation = formulation
        newton_solver_stress_control_homo.initialise_cell()

        result_stress_control = \
            newton_solver_stress_control.solve_load_increment(load_stress_control)

        C_eff_stress_control = newton_solver_stress_control_homo.compute_effective_stiffness()

        stress_stress_control = result_stress_control.stress
        grad_stress_control = result_stress_control.grad
        stress_stress_control = stress_stress_control.reshape(2, 2, *self.nb_grid_pts)
        grad_stress_control = grad_stress_control.reshape(2, 2, *self.nb_grid_pts)

        self.assertTrue(np.allclose(
            self.load_strain_control,
            (grad_stress_control.mean(axis=(2, 3))-strain_corrector), rtol=1e-15))

        self.assertTrue(np.allclose(
            C_eff_stress_control, C_eff_strain_control, rtol=1e-15))


class StressStrainControlCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [11, 11]
        self.domain_lens = [7., 5.]
        self.center = np.array([r // 2 for r in self.nb_grid_pts])
        self.incl = self.nb_grid_pts[0] // 5
        self.cell_data_strain_control =\
            cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cell_data_stress_control =\
            cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cell_data_strain_control.nb_quad_pts = 1
        self.cell_data_stress_control.nb_quad_pts = 1
        self.cg_tol = 1e-8
        self.newton_tol = 1e-6
        self.equil_tol = 1e-10
        self.maxiter = self.cell_data_strain_control.spatial_dim * 20
        self.verbose = muSpectre.Verbosity.Silent
        self.E0 = 1.0e+1
        self.del0 = 1e-6 * self.E0
        self.load_strain_control = np.array(
            [[self.del0/2.512, self.del0/6.157],
             [self.del0/6.157, self.del0/1.247]])
        self.control_strain_control = muSpectre.solvers.MeanControl.stress_control
        self.control_stress_control = muSpectre.solvers.MeanControl.strain_control
        self.formulation = muSpectre.Formulation.finite_strain

    def test_strain_stress_control(self):
        for form in (muSpectre.Formulation.finite_strain,
                     muSpectre.Formulation.small_strain):
            if form == muSpectre.Formulation.finite_strain:
                strain_corrector = np.identity(2)
                runner_strain_stress_control(from, strain_corrector)
            else:
                strain_corrector = np.zeros(2)
                runner_strain_stress_control(from, strain_corrector)

    def runner_stress_strain_control(self, formulation, strain_corrector):
        hard_strain_control = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_strain_control, "hard", 10.*self.E0, .3)
        soft_strain_control = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_strain_control, "soft", 0.1*self.E0, .3)
        # making a ""circular"" inclusion
        for i, pixel in self.cell_data_strain_control.pixels.enumerate():
            if np.linalg.norm(self.center - np.array(pixel), 2) < self.incl:
                soft_strain_control.add_pixel(i)
            else:
                hard_strain_control.add_pixel(i)
        krylov_solver_strain_control = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                                 self.maxiter,
                                                                 self.verbose)
        newton_solver_strain_control = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_strain_control, krylov_solver_strain_control,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter,
            self.control_strain_control)
        newton_solver_strain_control.formulation = formulation
        newton_solver_strain_control.initialise_cell()
        result_strain_control = \
            newton_solver_strain_control.solve_load_increment(self.load_strain_control)

        stress_strain_control = result_strain_control.stress
        grad_strain_control = result_strain_control.grad
        stress_strain_control = stress_strain_control.reshape(2, 2, *self.nb_grid_pts)
        grad_strain_control = grad_strain_control.reshape(2, 2, *self.nb_grid_pts)

        load_stress_control = grad_strain_control.mean(axis=(2, 3))-strain_corrector

        hard_stress_control = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_stress_control, "hard", 10.*self.E0, .3)
        soft_stress_control = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_stress_control, "soft", 0.1*self.E0, .3)
        # making a ""circular"" inclusion
        for i, pixel in self.cell_data_stress_control.pixels.enumerate():
            if np.linalg.norm(self.center - np.array(pixel), 2) < self.incl:
                soft_stress_control.add_pixel(i)
            else:
                hard_stress_control.add_pixel(i)
        krylov_solver_stress_control = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                                  self.maxiter,
                                                                  self.verbose)
        newton_solver_stress_control = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_stress_control, krylov_solver_stress_control,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter,
            self.control_stress_control)
        newton_solver_stress_control.formulation = formulation
        newton_solver_stress_control.initialise_cell()

        result_stress_control = \
            newton_solver_stress_control.solve_load_increment(load_stress_control)

        stress_stress_control = result_stress_control.stress
        grad_stress_control = result_stress_control.grad
        stress_stress_control = stress_stress_control.reshape(2, 2, *self.nb_grid_pts)
        grad_stress_control = grad_stress_control.reshape(2, 2, *self.nb_grid_pts)

        self.assertTrue(np.allclose(
            self.load_strain_control,
            (stress_stress_control.mean(axis=(2, 3))), rtol=1e-15))


if __name__ == '__main__':
    unittest.main()
