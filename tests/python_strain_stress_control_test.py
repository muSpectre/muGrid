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
        self.cell_data_forward =\
            cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cell_data_backward =\
            cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cell_data_forward.nb_quad_pts = 1
        self.cell_data_backward.nb_quad_pts = 1
        self.cg_tol = 1e-8
        self.newton_tol = 1e-6
        self.equil_tol = 1e-10
        self.maxiter = self.cell_data_forward.spatial_dim * 20
        self.verbose = muSpectre.Verbosity.Silent
        self.del0 = 1e-6
        self.load_forward = np.array(
            [[self.del0/2.512, self.del0/6.157],
             [self.del0/6.157, self.del0/1.247]])
        self.control_forward = muSpectre.solvers.MeanControl.strain_control
        self.control_backward = muSpectre.solvers.MeanControl.stress_control
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
        hard_forward = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_forward, "hard", 10.*self.E0, .3)
        soft_forward = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_forward, "soft", 0.1*self.E0, .3)
        # making a ""circular"" inclusion
        for i, pixel in self.cell_data_forward.pixels.enumerate():
            if np.linalg.norm(self.center - np.array(pixel), 2) < self.incl:
                soft_forward.add_pixel(i)
            else:
                hard_forward.add_pixel(i)
        krylov_solver_forward = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                                 self.maxiter,
                                                                 self.verbose)
        newton_solver_forward = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_forward, krylov_solver_forward,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter,
            self.control_forward)
        newton_solver_forward.formulation = formulation
        newton_solver_forward.initialise_cell()
        result_forward = \
            newton_solver_forward.solve_load_increment(self.load_forward)

        stress_forward = result_forward.stress
        grad_forward = result_forward.grad
        stress_forward = stress_forward.reshape(2, 2, *self.nb_grid_pts)
        grad_forward = grad_forward.reshape(2, 2, *self.nb_grid_pts)

        load_backward = stress_forward.mean(axis=(2, 3))
        hard_backward = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_backward, "hard", 10.*self.E0, .3)
        soft_backward = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_backward, "soft", 0.1*self.E0, .3)
        # making a ""circular"" inclusion
        for i, pixel in self.cell_data_backward.pixels.enumerate():
            if np.linalg.norm(self.center - np.array(pixel), 2) < self.incl:
                soft_backward.add_pixel(i)
            else:
                hard_backward.add_pixel(i)
        krylov_solver_backward = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                                  self.maxiter,
                                                                  self.verbose)
        newton_solver_backward = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_backward, krylov_solver_backward,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter,
            self.control_backward)
        newton_solver_backward.formulation = formulation
        newton_solver_backward.initialise_cell()

        result_backward = \
            newton_solver_backward.solve_load_increment(load_backward)

        stress_backward = result_backward.stress
        grad_backward = result_backward.grad
        stress_backward = stress_backward.reshape(2, 2, *self.nb_grid_pts)
        grad_backward = grad_backward.reshape(2, 2, *self.nb_grid_pts)

        self.assertTrue(np.allclose(
            self.load_forward,
            (grad_backward.mean(axis=(2, 3))-strain_corrector), rtol=1e-15))


class StressStrainControlCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [11, 11]
        self.domain_lens = [7., 5.]
        self.center = np.array([r // 2 for r in self.nb_grid_pts])
        self.incl = self.nb_grid_pts[0] // 5
        self.cell_data_forward =\
            cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cell_data_backward =\
            cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cell_data_forward.nb_quad_pts = 1
        self.cell_data_backward.nb_quad_pts = 1
        self.cg_tol = 1e-8
        self.newton_tol = 1e-6
        self.equil_tol = 1e-10
        self.maxiter = self.cell_data_forward.spatial_dim * 20
        self.verbose = muSpectre.Verbosity.Silent
        self.E0 = 1.0e+1
        self.del0 = 1e-6 * self.E0
        self.load_forward = np.array(
            [[self.del0/2.512, self.del0/6.157],
             [self.del0/6.157, self.del0/1.247]])
        self.control_forward = muSpectre.solvers.MeanControl.stress_control
        self.control_backward = muSpectre.solvers.MeanControl.strain_control
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
        hard_forward = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_forward, "hard", 10.*self.E0, .3)
        soft_forward = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_forward, "soft", 0.1*self.E0, .3)
        # making a ""circular"" inclusion
        for i, pixel in self.cell_data_forward.pixels.enumerate():
            if np.linalg.norm(self.center - np.array(pixel), 2) < self.incl:
                soft_forward.add_pixel(i)
            else:
                hard_forward.add_pixel(i)
        krylov_solver_forward = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                                 self.maxiter,
                                                                 self.verbose)
        newton_solver_forward = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_forward, krylov_solver_forward,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter,
            self.control_forward)
        newton_solver_forward.formulation = formulation
        newton_solver_forward.initialise_cell()
        result_forward = \
            newton_solver_forward.solve_load_increment(self.load_forward)

        stress_forward = result_forward.stress
        grad_forward = result_forward.grad
        stress_forward = stress_forward.reshape(2, 2, *self.nb_grid_pts)
        grad_forward = grad_forward.reshape(2, 2, *self.nb_grid_pts)

        load_backward = grad_forward.mean(axis=(2, 3))-strain_corrector

        hard_backward = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_backward, "hard", 10.*self.E0, .3)
        soft_backward = muSpectre.material.MaterialLinearElastic1_2d.make(
            self.cell_data_backward, "soft", 0.1*self.E0, .3)
        # making a ""circular"" inclusion
        for i, pixel in self.cell_data_backward.pixels.enumerate():
            if np.linalg.norm(self.center - np.array(pixel), 2) < self.incl:
                soft_backward.add_pixel(i)
            else:
                hard_backward.add_pixel(i)
        krylov_solver_backward = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                                  self.maxiter,
                                                                  self.verbose)
        newton_solver_backward = muSpectre.solvers.SolverNewtonCG(
            self.cell_data_backward, krylov_solver_backward,
            self.verbose, self.newton_tol,
            self.equil_tol, self.maxiter,
            self.control_backward)
        newton_solver_backward.formulation = formulation
        newton_solver_backward.initialise_cell()

        result_backward = \
            newton_solver_backward.solve_load_increment(load_backward)

        stress_backward = result_backward.stress
        grad_backward = result_backward.grad
        stress_backward = stress_backward.reshape(2, 2, *self.nb_grid_pts)
        grad_backward = grad_backward.reshape(2, 2, *self.nb_grid_pts)

        self.assertTrue(np.allclose(
            self.load_forward,
            (stress_backward.mean(axis=(2, 3))), rtol=1e-15))


if __name__ == '__main__':
    unittest.main()
