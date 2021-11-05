#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_solverclass_test.py

@author Till Junge <till.junge@altermail.ch>

@date   04 Sep 2020

@brief  test for the class solvers

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
from python_test_imports import muSpectre
from muSpectre import cell


class SolverClassCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 4]
        self.domain_lens = [1.4, 2.4]
        self.cell_data = cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cg_tol = 1e-8
        self.newton_tol = 1e-6
        self.equil_tol = 1e-10
        self.maxiter = self.cell_data.spatial_dim * 20
        self.verbose = muSpectre.Verbosity.Silent
        self.fem_stencil = muSpectre.FEMStencil.bilinear_quadrangle(
            self.cell_data)
        self.discretisation = muSpectre.Discretisation(self.fem_stencil)

    def setUp_mechanics(self):
        self.Mat = muSpectre.material.MaterialLinearElastic1_2d
        self.young_soft = 4
        self.young_hard = 8
        self.poisson = .3
        self.soft = self.Mat.make(self.cell_data,
                                  "soft", self.young_soft, self.poisson)
        self.hard = self.Mat.make(self.cell_data,
                                  "hard", self.young_hard, self.poisson)
        self.nb_hard = self.cell_data.nb_domain_grid_pts[0]
        for index in self.cell_data.pixel_indices:
            if self.nb_hard > 0:
                self.hard.add_pixel(index)
                self.nb_hard -= 1
            else:
                self.soft.add_pixel(index)

    def test_pcg_mechanics(self):
        self.setUp_mechanics()
        krylov_solver = muSpectre.solvers.KrylovSolverPCG(self.cg_tol,
                                                          self.maxiter,
                                                          self.verbose)
        solver = muSpectre.solvers.SolverFEMNewtonPCG(self.discretisation,
                                                      krylov_solver,
                                                      self.verbose,
                                                      self.newton_tol,
                                                      self.equil_tol,
                                                      self.maxiter)
        solver.formulation = muSpectre.Formulation.finite_strain
        solver.initialise_cell()
        grad = solver.eval_grad.field.array()
        # set eval_grad to identity matrix
        grad[:] = np.eye(2, 2)[:, :, np.newaxis, np.newaxis, np.newaxis]
        solver.evaluate_stress_tangent()
        reference_material = solver.tangent.map.mean()
        solver.set_reference_material(reference_material)

        load_step = np.array([[1., 0.],
                              [0., 2.]])
        result = solver.solve_load_increment(load_step)
        self.assertTrue(result.success)

    def test_cg_mechanics(self):
        self.setUp_mechanics()
        krylov_solver = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                         self.maxiter,
                                                         self.verbose)
        solver = muSpectre.solvers.SolverFEMNewtonCG(self.discretisation,
                                                     krylov_solver,
                                                     self.verbose,
                                                     self.newton_tol,
                                                     self.equil_tol,
                                                     self.maxiter)
        solver.formulation = muSpectre.Formulation.finite_strain
        # solver.initialise_cell()
        # grad = solver.eval_grad.field.array()
        # # set eval_grad to identity matrix
        # grad[:] = np.eye(2, 2)[:, :, np.newaxis, np.newaxis, np.newaxis]
        # solver.evaluate_stress_tangent()
        # reference_material = solver.tangent.map.mean()
        # solver.set_reference_material(reference_material)

        load_step = np.array([[1., 0.],
                              [0., 2.]])
        result = solver.solve_load_increment(load_step)
        self.assertTrue(result.success)

    def setUp_diffusion(self):
        self.Mat = muSpectre.material.MaterialLinearDiffusion_2d
        self.diffusion_coeff_soft = np.array([[1.2, 0.3],
                                              [0.3, 2.1]])
        self.diffusion_coeff_hard = 2 * self.diffusion_coeff_soft
        self.soft = self.Mat.make(self.cell_data,
                                  "soft", self.diffusion_coeff_soft)
        self.hard = self.Mat.make(self.cell_data,
                                  "hard", self.diffusion_coeff_hard)
        self.nb_hard = self.cell_data.nb_domain_grid_pts[0]
        for index in self.cell_data.pixel_indices:
            if self.nb_hard > 0:
                self.hard.add_pixel(index)
                self.nb_hard -= 1
            else:
                self.soft.add_pixel(index)

    def test_pcg_diffusion(self):
        self.setUp_diffusion()
        krylov_solver = muSpectre.solvers.KrylovSolverPCG(self.cg_tol,
                                                          self.maxiter,
                                                          self.verbose)
        solver = muSpectre.solvers.SolverFEMNewtonPCG(self.discretisation,
                                                      krylov_solver,
                                                      self.verbose,
                                                      self.newton_tol,
                                                      self.equil_tol,
                                                      self.maxiter)
        solver.initialise_cell()
        solver.evaluate_stress_tangent()
        reference_material = solver.tangent.map.mean()
        solver.set_reference_material(reference_material)

        load_step = np.array([[1.],
                              [0.]])
        result = solver.solve_load_increment(load_step)
        self.assertTrue(result.success)

    def test_cg_diffusion(self):
        self.setUp_diffusion()
        krylov_solver = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                         self.maxiter,
                                                         self.verbose)
        solver = muSpectre.solvers.SolverFEMNewtonCG(self.discretisation,
                                                     krylov_solver,
                                                     self.verbose,
                                                     self.newton_tol,
                                                     self.equil_tol,
                                                     self.maxiter)

        load_step = np.array([[1.],
                              [0.]])
        result = solver.solve_load_increment(load_step)
        self.assertTrue(result.success)


if __name__ == '__main__':
    unittest.main()
