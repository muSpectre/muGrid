#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_effective_tangent_method_vs_FD.py

@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   17 Feb 2022

@brief  Comparison of the compute_effective_stiffness method vs.
 finite difference approximation of the stiffness matrix of
 very small simplistic RVE


Copyright © 2022 Ali Falsafi

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


class SolverClassEffectiveStiffnessCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 3]
        self.domain_lens = [1.4, 2.4]
        self.cell_data = cell.CellData.make(self.nb_grid_pts, self.domain_lens)
        self.cg_tol = 1e-8
        self.newton_tol = 1e-6
        self.equil_tol = 1e-10
        self.maxiter = self.cell_data.spatial_dim * 50
        self.verbose = muSpectre.Verbosity.Full
        self.cell_data.nb_quad_pts = 1

    def setUp_mechanics(self):
        self.Mat = muSpectre.material.MaterialLinearElastic1_2d
        self.young_soft = 4
        self.young_hard = 8
        self.poisson = .3
        self.soft = self.Mat.make(self.cell_data,
                                  "soft", self.young_soft, self.poisson)
        self.hard = self.Mat.make(self.cell_data,
                                  "hard", self.young_hard, self.poisson)
        self.setUp_geometry()

    def setUp_geometry(self):
        counter = 0
        for i, pixel in self.cell_data.pixels.enumerate():
            counter = counter + 1
            if counter < 2:
                self.hard.add_pixel(i)
            else:
                self.soft.add_pixel(i)

    def test_mechanics(self):
        self.setUp_mechanics()
        krylov_solver = muSpectre.solvers.KrylovSolverCG(self.cg_tol,
                                                         self.maxiter,
                                                         self.verbose)
        solver = muSpectre.solvers.SolverNewtonCG(self.cell_data,
                                                  krylov_solver,
                                                  self.verbose,
                                                  self.newton_tol,
                                                  self.equil_tol,
                                                  self.maxiter)
        solver.formulation = muSpectre.Formulation.small_strain
        load_step = np.array([[1.3423423e-4, 3.24231674e-5],
                              [3.2423167e-5, 8.37461376e-5]])
        result = solver.solve_load_increment(load_step)
        self.assertTrue(result.success)

        mean_stress = solver.flux.map.mean()
        C_eff = solver.compute_effective_stiffness()
        mean_stress_eff = np.matmul(C_eff,
                                    load_step.reshape(-1, 1)).reshape(2, 2)
        self.assertTrue(np.allclose(mean_stress,
                                    mean_stress_eff,
                                    rtol=1e-12,
                                    atol=1e-12))


if __name__ == '__main__':
    unittest.main()
