#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_solver_test.py

@author Till Junge <till.junge@epfl.ch>

@date   22 Nov 2019

@brief  Unit tests for python bindings

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

import unittest
import numpy as np
from python_test_imports import µ


class SolverCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [5, 7]  # [5,7]
        self.lengths = [3., 3.]  # [5.2, 8.3]
        self.formulation = µ.Formulation.finite_strain
        try:
            from mpi4py import MPI
            self.cell = µ.Cell(self.nb_grid_pts,
                               self.lengths,
                               self.formulation,
                               fft='mpi',
                               communicator=MPI.COMM_WORLD)
        except ImportError:
            self.cell = µ.Cell(self.nb_grid_pts,
                               self.lengths,
                               self.formulation)
        self.hard = µ.material.MaterialLinearElastic1_2d.make(
            self.cell, "hard", 210e9, .33)
        self.soft = µ.material.MaterialLinearElastic1_2d.make(
            self.cell, "soft",  70e9, .33)

    def test_solve(self):
        for pix_id, (pix_x, pix_y) in enumerate(self.cell.pixels):
            if pix_y < 3:
                self.hard.add_pixel(pix_id)
            else:
                self.soft.add_pixel(pix_id)

        self.cell.initialise()
        cg_tol = 1e-8
        newton_tol = 1e-6
        equil_tol = 0.
        Del0 = np.array([[0, .1],
                         [0,  0]])
        maxiter = 100
        verbose = µ.Verbosity.Silent

        P, K = self.cell.evaluate_stress_tangent(self.cell.strain)

        solver = µ.solvers.KrylovSolverCG(self.cell, cg_tol, maxiter, verbose)

        for n in range(2):
            µ.solvers.newton_cg(
                self.cell, Del0, solver, newton_tol, equil_tol, verbose,
                IsStrainInitialised=μ.solvers.IsStrainInitialised.No
                if n == 0 else μ.solvers.IsStrainInitialised.Yes)

            mean_strain = self.cell.communicator.sum(
                np.sum(self.cell.strain.array(), axis=(2, 3, 4))) / \
                          np.prod(self.cell.nb_domain_grid_pts)
            self.assertTrue(np.allclose(mean_strain, np.eye(2) + (n+1)*Del0))


if __name__ == '__main__':
    unittest.main()
