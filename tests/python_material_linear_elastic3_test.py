#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_material_linear_elastic3_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   20 Feb 2018

@brief  description

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

class MaterialLinearElastic3_Check(unittest.TestCase):
    """
    Check the implementation of the fourth order stiffness tensor C for each
    cell. Assign the same Youngs modulus and Poisson ratio to each cell,
    calculate the stress and compare the result with stress=2*mu*Del0
    (Hooke law for small symmetric strains).
    """
    def setUp(self):
        self.nb_grid_pts = [5,5]
        self.lengths = [2.5, 3.1]
        self.formulation = µ.Formulation.small_strain
        self.cell = µ.Cell(self.nb_grid_pts,
                          self.lengths,
                          self.formulation)
        self.dim = len(self.lengths)
        self.mat = µ.material.MaterialLinearElastic3_2d.make(
            self.cell, "material")

    def test_solver(self):
        Young   = 10.
        Poisson = 0.3

        for pixel_id in self.cell.pixel_indices:
            self.mat.add_pixel(pixel_id, Young, Poisson)

        self.cell.initialise()
        tol = 1e-6
        Del0 = np.array([[0, 0.025],
                         [0.025,  0]])
        maxiter = 100
        verbose = µ.Verbosity.Silent

        solver=µ.solvers.KrylovSolverCG(self.cell, tol, maxiter, verbose)
        r = µ.solvers.newton_cg(self.cell, Del0,
                                solver, tol, tol, verbose)
        print('Solver has successfully been called')

        #compare the computed stress with the trivial by hand computed stress
        mu = (Young/(2*(1+Poisson)))
        stress = 2*mu*Del0

        self.assertLess(np.linalg.norm(r.stress.reshape(-1, self.dim**2) -
                                       stress.reshape(1, self.dim**2)), 1e-8)

if __name__ == '__main__':
    unittest.main()
