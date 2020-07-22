#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_material_linear_elastic4_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   27 Mar 2018

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


class MaterialLinearElastic4_Check(unittest.TestCase):
    """
    Check the implementation of storing the first and second Lame constant in
    each cell. Assign the same Youngs modulus and Poisson ratio to each cell,
    from which the two Lame constants are internally computed. Then calculate
    the stress and compare the result with stress=2*mu*Del0 (Hooke law for small
    symmetric strains).
    """

    def setUp(self):
        self.nb_grid_pts = [7, 7]
        self.lengths = [2.3, 3.9]
        self.formulation = µ.Formulation.small_strain
        self.dim = len(self.lengths)

    def test_solver(self):
        Youngs_modulus = 10.
        Poisson_ratio = 0.3

        cell = µ.Cell(self.nb_grid_pts, self.lengths, self.formulation)
        mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")

        for i in cell.pixel_indices:
            mat.add_pixel(i, Youngs_modulus, Poisson_ratio)

        cell.initialise()
        tol = 1e-6
        Del0 = np.array([[0, 0.025],
                         [0.025,  0]])
        maxiter = 100
        verbose = µ.Verbosity.Silent

        solver = µ.solvers.KrylovSolverCG(
            cell, tol, maxiter, verbose)
        r = µ.solvers.newton_cg(
            cell, Del0, solver, tol, tol, verbose)

        # compare the computed stress with the trivial by hand computed stress
        mu = (Youngs_modulus/(2*(1+Poisson_ratio)))
        stress = 2*mu*Del0

        self.assertLess(np.linalg.norm(r.stress.reshape(-1, self.dim**2) -
                                       stress.reshape(1,self.dim**2)), 1e-8)

    def test_tangent(self):
        Youngs_modulus = 10.*(1 + 0.1*np.random.random(np.prod(self.nb_grid_pts)))
        Poisson_ratio  = 0.3*(1 + 0.1*np.random.random(np.prod(self.nb_grid_pts)))

        cell = µ.Cell(self.nb_grid_pts, self.lengths, self.formulation)
        mat = µ.material.MaterialLinearElastic4_2d.make(cell,
                                                        "material")

        for i in cell.pixel_indices:
            mat.add_pixel(i, Youngs_modulus[i], Poisson_ratio[i])

        cell.initialise()
        tol = 1e-6
        Del0 = np.array([[0.1, 0.05],
                         [0.05,  -0.02]])
        maxiter = 100
        verbose = µ.Verbosity.Silent

        solver = µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
        r = µ.solvers.newton_cg(cell, Del0,
                                solver, tol, tol, verbose)

        ### Compute tangent through a finite differences approximation

        F = cell.strain.array()
        stress, tangent = cell.evaluate_stress_tangent(F)

        numerical_tangent = np.zeros_like(tangent)

        eps = 1e-4
        for i in range(2):
            for j in range(2):
                F[i, j] += eps
                stress_plus = cell.evaluate_stress(F).copy()
                F[i, j] -= 2*eps
                stress_minus = cell.evaluate_stress(F).copy()
                F[i, j] += eps
                numerical_tangent[i, j] = (stress_plus - stress_minus)/(2*eps)

        self.assertTrue(np.allclose(tangent, numerical_tangent))

if __name__ == '__main__':
    unittest.main()
